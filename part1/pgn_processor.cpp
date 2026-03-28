/*
Go through the pgn text files
create 16 threads where each thread reads the pgn,
generate board state using bitboards
make sure to flip to black
flatten results

return binary byte file

In this case, each pgn file has thousands of games and we only have 4 files
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <thread>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include "../utils/utils.h"
#include "../utils/utils.cpp"
#include "stats.h"
#include "stats.cpp"


/*
we assume we have enough RAM to fit the whole dataset

k threads open k pgn files. They read into k individual vectors so everything is in RAM. 
Track the size of the actual data.

16 threads, assign 16 / k files = number of processors per file

the original k threads jumps 1/(16 % k)th step and marks these windows where each jump reads forward and starts
at the first [event] label

Given now bounded segments, create std::string_view objects which are windows into the RAM buffer

16 threads all wake up and given its assigned chunk, process the games. Each thread has its own board. To do flips,
you can just flip the bits algorithm. 

These 16 threads write to their respective binary output files so we'll have chunk_0 -> chunk_15. 
*/


// parsing logic
// possible tokens to encounter: the whole tag, the number "1.", a move "Nf3", or game result
std::vector<std::string_view> extract_moves(std::string_view game_text) {
    std::vector<std::string_view> clean_moves;
    clean_moves.reserve(100); // most games are under 100 moves

    size_t pos = 0;
    size_t length = game_text.length();

    while (pos < length) {

        // skip the whitespace
        while (pos < length && std::isspace(game_text[pos])) {
            pos++;
        }

        if (pos < length && game_text[pos] == '[') {
            while (pos < length && game_text[pos] != '\n') {
                pos++;
            }
        } else {
            break;
        }
    }

    while (pos < length) {
        while (pos < length && std::isspace(game_text[pos])) {
            pos++;
        }
        if (pos >= length) {
            break;
        }

        size_t end_pos = pos;
        while (end_pos < length && !std::isspace(game_text[end_pos])) {
            end_pos++;
        }

        std::string_view token = game_text.substr(pos, end_pos - pos);
        pos = end_pos;

        // disregard move numbers like "1." or "12."
        if (token.find('.') != std::string_view::npos) {
            continue;
        }

        // get rid of game results
        if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") {
            continue;
        }

        clean_moves.push_back(token);
    }
    return clean_moves;
}


// read file function
std::string read_file(const std::string& filepath) {

    // open file in binary mode
    std::ifstream in(filepath, std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open: " << filepath << "\n";
        return "";
    }

    // look at the end to see the size of file and resize the string buffer to match it
    in.seekg(0, std::ios::end);
    std::string contents;
    contents.resize(in.tellg());

    // read everything
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    
    return contents;
}

// create the job for the worker thread, takes the id and the windows to the chunks it's responsible for
void worker_thread(int thread_id, const std::vector<std::string_view>& chunks, ThreadStats* my_stats) {

    auto t_start = std::chrono::steady_clock::now();
    my_stats->thread_id = thread_id;
    my_stats->chunks_assigned = chunks.size();

    // create the binary file
    std::string out_filepath = "chunk_" + std::to_string(thread_id) + ".bin";
    std::ofstream out_bin(out_filepath, std::ios::binary | std::ios::trunc); 

    for (std::string_view chunk : chunks) {
        my_stats->bytes_input += chunk.size();
        size_t curr_pos = 0;
        const std::string_view event_tag = "[Event ";

        while (curr_pos < chunk.length()) {

            // get this game start
            size_t game_start = chunk.find(event_tag, curr_pos);
            if (game_start == std::string_view::npos) {
                break;
            }

            // get next game start
            size_t next_game = chunk.find(event_tag, game_start + event_tag.length()); // skip this tag, find the next

            // get the game_length so you now have the window to the game    
            size_t game_length = (next_game == std::string_view::npos) ? (chunk.length() - game_start) : (next_game - game_start);
            
            std::string_view single_game = chunk.substr(game_start, game_length);
            
            // parsing logic goes here
            std::vector<std::string_view> moves = extract_moves(single_game);
            my_stats->games_processed++;
            my_stats->moves_processed += moves.size();

            // write to boardstate
            BoardState current_board;
            set_starting_position(current_board);
            int color = WHITE; // xor this after every move, used as a flag
            for (std::string_view move : moves) {
                // castling
                if (move == "O-O" || move == "O-O-O") {
                    int k_source = (color == WHITE) ? 4 : 60;
                    int k_dest   = (move == "O-O") ? k_source + 2 : k_source - 2; 
                    int r_source = (move == "O-O") ? k_source + 3 : k_source - 4;
                    int r_dest   = (move == "O-O") ? k_source + 1 : k_source - 1;

                    current_board.bitboards[color * 6 + KING] ^= (1ULL << k_source) | (1ULL << k_dest);
                    current_board.bitboards[color * 6 + ROOK] ^= (1ULL << r_source) | (1ULL << r_dest);

                    // Write board state to binary output
                    out_bin.write(reinterpret_cast<const char*>(current_board.bitboards), sizeof(current_board.bitboards));
                    my_stats->board_states++;
                    my_stats->bytes_output += sizeof(current_board.bitboards);

                    color ^= 1;
                    continue;
                }
                int len = move.length();
                // clean input for + and # (check/checkmate indicators)
                while (len > 0 && (move[len-1] == '+' || move[len-1] == '#')) {
                    len--;
                }

                std::string_view clean_move = move.substr(0, len);

                // handle promotion: strip =Q, =R, =B, =N from the end for destination parsing
                // but remember the promotion piece
                int promo_piece = -1;
                if (clean_move.find('=') != std::string_view::npos) {
                    // last char after '=' is the promotion piece
                    char promo_char = clean_move.back();
                    if (promo_char == 'Q') promo_piece = QUEEN;
                    else if (promo_char == 'R') promo_piece = ROOK;
                    else if (promo_char == 'B') promo_piece = BISHOP;
                    else if (promo_char == 'N') promo_piece = KNIGHT;

                    // strip from '=' onwards
                    size_t eq_pos = clean_move.find('=');
                    clean_move = clean_move.substr(0, eq_pos);
                    len = clean_move.length();
                }

                // get destination square
                char file_char = clean_move[len - 2];
                char rank_char = clean_move[len - 1];
                int dest_sq = (rank_char - '1') * 8 + (file_char - 'a'); //bitboard

                int piece = PAWN; 
                char first = clean_move[0];

                if (first == 'B') {
                    piece = BISHOP;
                } else if (first == 'N') {
                    piece = KNIGHT;
                } else if (first == 'R') {
                    piece = ROOK;
                } else if (first == 'Q') {
                    piece = QUEEN;
                } else if (first == 'K') {
                    piece = KING; 
                } 

                // build the reverse attack mask
                uint64_t target_mask = 0ULL;
                uint64_t occ = get_occupancy_board(current_board);

                if (piece == PAWN) {
                    bool is_capture = (clean_move.find('x') != std::string_view::npos);
                    int forward = (color == WHITE) ? 8 : -8;

                    if (!is_capture) {

                        // pawn push
                        int single_sq = dest_sq - forward;

                        // check if there is a pawn there, if it is, src
                        if (current_board.bitboards[color * 6 + PAWN] & (1ULL << single_sq)) {
                            target_mask = (1ULL << single_sq);
                        } else {
                            // it is a double push from start
                            int double_sq = dest_sq - (2 * forward);
                            target_mask = (1ULL << double_sq);
                        }
                    } else {
                        target_mask = (color == WHITE) ? BLACK_PAWN_ATTACKS[dest_sq] : WHITE_PAWN_ATTACKS[dest_sq];
                    }
                } else if (piece == KNIGHT) {
                    target_mask = KNIGHT_ATTACKS[dest_sq];
                } else if (piece == BISHOP) {
                    target_mask = get_sliding_attacks(dest_sq, occ, true, false);
                } else if (piece == QUEEN) {
                    target_mask = get_sliding_attacks(dest_sq, occ, true, true);
                } else if (piece == ROOK) {
                    target_mask = get_sliding_attacks(dest_sq, occ, false, true);
                } else if(piece == KING) {
                    target_mask = KING_ATTACKS[dest_sq];
                }


                // disambigulate which piece is moving
                uint64_t my_pieces = current_board.bitboards[color * 6 + piece];
                uint64_t candidates = target_mask & my_pieces;

                if (__builtin_popcountll(candidates) > 1) {
                    char dis = (piece == PAWN) ? clean_move[0] : clean_move[1]; // this shows file/rank of the moving piece
                    if (dis >= 'a' && dis <= 'h') {
                        candidates &= FILE_MASKS[dis - 'a'];
                    } else if (dis >= '1' && dis <= '8') {
                        candidates &= RANK_MASKS[dis - '1'];
                    }
                }
                int source_sq = __builtin_ctzll(candidates);

                // capture
                int enemy_color = color ^ 1;
                bool captured_something = false;
                for (int p = 0; p < 6; p++) {
                    uint64_t& enemy_board = current_board.bitboards[enemy_color * 6 + p];
                    if (enemy_board & (1ULL << dest_sq)) {
                        enemy_board &= ~(1ULL << dest_sq);
                        captured_something = true;
                        break; // only capture 1 at a time
                    }
                }

                // en passant detection with capture
                if (piece == PAWN && clean_move.find('x') != std::string_view::npos && !captured_something) {
                    int ep_sq = dest_sq - ((color == WHITE) ? 8 : -8);
                    current_board.bitboards[enemy_color * 6 + PAWN] &= ~(1ULL << ep_sq);
                } 

                
                // move the main piece
                current_board.bitboards[color * 6 + piece] ^= (1ULL << source_sq) | (1ULL << dest_sq);

                // promotion
                if (promo_piece >= 0) {
                    // Remove the pawn from dest_sq, add the promoted piece
                    current_board.bitboards[color * 6 + PAWN] ^= (1ULL << dest_sq);
                    current_board.bitboards[color * 6 + promo_piece] ^= (1ULL << dest_sq);
                }

                // Write board state to binary output after each move
                out_bin.write(reinterpret_cast<const char*>(current_board.bitboards), sizeof(current_board.bitboards));
                my_stats->board_states++;
                my_stats->bytes_output += sizeof(current_board.bitboards);

                // Swap Turn
                color ^= 1;
            }

            // advance past this game
            curr_pos = game_start + game_length;
        }
    }

    out_bin.close();

    auto t_end = std::chrono::steady_clock::now();
    my_stats->wall_time_sec = std::chrono::duration<double>(t_end - t_start).count();
}

namespace fs = std::filesystem;
std::vector<std::string> get_pgn_files(const std::string &folder_path) {
    std::vector<std::string> files; 

    // check if dir exists
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Directory not found: " << folder_path << "\n";
        return files; 
    }

    for (const auto&entry : fs::directory_iterator(folder_path)) {
        auto ext = entry.path().extension().string();
        if (entry.is_regular_file() && (ext == ".pgn" || ext == ".PGN")) {
            files.push_back(entry.path().filename().string());
        }
    }
    return files;
}

// main method orchestrates
int main() {
    auto t_total_start = std::chrono::steady_clock::now();

    // Initialize all lookup tables BEFORE any processing
    init_lookup_tables();
    init_ray_masks();
    init_pawn_attacks();

    // Dynamically detect the number of hardware threads available (e.g., 16 for Ryzen 7 2700x)
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 16; // Safe fallback
    std::string pgn_folder = "../pgn";
    std::vector<std::string> files = get_pgn_files(pgn_folder);
    
    if (files.empty()) {
        std::cerr << "No PGN files found in " << pgn_folder << "\n";
        return 1;
    }

    for (const auto& f : files) {
        std::cout << "Found: " << f << "\n";
    }

    int k = files.size();

    // Allocate per-thread stats (cache-line padded, zero contention)
    std::vector<ThreadStats> thread_stats(num_threads);

    // the std buffer is alive for the entire duration
    std::vector<std::string> file_buffers(k);

    std::vector<std::vector<std::string_view>> thread_assignments(num_threads);

    int threads_per_file = num_threads / k;
    if (threads_per_file < 1) threads_per_file = 1;

    // Phase 1: Read all files and assign chunks to threads
    for (int i = 0; i < k; i++) {
        std::string full_path = pgn_folder + "/" + files[i];
        file_buffers[i] = read_file(full_path);
        if (file_buffers[i].empty()) {
            std::cerr << "Warning: Could not read " << full_path << ", skipping.\n";
            continue;
        }

        std::cout << "Read " << files[i] << " (" << file_buffers[i].size() << " bytes)\n";

        std::string_view view(file_buffers[i]);
        size_t total_bytes = view.size();
        size_t chunk_size = total_bytes / threads_per_file;

        size_t current_start = 0;
        const std::string_view tag = "[Event ";

        for (int j = 0; j < threads_per_file; ++j) {
            // Calculate which global thread ID gets this chunk (0 to 15)
            int target_thread_id = (i * threads_per_file) + j;
            if (target_thread_id >= num_threads) break;
            
            size_t target_end = current_start + chunk_size;
            
            // If it's the last chunk for this file, just take the rest of it
            if (j == threads_per_file - 1 || target_end >= total_bytes) {
                thread_assignments[target_thread_id].push_back(view.substr(current_start));
                break;
            }

            // Scan forward to find a safe boundary
            size_t safe_cut_point = view.find(tag, target_end);
            
            if (safe_cut_point == std::string_view::npos) {
                thread_assignments[target_thread_id].push_back(view.substr(current_start));
                break;
            }

            thread_assignments[target_thread_id].push_back(view.substr(current_start, safe_cut_point - current_start));
            current_start = safe_cut_point;
        }
    }

    auto t_read_end = std::chrono::steady_clock::now();
    double file_read_sec = std::chrono::duration<double>(t_read_end - t_total_start).count();

    auto t_thread_start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        if (!thread_assignments[t].empty()) {
            workers.emplace_back(worker_thread, t, std::cref(thread_assignments[t]), &thread_stats[t]);
        }
    }

    std::cout << "Launched " << workers.size() << " worker threads...\n";

    // Wait for all threads to finish processing
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto t_total_end = std::chrono::steady_clock::now();
    double total_wall_sec = std::chrono::duration<double>(t_total_end - t_total_start).count();
    double thread_proc_sec = std::chrono::duration<double>(t_total_end - t_thread_start).count();

    // Compute and output stats
    RunStats run = compute_run_stats(thread_stats.data(), num_threads, total_wall_sec, file_read_sec, thread_proc_sec);
    run.num_files = k;

    print_stats_summary(run, thread_stats.data(), num_threads);
    write_stats_json("stats_report.json", run, thread_stats.data(), num_threads);

    std::cout << "Stats saved to stats_report.json\n";
    return 0;
}