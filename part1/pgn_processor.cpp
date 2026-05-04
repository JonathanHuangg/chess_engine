#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <thread>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <cmath> 
#include "../utils/utils.h"
#include "stats.h"

// ==========================================
// PGN PARSER (STATE MACHINE)
// ==========================================
std::vector<std::string_view> extract_moves(std::string_view game_text) {
    std::vector<std::string_view> clean_moves;
    clean_moves.reserve(100);

    size_t pos = 0;
    size_t length = game_text.length();
    
    // State trackers to ignore comments and variations
    int in_comment_brace = 0;   // {}
    int in_variation_paren = 0; // ()
    int in_tag_bracket = 0;     // []

    while (pos < length) {
        char c = game_text[pos];

        // Handle State Entrances and Exits
        if (c == '{') { in_comment_brace++; pos++; continue; }
        if (c == '}') { in_comment_brace--; pos++; continue; }
        if (c == '(') { in_variation_paren++; pos++; continue; }
        if (c == ')') { in_variation_paren--; pos++; continue; }
        if (c == '[') { in_tag_bracket++; pos++; continue; }
        if (c == ']') { in_tag_bracket--; pos++; continue; }

        // If we are inside any metadata block, skip the character
        if (in_comment_brace > 0 || in_variation_paren > 0 || in_tag_bracket > 0) {
            pos++;
            continue;
        }

        if (std::isspace(c)) {
            pos++;
            continue;
        }

        // We are at the start of a valid token
        size_t end_pos = pos;
        while (end_pos < length && !std::isspace(game_text[end_pos]) && 
               game_text[end_pos] != '{' && game_text[end_pos] != '(' && game_text[end_pos] != '[') {
            end_pos++;
        }

        std::string_view token = game_text.substr(pos, end_pos - pos);
        pos = end_pos;

        // Disregard move numbers and game results
        if (token.find('.') != std::string_view::npos) continue;
        if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") continue;

        clean_moves.push_back(token);
    }
    return clean_moves;
}


// FILE I/O
std::string read_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open: " << filepath << "\n";
        return "";
    }
    in.seekg(0, std::ios::end);
    std::string contents;
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return contents;
}


// ==========================================
// WORKER THREAD
// ==========================================
void worker_thread(int thread_id, const std::vector<std::string_view>& chunks, ThreadStats* my_stats) {
    auto t_start = std::chrono::steady_clock::now();
    my_stats->thread_id = thread_id;
    my_stats->chunks_assigned = chunks.size();

    std::string out_filepath = "chunk_" + std::to_string(thread_id) + ".bin";
    std::ofstream out_bin(out_filepath, std::ios::binary | std::ios::trunc); 

    std::vector<TrainingSample> write_buffer;
    write_buffer.reserve(8192); // ~1MB chunking

    for (std::string_view chunk : chunks) {
        my_stats->bytes_input += chunk.size();
        size_t curr_pos = 0;
        const std::string_view event_tag = "[Event ";

        while (curr_pos < chunk.length()) {
            size_t game_start = chunk.find(event_tag, curr_pos);
            if (game_start == std::string_view::npos) break;

            size_t next_game = chunk.find(event_tag, game_start + event_tag.length()); 
            size_t game_length = (next_game == std::string_view::npos) ? (chunk.length() - game_start) : (next_game - game_start);
            
            std::string_view single_game = chunk.substr(game_start, game_length);
            float absolute_result = get_game_result(single_game);
            
            std::vector<std::string_view> moves = extract_moves(single_game);
            my_stats->games_processed++;
            my_stats->moves_processed += moves.size();

            BoardState current_board;
            set_starting_position(current_board);
            int color = WHITE; 

            bool w_k_castle = true;
            bool w_q_castle = true;
            bool b_k_castle = true;
            bool b_q_castle = true;
            int ep_square = -1;

            for (std::string_view move : moves) {
                int source_sq = -1;
                int dest_sq = -1;
                int piece = -1;
                int promo_piece = -1;
                bool is_castling = false;
                
                int k_source, k_dest, r_source, r_dest;

                // --- PHASE 1: PARSING & DISAMBIGUATION ---
                if (move == "O-O" || move == "O-O-O") {
                    is_castling = true;
                    piece = KING;
                    k_source = (color == WHITE) ? 4 : 60;
                    k_dest   = (move == "O-O") ? k_source + 2 : k_source - 2; 
                    r_source = (move == "O-O") ? k_source + 3 : k_source - 4;
                    r_dest   = (move == "O-O") ? k_source + 1 : k_source - 1;
                    source_sq = k_source;
                    dest_sq = k_dest;
                } else {
                    int len = move.length();
                    while (len > 0 && (move[len-1] == '+' || move[len-1] == '#')) len--;
                    std::string_view clean_move = move.substr(0, len);

                    if (clean_move.find('=') != std::string_view::npos) {
                        char promo_char = clean_move.back();
                        if (promo_char == 'Q') promo_piece = QUEEN;
                        else if (promo_char == 'R') promo_piece = ROOK;
                        else if (promo_char == 'B') promo_piece = BISHOP;
                        else if (promo_char == 'N') promo_piece = KNIGHT;

                        size_t eq_pos = clean_move.find('=');
                        clean_move = clean_move.substr(0, eq_pos);
                        len = clean_move.length();
                    }

                    char file_char = clean_move[len - 2];
                    char rank_char = clean_move[len - 1];
                    dest_sq = (rank_char - '1') * 8 + (file_char - 'a');

                    piece = PAWN; 
                    char first = clean_move[0];
                    if (first == 'B') piece = BISHOP;
                    else if (first == 'N') piece = KNIGHT;
                    else if (first == 'R') piece = ROOK;
                    else if (first == 'Q') piece = QUEEN;
                    else if (first == 'K') piece = KING; 

                    uint64_t target_mask = 0ULL;
                    uint64_t occ = get_occupancy_board(current_board);

                    if (piece == PAWN) {
                        bool is_capture = (clean_move.find('x') != std::string_view::npos);
                        int forward = (color == WHITE) ? 8 : -8;
                        if (!is_capture) {
                            int single_sq = dest_sq - forward;
                            if (current_board.bitboards[color * 6 + PAWN] & (1ULL << single_sq)) {
                                target_mask = (1ULL << single_sq);
                            } else {
                                int double_sq = dest_sq - (2 * forward);
                                target_mask = (1ULL << double_sq);
                            }
                        } else {
                            target_mask = (color == WHITE) ? BLACK_PAWN_ATTACKS[dest_sq] : WHITE_PAWN_ATTACKS[dest_sq];
                        }
                    } else if (piece == KNIGHT) target_mask = KNIGHT_ATTACKS[dest_sq];
                    else if (piece == BISHOP) target_mask = get_sliding_attacks(dest_sq, occ, true, false);
                    else if (piece == QUEEN)  target_mask = get_sliding_attacks(dest_sq, occ, true, true);
                    else if (piece == ROOK)   target_mask = get_sliding_attacks(dest_sq, occ, false, true);
                    else if (piece == KING)   target_mask = KING_ATTACKS[dest_sq];

                    uint64_t my_pieces = current_board.bitboards[color * 6 + piece];
                    uint64_t candidates = target_mask & my_pieces;

                    if (__builtin_popcountll(candidates) > 1) {
                        char dis = (piece == PAWN) ? clean_move[0] : clean_move[1]; 
                        if (dis >= 'a' && dis <= 'h') candidates &= FILE_MASKS[dis - 'a'];
                        else if (dis >= '1' && dis <= '8') candidates &= RANK_MASKS[dis - '1'];
                    }
                    if (candidates == 0) {
                        std::cerr << "Warning: no source square found for move '" 
                                  << move << "', skipping.\n";
                        continue;
                    }
                    source_sq = __builtin_ctzll(candidates);
                }

                // turn to traingsample
                TrainingSample sample;
                
                // Copy & Flip Bitboards
                for (int i = 0; i < 12; ++i) {
                    if (color == WHITE) {
                        sample.state.bitboards[i] = current_board.bitboards[i];
                    } else {
                        sample.state.bitboards[i] = flip_bitboard_vertical(current_board.bitboards[i]);
                    }
                }

                // Write Metadata Planes (12 = Turn, 13 = Rights)
                sample.state.bitboards[12] = (color == WHITE) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;

                // if I am white, we castle if the white king/queen castles
                bool my_king_castle = (color == WHITE) ? w_k_castle : b_k_castle;
                bool my_queen_castle = (color == WHITE) ? w_q_castle : b_q_castle;
                bool enemy_king_castle = (color == WHITE) ? b_k_castle : w_k_castle;
                bool enemy_queen_castle = (color == WHITE) ? b_q_castle : w_q_castle;
                
                // if I castle, set to 1s else 0
                sample.state.bitboards[13] = my_king_castle ? 0xFFFFFFFFFFFFFFFFULL : 0LL;
                sample.state.bitboards[14] = my_queen_castle ? 0xFFFFFFFFFFFFFFFFULL : 0LL;
                sample.state.bitboards[15] = enemy_king_castle ? 0xFFFFFFFFFFFFFFFFULL : 0LL;
                sample.state.bitboards[16] = enemy_queen_castle ? 0xFFFFFFFFFFFFFFFFULL : 0LL;

                // em passant square
                sample.state.bitboards[17] = 0ULL;
                if (ep_square != -1) {
                    uint64_t ep_board = (1ULL << ep_square);
                    sample.state.bitboards[17] = (color == WHITE) ? ep_board : flip_bitboard_vertical(ep_board);
                }

                // Set Labels 
                sample.result = (color == WHITE) ? absolute_result : (absolute_result * -1.0f);
                sample.move_idx = encode_move_flipped(source_sq, dest_sq, color, promo_piece);
                sample.padding = 0;

                // Push to Buffer
                write_buffer.push_back(sample);
                my_stats->board_states++;
                my_stats->bytes_output += sizeof(TrainingSample);

                if (write_buffer.size() >= 8192) {
                    out_bin.write(reinterpret_cast<const char*>(write_buffer.data()), 
                                  write_buffer.size() * sizeof(TrainingSample));
                    write_buffer.clear();
                }

                // --- PHASE 3: EXECUTE BOARD UPDATE & STATE TRACKING ---
                int next_ep_square = -1;

                if (is_castling) {
                    current_board.bitboards[color * 6 + KING] ^= (1ULL << k_source) | (1ULL << k_dest);
                    current_board.bitboards[color * 6 + ROOK] ^= (1ULL << r_source) | (1ULL << r_dest);
                    
                    if (color == WHITE) { w_k_castle = false; w_q_castle = false; }
                    else                { b_k_castle = false; b_q_castle = false; }
                } else {
                    int enemy_color = color ^ 1;
                    bool captured_something = false;
                    for (int p = 0; p < 6; p++) {
                        uint64_t& enemy_board = current_board.bitboards[enemy_color * 6 + p];
                        if (enemy_board & (1ULL << dest_sq)) {
                            enemy_board &= ~(1ULL << dest_sq);
                            captured_something = true;
                            break; 
                        }
                    }

                    if (piece == PAWN && move.find('x') != std::string_view::npos && !captured_something) {
                        int ep_sq = dest_sq - ((color == WHITE) ? 8 : -8);
                        current_board.bitboards[enemy_color * 6 + PAWN] &= ~(1ULL << ep_sq);
                    } 
                    
                    current_board.bitboards[color * 6 + piece] ^= (1ULL << source_sq) | (1ULL << dest_sq);

                    if (promo_piece >= 0) {
                        current_board.bitboards[color * 6 + PAWN] ^= (1ULL << dest_sq);
                        current_board.bitboards[color * 6 + promo_piece] ^= (1ULL << dest_sq);
                    }

                    // Revoke Castling on Move/Capture
                    if (piece == KING) {
                        if (color == WHITE) { w_k_castle = false; w_q_castle = false; }
                        else                { b_k_castle = false; b_q_castle = false; }
                    }
                    if (piece == ROOK) {
                        if (source_sq == 7) w_k_castle = false;
                        if (source_sq == 0) w_q_castle = false;
                        if (source_sq == 63) b_k_castle = false;
                        if (source_sq == 56) b_q_castle = false;
                    }
                    if (dest_sq == 7) w_k_castle = false;
                    if (dest_sq == 0) w_q_castle = false;
                    if (dest_sq == 63) b_k_castle = false;
                    if (dest_sq == 56) b_q_castle = false;

                    // Detect Double Pawn Pushes
                    if (piece == PAWN && std::abs(dest_sq - source_sq) == 16) {
                        next_ep_square = dest_sq - ((color == WHITE) ? 8 : -8);
                    }
                }

                ep_square = next_ep_square;
                color ^= 1;
            }
            curr_pos = game_start + game_length;
        }
    }

    // Flush remaining buffer data
    if (!write_buffer.empty()) {
        out_bin.write(reinterpret_cast<const char*>(write_buffer.data()), 
                      write_buffer.size() * sizeof(TrainingSample));
    }

    out_bin.close();
    auto t_end = std::chrono::steady_clock::now();
    my_stats->wall_time_sec = std::chrono::duration<double>(t_end - t_start).count();
}

// MAIN ORCHESTRATOR
namespace fs = std::filesystem;
std::vector<std::string> get_pgn_files(const std::string &folder_path) {
    std::vector<std::string> files; 
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

int main() {
    auto t_total_start = std::chrono::steady_clock::now();

    // Initialize all lookup tables BEFORE any processing
    init_lookup_tables();
    init_ray_masks();
    init_pawn_attacks();

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 16; 
    std::string pgn_folder = "../pgn";
    std::vector<std::string> files = get_pgn_files(pgn_folder);
    
    if (files.empty()) {
        std::cerr << "No PGN files found in " << pgn_folder << "\n";
        return 1;
    }

    for (const auto& f : files) std::cout << "Found: " << f << "\n";

    int k = files.size();
    std::vector<ThreadStats> thread_stats(num_threads);
    std::vector<std::string> file_buffers(k);
    std::vector<std::vector<std::string_view>> thread_assignments(num_threads);

    int threads_per_file = num_threads / k;
    if (threads_per_file < 1) threads_per_file = 1;

    // Phase 1: Read all files and assign chunks to threads
    for (int i = 0; i < k; i++) {
        std::string full_path = pgn_folder + "/" + files[i];
        file_buffers[i] = read_file(full_path);
        if (file_buffers[i].empty()) continue;

        std::cout << "Read " << files[i] << " (" << file_buffers[i].size() << " bytes)\n";

        std::string_view view(file_buffers[i]);
        size_t total_bytes = view.size();
        size_t chunk_size = total_bytes / threads_per_file;

        size_t current_start = 0;
        const std::string_view tag = "[Event ";

        for (int j = 0; j < threads_per_file; ++j) {
            int target_thread_id = (i * threads_per_file) + j;
            if (target_thread_id >= num_threads) break;
            
            size_t target_end = current_start + chunk_size;
            
            if (j == threads_per_file - 1 || target_end >= total_bytes) {
                thread_assignments[target_thread_id].push_back(view.substr(current_start));
                break;
            }

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

    for (auto& t : workers) {
        if (t.joinable()) t.join();
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