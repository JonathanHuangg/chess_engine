#include "mcts.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

// ============================================================
// Phase 4 (C++ side) — plays chess from the starting position
// using MCTS over engine/movegen.cpp with the built-in material
// evaluator. The neural-network search lives in part1/inference.py.
//
// Usage:
//   ./play [--sims N] [--plies N] [--seed N]
// Defaults: --sims 200 --plies 10 --seed 12345
//
// Output: one UCI move per line on stdout, e.g. `e2e4`.
// Diagnostics (side to move, visits, value, timing) go to stderr.
// Exit code 0 unless no legal move exists at the start (never for
// the start position) or an argument is invalid.
// ============================================================

static std::string move_to_uci(const Move& m) {
    const char files[] = "abcdefgh";
    const char ranks[] = "12345678";
    std::string s;
    s += files[m.from % 8];
    s += ranks[m.from / 8];
    s += files[m.to % 8];
    s += ranks[m.to / 8];
    int promo = get_promo_piece(m.flags);
    if (promo >= 0) {
        // PAWN=0, QUEEN=1, KING=2, BISHOP=3, KNIGHT=4, ROOK=5
        const char pc[] = {' ', 'q', ' ', 'b', 'n', 'r'};
        s += pc[promo];
    }
    return s;
}

int main(int argc, char* argv[]) {
    int sims = 200;
    int plies = 10;
    unsigned int seed = 12345;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: " << name << " needs a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--sims") sims = std::stoi(need_value("--sims"));
        else if (arg == "--plies") plies = std::stoi(need_value("--plies"));
        else if (arg == "--seed") seed = static_cast<unsigned int>(std::stoul(need_value("--seed")));
        else {
            std::cerr << "error: unknown argument '" << arg << "'\n"
                      << "usage: ./play [--sims N] [--plies N] [--seed N]\n";
            return 2;
        }
    }
    if (sims < 1 || plies < 1) {
        std::cerr << "error: --sims and --plies must be >= 1\n";
        return 2;
    }

    init_lookup_tables();
    init_ray_masks();
    init_pawn_attacks();

    Position pos;
    init_position(pos);

    MaterialEvaluator evaluator;
    for (int ply = 1; ply <= plies; ply++) {
        auto t0 = std::chrono::steady_clock::now();
        Mcts search(evaluator, sims, 1.5f, seed + ply);
        Move best;
        if (!search.best_move(pos, best)) {
            std::cerr << "ply " << ply << ": no legal moves (game over)\n";
            break;
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::string uci = move_to_uci(best);
        std::cout << uci << "\n" << std::flush;
        std::cerr << "ply " << ply << " (" << (pos.color == WHITE ? "white" : "black")
                  << "): " << uci << "  visits=" << search.root_visits()
                  << " value=" << search.root_value()
                  << " sims=" << sims << " time=" << elapsed << "s\n";

        Position next = pos;
        if (!make_move(next, best)) {
            std::cerr << "error: chosen move " << uci << " is illegal\n";
            return 1;
        }
        pos = next;
    }
    return 0;
}
