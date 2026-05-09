#include "movegen.h"
#include <iostream>
#include <chrono>
#include <cstdint>

/*
 * Perft validation harness for the Chess Brain move generator.
 *
 * Expected results from the standard starting position:
 *   Depth 1:          20
 *   Depth 2:         400
 *   Depth 3:       8,902
 *   Depth 4:     197,281
 *   Depth 5:   4,865,609
 *   Depth 6: 119,060,324
 */

struct PerftTest {
    const char* fen_label;
    int depth;
    uint64_t expected;
};

// Standard starting position tests
static PerftTest start_tests[] = {
    {"startpos", 1,          20},
    {"startpos", 2,         400},
    {"startpos", 3,        8902},
    {"startpos", 4,      197281},
    {"startpos", 5,     4865609},
    {"startpos", 6,   119060324},
};

int main(int argc, char* argv[]) {
    // Initialize all attack lookup tables
    init_lookup_tables();
    init_ray_masks();
    init_pawn_attacks();

    Position pos;
    init_position(pos);

    int max_depth = 6;
    bool do_divide = false;
    int divide_depth = -1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--divide" && i + 1 < argc) {
            do_divide = true;
            divide_depth = std::stoi(argv[++i]);
        } else if (arg == "--depth" && i + 1 < argc) {
            max_depth = std::stoi(argv[++i]);
        }
    }

    if (do_divide) {
        std::cout << "=== Perft Divide (depth " << divide_depth << ") ===\n";
        perft_divide(pos, divide_depth);
        return 0;
    }

    // Run all perft tests up to max_depth
    std::cout << "=== Chess Brain Perft Validation ===\n\n";

    bool all_passed = true;
    for (auto& test : start_tests) {
        if (test.depth > max_depth) break;

        Position test_pos;
        init_position(test_pos);

        auto t0 = std::chrono::steady_clock::now();
        uint64_t result = perft(test_pos, test.depth);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        bool passed = (result == test.expected);
        double mnps = (result / 1e6) / elapsed;

        std::cout << "Depth " << test.depth << ": "
                  << result;

        if (passed) {
            std::cout << " ✓ PASS";
        } else {
            std::cout << " ✗ FAIL (expected " << test.expected << ")";
            all_passed = false;
        }

        std::cout << "  [" << elapsed << "s, " << mnps << " Mnps]\n";
    }

    std::cout << "\n";
    if (all_passed) {
        std::cout << "All perft tests PASSED. Move generator is correct.\n";
        std::cout << "You may proceed to TensorRT integration.\n";
    } else {
        std::cout << "PERFT FAILED. Bitwise operations have bugs.\n";
        std::cout << "Use --divide <depth> to debug per-move counts.\n";
    }

    return all_passed ? 0 : 1;
}
