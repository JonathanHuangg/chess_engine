/*
HPC Thread Statistics — Implementation

Computes aggregate metrics, writes JSON report, prints summary table.
All collection is done via the cache-line-padded ThreadStats structs
that each thread writes to exclusively (zero contention).
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include "stats.h"

RunStats compute_run_stats(const ThreadStats* thread_stats, int num_threads,
                           double total_wall_sec, double file_read_sec, double thread_proc_sec) {
    RunStats r;
    r.total_wall_time_sec         = total_wall_sec;
    r.file_read_time_sec          = file_read_sec;
    r.thread_processing_time_sec  = thread_proc_sec;

    // Find active threads and accumulate
    int active = 0;
    double time_sum = 0.0;
    double time_sum_sq = 0.0;
    r.thread_time_min = 1e18;
    r.thread_time_max = 0.0;
    r.thread_games_min = UINT64_MAX;
    r.thread_games_max = 0;
    r.thread_boards_min = UINT64_MAX;
    r.thread_boards_max = 0;

    for (int i = 0; i < num_threads; i++) {
        const ThreadStats& ts = thread_stats[i];
        if (ts.thread_id < 0) continue; // unused thread slot

        active++;
        r.total_games        += ts.games_processed;
        r.total_board_states += ts.board_states;
        r.total_moves        += ts.moves_processed;
        r.total_bytes_in     += ts.bytes_input;
        r.total_bytes_out    += ts.bytes_output;

        time_sum    += ts.wall_time_sec;
        time_sum_sq += ts.wall_time_sec * ts.wall_time_sec;

        r.thread_time_min  = std::min(r.thread_time_min, ts.wall_time_sec);
        r.thread_time_max  = std::max(r.thread_time_max, ts.wall_time_sec);
        r.thread_games_min = std::min(r.thread_games_min, ts.games_processed);
        r.thread_games_max = std::max(r.thread_games_max, ts.games_processed);
        r.thread_boards_min = std::min(r.thread_boards_min, ts.board_states);
        r.thread_boards_max = std::max(r.thread_boards_max, ts.board_states);
    }

    r.num_threads_used = active;

    if (active > 0) {
        r.thread_time_avg = time_sum / active;
        double variance = (time_sum_sq / active) - (r.thread_time_avg * r.thread_time_avg);
        r.thread_time_stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;
        r.load_imbalance_pct = (r.thread_time_avg > 0.0)
            ? ((r.thread_time_max - r.thread_time_min) / r.thread_time_avg) * 100.0
            : 0.0;
    }

    // Throughput based on thread processing wall time (not total wall time)
    if (thread_proc_sec > 0.0) {
        r.games_per_sec         = r.total_games / thread_proc_sec;
        r.board_states_per_sec  = r.total_board_states / thread_proc_sec;
        r.input_throughput_mbps = (r.total_bytes_in / (1024.0 * 1024.0)) / thread_proc_sec;
        r.output_throughput_mbps = (r.total_bytes_out / (1024.0 * 1024.0)) / thread_proc_sec;
    }

    return r;
}

// Helper: escape a JSON string (minimal)
static std::string json_str(const std::string& s) {
    return "\"" + s + "\"";
}

void write_stats_json(const std::string& filepath, const RunStats& run,
                      const ThreadStats* thread_stats, int num_threads) {
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Failed to write stats to " << filepath << "\n";
        return;
    }

    out << std::fixed << std::setprecision(4);
    out << "{\n";

    // Global timing
    out << "  \"timing\": {\n";
    out << "    \"total_wall_time_sec\": "        << run.total_wall_time_sec << ",\n";
    out << "    \"file_read_time_sec\": "          << run.file_read_time_sec << ",\n";
    out << "    \"thread_processing_time_sec\": "  << run.thread_processing_time_sec << "\n";
    out << "  },\n";

    // Aggregate work
    out << "  \"work\": {\n";
    out << "    \"total_games\": "        << run.total_games << ",\n";
    out << "    \"total_board_states\": "  << run.total_board_states << ",\n";
    out << "    \"total_moves\": "         << run.total_moves << ",\n";
    out << "    \"total_bytes_in\": "      << run.total_bytes_in << ",\n";
    out << "    \"total_bytes_out\": "     << run.total_bytes_out << ",\n";
    out << "    \"num_threads_used\": "    << run.num_threads_used << ",\n";
    out << "    \"num_files\": "           << run.num_files << "\n";
    out << "  },\n";

    // Throughput
    out << "  \"throughput\": {\n";
    out << "    \"games_per_sec\": "          << run.games_per_sec << ",\n";
    out << "    \"board_states_per_sec\": "    << run.board_states_per_sec << ",\n";
    out << "    \"input_mb_per_sec\": "        << run.input_throughput_mbps << ",\n";
    out << "    \"output_mb_per_sec\": "       << run.output_throughput_mbps << "\n";
    out << "  },\n";

    // Load balance
    out << "  \"load_balance\": {\n";
    out << "    \"thread_time_min_sec\": "    << run.thread_time_min << ",\n";
    out << "    \"thread_time_max_sec\": "    << run.thread_time_max << ",\n";
    out << "    \"thread_time_avg_sec\": "    << run.thread_time_avg << ",\n";
    out << "    \"thread_time_stddev_sec\": " << run.thread_time_stddev << ",\n";
    out << "    \"load_imbalance_pct\": "     << run.load_imbalance_pct << ",\n";
    out << "    \"thread_games_min\": "       << run.thread_games_min << ",\n";
    out << "    \"thread_games_max\": "       << run.thread_games_max << ",\n";
    out << "    \"thread_boards_min\": "      << run.thread_boards_min << ",\n";
    out << "    \"thread_boards_max\": "      << run.thread_boards_max << "\n";
    out << "  },\n";

    // Per-thread breakdown
    out << "  \"threads\": [\n";
    bool first = true;
    for (int i = 0; i < num_threads; i++) {
        const ThreadStats& ts = thread_stats[i];
        if (ts.thread_id < 0) continue;

        if (!first) out << ",\n";
        first = false;

        out << "    {\n";
        out << "      \"thread_id\": "        << ts.thread_id << ",\n";
        out << "      \"wall_time_sec\": "     << ts.wall_time_sec << ",\n";
        out << "      \"games_processed\": "   << ts.games_processed << ",\n";
        out << "      \"board_states\": "       << ts.board_states << ",\n";
        out << "      \"moves_processed\": "    << ts.moves_processed << ",\n";
        out << "      \"bytes_input\": "        << ts.bytes_input << ",\n";
        out << "      \"bytes_output\": "       << ts.bytes_output << ",\n";
        out << "      \"chunks_assigned\": "    << ts.chunks_assigned << "\n";
        out << "    }";
    }
    out << "\n  ]\n";
    out << "}\n";

    out.close();
}

void print_stats_summary(const RunStats& run, const ThreadStats* thread_stats, int num_threads) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ETL RUN STATISTICS                       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    std::cout << std::fixed << std::setprecision(2);

    // Timing
    std::cout << "║  TIMING                                                     ║\n";
    std::cout << "║    Total wall time:          " << std::setw(12) << run.total_wall_time_sec << " sec" << std::setw(17) << "║\n";
    std::cout << "║    File I/O time:            " << std::setw(12) << run.file_read_time_sec << " sec" << std::setw(17) << "║\n";
    std::cout << "║    Thread processing time:   " << std::setw(12) << run.thread_processing_time_sec << " sec" << std::setw(17) << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    // Work
    std::cout << "║  WORK                                                       ║\n";
    std::cout << "║    Games processed:          " << std::setw(16) << run.total_games << std::setw(17) << "║\n";
    std::cout << "║    Board states generated:   " << std::setw(16) << run.total_board_states << std::setw(17) << "║\n";
    std::cout << "║    Total moves parsed:       " << std::setw(16) << run.total_moves << std::setw(17) << "║\n";
    std::cout << "║    Input data:               " << std::setw(12) << (run.total_bytes_in / (1024.0 * 1024.0)) << " MB" << std::setw(18) << "║\n";
    std::cout << "║    Output data:              " << std::setw(12) << (run.total_bytes_out / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::setw(18) << "║\n";
    std::cout << "║    Threads used:             " << std::setw(16) << run.num_threads_used << std::setw(17) << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    // Throughput
    std::cout << "║  THROUGHPUT                                                 ║\n";
    std::cout << "║    Games/sec:                " << std::setw(12) << run.games_per_sec << std::setw(21) << "║\n";
    std::cout << "║    Board states/sec:         " << std::setw(12) << run.board_states_per_sec << std::setw(21) << "║\n";
    std::cout << "║    Input throughput:         " << std::setw(12) << run.input_throughput_mbps << " MB/s" << std::setw(16) << "║\n";
    std::cout << "║    Output throughput:        " << std::setw(12) << run.output_throughput_mbps << " MB/s" << std::setw(16) << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    // Load balance
    std::cout << "║  LOAD BALANCE                                               ║\n";
    std::cout << "║    Thread time (min/avg/max):" 
              << std::setw(7) << run.thread_time_min << " / "
              << std::setw(7) << run.thread_time_avg << " / "
              << std::setw(7) << run.thread_time_max << " sec"
              << std::setw(6) << "║\n";
    std::cout << "║    Stddev:                   " << std::setw(12) << run.thread_time_stddev << " sec" << std::setw(17) << "║\n";
    std::cout << "║    Load imbalance:           " << std::setw(12) << run.load_imbalance_pct << " %" << std::setw(19) << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

    // Per-thread table
    std::cout << "║  PER-THREAD BREAKDOWN                                       ║\n";
    std::cout << "║  TID   Time(s)     Games     Boards     Moves   Out(MB)     ║\n";
    std::cout << "║  ---  --------  --------  ---------  --------  --------     ║\n";

    for (int i = 0; i < num_threads; i++) {
        const ThreadStats& ts = thread_stats[i];
        if (ts.thread_id < 0) continue;

        std::cout << "║  " << std::setw(3) << ts.thread_id
                  << "  " << std::setw(8) << ts.wall_time_sec
                  << "  " << std::setw(8) << ts.games_processed
                  << "  " << std::setw(9) << ts.board_states
                  << "  " << std::setw(8) << ts.moves_processed
                  << "  " << std::setw(8) << std::setprecision(1) << (ts.bytes_output / (1024.0 * 1024.0))
                  << std::setprecision(2)
                  << std::setw(6) << "║\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;
}