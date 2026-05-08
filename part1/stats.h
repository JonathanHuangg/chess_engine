/*
HPC Thread Statistics Collection
*/

#ifndef STATS_H
#define STATS_H

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <string>

constexpr size_t CACHE_LINE_SIZE = 64;


struct alignas(CACHE_LINE_SIZE) ThreadStats {
    // Timing
    double wall_time_sec    = 0.0; 
    
    // Work counters
    uint64_t games_processed   = 0;
    uint64_t board_states      = 0;
    uint64_t moves_processed   = 0;
    uint64_t bytes_input       = 0;
    uint64_t bytes_output      = 0;
    
    // Thread metadata
    int thread_id              = -1;
    int chunks_assigned        = 0;

    char _pad[128 - 72];
};

struct RunStats {
    // Global timing
    double total_wall_time_sec       = 0.0; 
    double file_read_time_sec        = 0.0; 
    double thread_processing_time_sec = 0.0;

    // Aggregate work
    uint64_t total_games       = 0;
    uint64_t total_board_states = 0;
    uint64_t total_moves       = 0;
    uint64_t total_bytes_in    = 0;
    uint64_t total_bytes_out   = 0;
    int      num_threads_used  = 0;
    int      num_files         = 0;

    // Throughput
    double games_per_sec             = 0.0;
    double board_states_per_sec      = 0.0;
    double input_throughput_mbps     = 0.0;
    double output_throughput_mbps    = 0.0; 

    // Load balance (across threads)
    double thread_time_min           = 0.0;
    double thread_time_max           = 0.0;
    double thread_time_avg           = 0.0;
    double thread_time_stddev        = 0.0;
    double load_imbalance_pct        = 0.0;

    uint64_t thread_games_min        = 0;
    uint64_t thread_games_max        = 0;
    uint64_t thread_boards_min       = 0;
    uint64_t thread_boards_max       = 0;
};

// Compute aggregate stats from per-thread stats
RunStats compute_run_stats(const ThreadStats* thread_stats, int num_threads,
                           double total_wall_sec, double file_read_sec, double thread_proc_sec);

// Write stats to a JSON file
bool write_stats_json(const std::string& filepath, const RunStats& run,
                      const ThreadStats* thread_stats, int num_threads);

// Print a summary table to stdout
void print_stats_summary(const RunStats& run, const ThreadStats* thread_stats, int num_threads);

#endif
