#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
#include <vector>
#include <string_view>

#define BLACK 0 
#define WHITE 1
#define PAWN 0
#define QUEEN 1
#define KING 2
#define BISHOP 3
#define KNIGHT 4
#define ROOK 5


struct BoardState {
    uint64_t bitboards[14]; // each piece has its own board. 14 planes * 8 bytes = 112 bytes
};

extern uint64_t KNIGHT_ATTACKS[64];
extern uint64_t WHITE_PAWN_ATTACKS[64];
extern uint64_t BLACK_PAWN_ATTACKS[64];
extern uint64_t KING_ATTACKS[64];
extern uint64_t FILE_MASKS[8];
extern uint64_t RANK_MASKS[8];

// Initialization functions — must be called before any move processing
void init_lookup_tables();
void init_ray_masks();
void init_pawn_attacks();

// Board utilities
uint64_t get_occupancy_board(const BoardState &board);
uint64_t get_sliding_attacks(int sq, uint64_t occ, bool is_diagonal, bool is_orthogonal);
void set_starting_position(BoardState &board);

// Move parsing
std::vector<std::string_view> extract_moves(std::string_view game_text);

#endif