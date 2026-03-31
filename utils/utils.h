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

struct TrainingSample {
    BoardState state; // 112 bytes
    float result; // 4 bytes
    int16_t move_idx; // 2 bytes
    int16_t padding; // 2 bytes
};

// for black, you want to transition it to white so you have to flip it
// need to add in 
inline int16_t encode_move_flipped(int src_sq, int dest_sq, int color, int promo_piece) {

    if (color == BLACK) {
        src_sq = src_sq ^ 56;
        dest_sq = dest_sq ^ 56;
    }
    
    if (promo_piece == -1 || promo_piece == QUEEN) {
        return (int16_t)((src_sq * 64) + dest_sq);
    }

    int file_from = src_sq % 8;
    int file_to = dest_sq % 8;
    
    int direction = file_to - file_from + 1;

    int piece_offset = 0;
    if (promo_piece == KNIGHT) {
        piece_offset = 0;
    } else if (promo_piece == BISHOP) {
        piece_offset = 1;
    } else if (promo_piece == ROOK) {
        piece_offset = 2;
    }

    int underpromo_index = (piece_offset * 24) + (file_from * 3) + direction;
    
    return (int16_t)(4096 + underpromo_index);
}

inline uint64_t flip_bitboard_vertical(uint64_t b) {
    return __builtin_bswap64(b);
}
inline float get_game_result(std::string_view game_text) {
    if (game_text.find("[Result \"1-0\"]") != std::string_view::npos) return 1.0f;
    if (game_text.find("[Result \"0-1\"]") != std::string_view::npos) return -1.0f;
    return 0.0f;
}



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