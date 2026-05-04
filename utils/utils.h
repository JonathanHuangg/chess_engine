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

/*
0 = white pawn
1 = white queen
2 = white king
3 = white bishop
4 = white knight
5 = white rook

6 = black pawn
7 = black queen
8 = black king
9 = black bishop
10 = black knight
11 = black rook

12 = white on move/Black on move (1s or 0s)
13 = white king side castling
14 = white queen side castling
15 = black king side castling
16 = black queen side castling
17 = em_passant square (1s or 0s)
*/
struct BoardState {
    uint64_t bitboards[18];
};

struct TrainingSample {
    BoardState state;
    float result;
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
    
    int src_x = src_sq % 8;
    int src_y = src_sq / 8;
    int dest_x = dest_sq % 8;
    int dest_y = dest_sq / 8;
    
    int dx = dest_x - src_x;
    int dy = dest_y - src_y;
    
    int move_type = -1;
    
    if (promo_piece == KNIGHT || promo_piece == BISHOP || promo_piece == ROOK) {
        int dir_idx = 0; // straight
        if (dx == -1) dir_idx = 1; // capture left
        else if (dx == 1) dir_idx = 2; // capture right
        
        int p_idx = 0;
        if (promo_piece == BISHOP) p_idx = 1;
        else if (promo_piece == ROOK) p_idx = 2;
        
        move_type = 64 + (p_idx * 3) + dir_idx;
    } else {
        int abs_dx = dx > 0 ? dx : -dx;
        int abs_dy = dy > 0 ? dy : -dy;

        if (abs_dx > 0 && abs_dy > 0 && abs_dx + abs_dy == 3) {
            if (dx == 1 && dy == 2) move_type = 56;
            else if (dx == 2 && dy == 1) move_type = 57;
            else if (dx == 2 && dy == -1) move_type = 58;
            else if (dx == 1 && dy == -2) move_type = 59;
            else if (dx == -1 && dy == -2) move_type = 60;
            else if (dx == -2 && dy == -1) move_type = 61;
            else if (dx == -2 && dy == 1) move_type = 62;
            else if (dx == -1 && dy == 2) move_type = 63;
        } else {
            int dir_idx = -1;
            if (dx == 0 && dy > 0) dir_idx = 0; // N
            else if (dx > 0 && dy > 0) dir_idx = 1; // NE
            else if (dx > 0 && dy == 0) dir_idx = 2; // E
            else if (dx > 0 && dy < 0) dir_idx = 3; // SE
            else if (dx == 0 && dy < 0) dir_idx = 4; // S
            else if (dx < 0 && dy < 0) dir_idx = 5; // SW
            else if (dx < 0 && dy == 0) dir_idx = 6; // W
            else if (dx < 0 && dy > 0) dir_idx = 7; // NW
            
            int dist = (abs_dx > abs_dy) ? abs_dx : abs_dy;
            move_type = dir_idx * 7 + (dist - 1);
        }
    }
    
    return (int16_t)((src_sq * 73) + move_type);
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