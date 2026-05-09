#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "../utils/utils.h"
#include <cstdint>

// ============================================================
// Move representation — 4 bytes, compact
// ============================================================
struct Move {
    uint8_t from;      // source square 0-63
    uint8_t to;        // destination square 0-63
    uint8_t piece;     // piece type moving (PAWN, KNIGHT, etc.)
    uint8_t flags;     // see FLAG_* constants below
};

constexpr uint8_t FLAG_NONE      = 0x00;
constexpr uint8_t FLAG_CAPTURE   = 0x01;
constexpr uint8_t FLAG_EP        = 0x02;
constexpr uint8_t FLAG_CASTLE    = 0x04;
constexpr uint8_t FLAG_PROMO_Q   = 0x08;
constexpr uint8_t FLAG_PROMO_R   = 0x10;
constexpr uint8_t FLAG_PROMO_B   = 0x20;
constexpr uint8_t FLAG_PROMO_N   = 0x40;

constexpr uint8_t FLAG_PROMO_ANY = FLAG_PROMO_Q | FLAG_PROMO_R | FLAG_PROMO_B | FLAG_PROMO_N;

// Helper to extract promotion piece type from flags (-1 if not a promotion)
inline int get_promo_piece(uint8_t flags) {
    if (flags & FLAG_PROMO_Q) return QUEEN;
    if (flags & FLAG_PROMO_R) return ROOK;
    if (flags & FLAG_PROMO_B) return BISHOP;
    if (flags & FLAG_PROMO_N) return KNIGHT;
    return -1;
}

// ============================================================
// Position — extends BoardState with game state for the engine
// ============================================================
struct Position {
    BoardState board;      // 18 x uint64_t (only 0-11 used for pieces)
    bool castling[4];      // [WK, WQ, BK, BQ]
    int ep_square;         // -1 if no en passant available
    int color;             // WHITE or BLACK (side to move)
};

// Constants for castling array indices
constexpr int CASTLE_WK = 0;
constexpr int CASTLE_WQ = 1;
constexpr int CASTLE_BK = 2;
constexpr int CASTLE_BQ = 3;

// File masks for move generation
constexpr uint64_t NOT_FILE_A = ~0x0101010101010101ULL;
constexpr uint64_t NOT_FILE_H = ~0x8080808080808080ULL;

// ============================================================
// Core engine functions
// ============================================================

// Set up starting position
void init_position(Position& pos);

// Generate all pseudo-legal moves. Returns move count.
// move_list must be pre-allocated (256 entries is safe).
int generate_pseudo_legal(const Position& pos, Move* move_list);

// Apply a move to the position. Returns false if the move leaves
// the king in check (illegal). Position is modified in-place.
bool make_move(Position& pos, const Move& m);

// Perft — counts leaf nodes at given depth. Used for validation.
uint64_t perft(Position& pos, int depth);

// Perft divide — prints per-move node counts for debugging.
void perft_divide(Position& pos, int depth);

#endif
