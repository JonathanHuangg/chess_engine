#include "movegen.h"
#include <cstring>
#include <iostream>
#include <cmath>

// ============================================================
// Position initialization
// ============================================================
void init_position(Position& pos) {
    set_starting_position(pos.board);
    pos.castling[CASTLE_WK] = true;
    pos.castling[CASTLE_WQ] = true;
    pos.castling[CASTLE_BK] = true;
    pos.castling[CASTLE_BQ] = true;
    pos.ep_square = -1;
    pos.color = WHITE;
}

// ============================================================
// Internal helpers
// ============================================================

// Get combined occupancy for one side
static inline uint64_t side_occupancy(const BoardState& board, int color) {
    uint64_t occ = 0;
    for (int i = 0; i < 6; i++)
        occ |= board.bitboards[color * 6 + i];
    return occ;
}

// Add a move to the list
static inline void add_move(Move* list, int& count,
                            uint8_t from, uint8_t to,
                            uint8_t piece, uint8_t flags) {
    list[count++] = {from, to, piece, flags};
}

// Add promotion moves (4 variants, optionally with capture flag)
static inline void add_promotions(Move* list, int& count,
                                  uint8_t from, uint8_t to,
                                  uint8_t capture_flag) {
    add_move(list, count, from, to, PAWN, capture_flag | FLAG_PROMO_Q);
    add_move(list, count, from, to, PAWN, capture_flag | FLAG_PROMO_R);
    add_move(list, count, from, to, PAWN, capture_flag | FLAG_PROMO_B);
    add_move(list, count, from, to, PAWN, capture_flag | FLAG_PROMO_N);
}


static int gen_pawn_moves(const Position& pos, uint64_t enemies,
                          uint64_t empty, Move* moves, int count) {
    int color = pos.color;
    uint64_t pawns = pos.board.bitboards[color * 6 + PAWN];

    if (color == WHITE) {
        uint64_t single = (pawns << 8) & empty; // check the space the row above only if empty

        uint64_t promo_push = single & RANK_MASKS[7];
        single &= ~RANK_MASKS[7]; // non-promotion pushes

        while (single) {
            int to = ctz64(single);
            add_move(moves, count, to - 8, to, PAWN, FLAG_NONE);
            single &= single - 1;
        }
        while (promo_push) {
            int to = ctz64(promo_push);
            add_promotions(moves, count, to - 8, to, FLAG_NONE);
            promo_push &= promo_push - 1;
        }

        // --- Double pushes (from rank 1, 0-indexed) ---
        uint64_t rank2_pawns = pawns & RANK_MASKS[1];
        uint64_t step1 = (rank2_pawns << 8) & empty;
        uint64_t dbl = (step1 << 8) & empty;
        while (dbl) {
            int to = ctz64(dbl);
            add_move(moves, count, to - 16, to, PAWN, FLAG_NONE);
            dbl &= dbl - 1;
        }

        // --- Captures left (NW: +7) ---
        uint64_t cap_left = ((pawns & NOT_FILE_A) << 7) & enemies;
        uint64_t promo_cap_l = cap_left & RANK_MASKS[7];
        cap_left &= ~RANK_MASKS[7];
        while (cap_left) {
            int to = ctz64(cap_left);
            add_move(moves, count, to - 7, to, PAWN, FLAG_CAPTURE);
            cap_left &= cap_left - 1;
        }
        while (promo_cap_l) {
            int to = ctz64(promo_cap_l);
            add_promotions(moves, count, to - 7, to, FLAG_CAPTURE);
            promo_cap_l &= promo_cap_l - 1;
        }

        // --- Captures right (NE: +9) ---
        uint64_t cap_right = ((pawns & NOT_FILE_H) << 9) & enemies;
        uint64_t promo_cap_r = cap_right & RANK_MASKS[7];
        cap_right &= ~RANK_MASKS[7];
        while (cap_right) {
            int to = ctz64(cap_right);
            add_move(moves, count, to - 9, to, PAWN, FLAG_CAPTURE);
            cap_right &= cap_right - 1;
        }
        while (promo_cap_r) {
            int to = ctz64(promo_cap_r);
            add_promotions(moves, count, to - 9, to, FLAG_CAPTURE);
            promo_cap_r &= promo_cap_r - 1;
        }

        // --- En passant ---
        if (pos.ep_square != -1) {
            uint64_t ep_bit = 1ULL << pos.ep_square;
            // Pawns that can capture left to ep square
            uint64_t ep_left = ((pawns & NOT_FILE_A) << 7) & ep_bit;
            if (ep_left)
                add_move(moves, count, pos.ep_square - 7, pos.ep_square, PAWN, FLAG_CAPTURE | FLAG_EP);
            // Pawns that can capture right to ep square
            uint64_t ep_right = ((pawns & NOT_FILE_H) << 9) & ep_bit;
            if (ep_right)
                add_move(moves, count, pos.ep_square - 9, pos.ep_square, PAWN, FLAG_CAPTURE | FLAG_EP);
        }
    } else {
        // BLACK pawns — mirror everything
        uint64_t single = (pawns >> 8) & empty;
        uint64_t promo_push = single & RANK_MASKS[0];
        single &= ~RANK_MASKS[0];

        while (single) {
            int to = ctz64(single);
            add_move(moves, count, to + 8, to, PAWN, FLAG_NONE);
            single &= single - 1;
        }
        while (promo_push) {
            int to = ctz64(promo_push);
            add_promotions(moves, count, to + 8, to, FLAG_NONE);
            promo_push &= promo_push - 1;
        }

        uint64_t rank6_pawns = pawns & RANK_MASKS[6];
        uint64_t step1 = (rank6_pawns >> 8) & empty;
        uint64_t dbl = (step1 >> 8) & empty;
        while (dbl) {
            int to = ctz64(dbl);
            add_move(moves, count, to + 16, to, PAWN, FLAG_NONE);
            dbl &= dbl - 1;
        }

        uint64_t cap_left = ((pawns & NOT_FILE_A) >> 9) & enemies;
        uint64_t promo_cap_l = cap_left & RANK_MASKS[0];
        cap_left &= ~RANK_MASKS[0];
        while (cap_left) {
            int to = ctz64(cap_left);
            add_move(moves, count, to + 9, to, PAWN, FLAG_CAPTURE);
            cap_left &= cap_left - 1;
        }
        while (promo_cap_l) {
            int to = ctz64(promo_cap_l);
            add_promotions(moves, count, to + 9, to, FLAG_CAPTURE);
            promo_cap_l &= promo_cap_l - 1;
        }

        uint64_t cap_right = ((pawns & NOT_FILE_H) >> 7) & enemies;
        uint64_t promo_cap_r = cap_right & RANK_MASKS[0];
        cap_right &= ~RANK_MASKS[0];
        while (cap_right) {
            int to = ctz64(cap_right);
            add_move(moves, count, to + 7, to, PAWN, FLAG_CAPTURE);
            cap_right &= cap_right - 1;
        }
        while (promo_cap_r) {
            int to = ctz64(promo_cap_r);
            add_promotions(moves, count, to + 7, to, FLAG_CAPTURE);
            promo_cap_r &= promo_cap_r - 1;
        }

        // --- En passant ---
        if (pos.ep_square != -1) {
            uint64_t ep_bit = 1ULL << pos.ep_square;
            uint64_t ep_left = ((pawns & NOT_FILE_A) >> 9) & ep_bit;
            if (ep_left)
                add_move(moves, count, pos.ep_square + 9, pos.ep_square, PAWN, FLAG_CAPTURE | FLAG_EP);
            uint64_t ep_right = ((pawns & NOT_FILE_H) >> 7) & ep_bit;
            if (ep_right)
                add_move(moves, count, pos.ep_square + 7, pos.ep_square, PAWN, FLAG_CAPTURE | FLAG_EP);
        }
    }

    return count;
}

// ============================================================
// Move generation — piece moves (knights, bishops, rooks, queens)
// ============================================================
static int gen_piece_moves(const Position& pos, uint64_t friendly,
                           uint64_t enemies, uint64_t occ,
                           Move* moves, int count) {
    int color = pos.color;

    // --- Knights ---
    uint64_t knights = pos.board.bitboards[color * 6 + KNIGHT];
    while (knights) {
        int sq = ctz64(knights);
        uint64_t attacks = KNIGHT_ATTACKS[sq] & ~friendly;
        while (attacks) {
            int to = ctz64(attacks);
            uint8_t flags = (enemies & (1ULL << to)) ? FLAG_CAPTURE : FLAG_NONE;
            add_move(moves, count, sq, to, KNIGHT, flags);
            attacks &= attacks - 1;
        }
        knights &= knights - 1;
    }

    // --- Bishops ---
    uint64_t bishops = pos.board.bitboards[color * 6 + BISHOP];
    while (bishops) {
        int sq = ctz64(bishops);
        uint64_t attacks = get_sliding_attacks(sq, occ, true, false) & ~friendly;
        while (attacks) {
            int to = ctz64(attacks);
            uint8_t flags = (enemies & (1ULL << to)) ? FLAG_CAPTURE : FLAG_NONE;
            add_move(moves, count, sq, to, BISHOP, flags);
            attacks &= attacks - 1;
        }
        bishops &= bishops - 1;
    }

    // --- Rooks ---
    uint64_t rooks = pos.board.bitboards[color * 6 + ROOK];
    while (rooks) {
        int sq = ctz64(rooks);
        uint64_t attacks = get_sliding_attacks(sq, occ, false, true) & ~friendly;
        while (attacks) {
            int to = ctz64(attacks);
            uint8_t flags = (enemies & (1ULL << to)) ? FLAG_CAPTURE : FLAG_NONE;
            add_move(moves, count, sq, to, ROOK, flags);
            attacks &= attacks - 1;
        }
        rooks &= rooks - 1;
    }

    // --- Queens ---
    uint64_t queens = pos.board.bitboards[color * 6 + QUEEN];
    while (queens) {
        int sq = ctz64(queens);
        uint64_t attacks = get_sliding_attacks(sq, occ, true, true) & ~friendly;
        while (attacks) {
            int to = ctz64(attacks);
            uint8_t flags = (enemies & (1ULL << to)) ? FLAG_CAPTURE : FLAG_NONE;
            add_move(moves, count, sq, to, QUEEN, flags);
            attacks &= attacks - 1;
        }
        queens &= queens - 1;
    }

    return count;
}

/
static int gen_king_moves(const Position& pos, uint64_t friendly,
                          uint64_t enemies, uint64_t occ,
                          Move* moves, int count) {
    int color = pos.color;
    int enemy = color ^ 1;
    uint64_t king = pos.board.bitboards[color * 6 + KING];
    if (!king) return count;
    int king_sq = ctz64(king);

    // --- Normal king moves ---
    uint64_t attacks = KING_ATTACKS[king_sq] & ~friendly;
    while (attacks) {
        int to = ctz64(attacks);
        uint8_t flags = (enemies & (1ULL << to)) ? FLAG_CAPTURE : FLAG_NONE;
        add_move(moves, count, king_sq, to, KING, flags);
        attacks &= attacks - 1;
    }

    // --- Castling --- can only generate if king is not currently in check
    if (is_square_attacked(king_sq, enemy, pos.board))
        return count;

    if (color == WHITE) {
        // Kingside: e1(4) -> g1(6), rook h1(7) -> f1(5)
        if (pos.castling[CASTLE_WK]) {
            bool path_clear = !(occ & ((1ULL << 5) | (1ULL << 6)));
            bool path_safe = !is_square_attacked(5, enemy, pos.board) &&
                             !is_square_attacked(6, enemy, pos.board);
            if (path_clear && path_safe)
                add_move(moves, count, 4, 6, KING, FLAG_CASTLE);
        }
        // Queenside: e1(4) -> c1(2), rook a1(0) -> d1(3)
        if (pos.castling[CASTLE_WQ]) {
            bool path_clear = !(occ & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3)));
            bool path_safe = !is_square_attacked(2, enemy, pos.board) &&
                             !is_square_attacked(3, enemy, pos.board);
            if (path_clear && path_safe)
                add_move(moves, count, 4, 2, KING, FLAG_CASTLE);
        }
    } else {
        // Kingside: e8(60) -> g8(62), rook h8(63) -> f8(61)
        if (pos.castling[CASTLE_BK]) {
            bool path_clear = !(occ & ((1ULL << 61) | (1ULL << 62)));
            bool path_safe = !is_square_attacked(61, enemy, pos.board) &&
                             !is_square_attacked(62, enemy, pos.board);
            if (path_clear && path_safe)
                add_move(moves, count, 60, 62, KING, FLAG_CASTLE);
        }
        // Queenside: e8(60) -> c8(58), rook a8(56) -> d8(59)
        if (pos.castling[CASTLE_BQ]) {
            bool path_clear = !(occ & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59)));
            bool path_safe = !is_square_attacked(58, enemy, pos.board) &&
                             !is_square_attacked(59, enemy, pos.board);
            if (path_clear && path_safe)
                add_move(moves, count, 60, 58, KING, FLAG_CASTLE);
        }
    }

    return count;
}

// generates all legal moves
int generate_pseudo_legal(const Position& pos, Move* move_list) {
    int color = pos.color;
    int enemy = color ^ 1;
    uint64_t occ = get_occupancy_board(pos.board);
    uint64_t friendly = side_occupancy(pos.board, color);
    uint64_t enemies = side_occupancy(pos.board, enemy);
    uint64_t empty = ~occ;

    int count = 0;
    count = gen_pawn_moves(pos, enemies, empty, move_list, count);
    count = gen_piece_moves(pos, friendly, enemies, occ, move_list, count);
    count = gen_king_moves(pos, friendly, enemies, occ, move_list, count);
    return count;
}

// make_move — applies move in-place, returns false if illegal
bool make_move(Position& pos, const Move& m) {
    int color = pos.color;
    int enemy = color ^ 1;

    if (m.flags & FLAG_CASTLE) {
        // Move the king
        pos.board.bitboards[color * 6 + KING] ^= (1ULL << m.from) | (1ULL << m.to);

        // Move the rook
        int rook_from, rook_to;
        if (m.to > m.from) {
            // Kingside
            rook_from = (color == WHITE) ? 7 : 63;
            rook_to   = (color == WHITE) ? 5 : 61;
        } else {
            // Queenside
            rook_from = (color == WHITE) ? 0 : 56;
            rook_to   = (color == WHITE) ? 3 : 59;
        }
        pos.board.bitboards[color * 6 + ROOK] ^= (1ULL << rook_from) | (1ULL << rook_to);

        // Revoke all castling rights for this side
        if (color == WHITE) {
            pos.castling[CASTLE_WK] = false;
            pos.castling[CASTLE_WQ] = false;
        } else {
            pos.castling[CASTLE_BK] = false;
            pos.castling[CASTLE_BQ] = false;
        }
    } else {
        // --- Handle captures ---
        if (m.flags & FLAG_CAPTURE) {
            if (m.flags & FLAG_EP) {
                // En passant: captured pawn is on a different square
                int cap_sq = (color == WHITE) ? m.to - 8 : m.to + 8;
                pos.board.bitboards[enemy * 6 + PAWN] &= ~(1ULL << cap_sq);
            } else {
                // Regular capture: remove enemy piece on destination
                for (int p = 0; p < 6; p++) {
                    if (pos.board.bitboards[enemy * 6 + p] & (1ULL << m.to)) {
                        pos.board.bitboards[enemy * 6 + p] &= ~(1ULL << m.to);
                        break;
                    }
                }
            }
        }

        // --- Move the piece ---
        pos.board.bitboards[color * 6 + m.piece] ^= (1ULL << m.from) | (1ULL << m.to);

        // --- Handle promotion ---
        if (m.flags & FLAG_PROMO_ANY) {
            int promo = get_promo_piece(m.flags);
            // Remove pawn from destination, place promoted piece
            pos.board.bitboards[color * 6 + PAWN] ^= (1ULL << m.to);
            pos.board.bitboards[color * 6 + promo] |= (1ULL << m.to);
        }

        // --- Update castling rights ---
        // King moved
        if (m.piece == KING) {
            if (color == WHITE) {
                pos.castling[CASTLE_WK] = false;
                pos.castling[CASTLE_WQ] = false;
            } else {
                pos.castling[CASTLE_BK] = false;
                pos.castling[CASTLE_BQ] = false;
            }
        }
        // Rook moved from its starting square
        if (m.piece == ROOK) {
            if (m.from == 0)  pos.castling[CASTLE_WQ] = false;
            if (m.from == 7)  pos.castling[CASTLE_WK] = false;
            if (m.from == 56) pos.castling[CASTLE_BQ] = false;
            if (m.from == 63) pos.castling[CASTLE_BK] = false;
        }
        // Rook captured on its starting square (enemy loses that right)
        if (m.to == 0)  pos.castling[CASTLE_WQ] = false;
        if (m.to == 7)  pos.castling[CASTLE_WK] = false;
        if (m.to == 56) pos.castling[CASTLE_BQ] = false;
        if (m.to == 63) pos.castling[CASTLE_BK] = false;
    }

    if (m.piece == PAWN && std::abs((int)m.to - (int)m.from) == 16) {
        pos.ep_square = (color == WHITE) ? m.from + 8 : m.from - 8;
    } else {
        pos.ep_square = -1;
    }

    // --- Flip turn ---
    pos.color = enemy;

    uint64_t our_king = pos.board.bitboards[color * 6 + KING];
    if (!our_king) return false;
    int king_sq = ctz64(our_king);
    if (is_square_attacked(king_sq, enemy, pos.board)) {
        return false; /
    }

    return true;
}

// ============================================================
// Perft — exhaustive move tree count for validation
// ============================================================
uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    Move moves[256];
    int count = generate_pseudo_legal(pos, moves);

    uint64_t nodes = 0;
    for (int i = 0; i < count; i++) {
        Position copy = pos; // cheap struct copy for state reversion
        if (make_move(copy, moves[i])) {
            nodes += perft(copy, depth - 1);
        }
    }
    return nodes;
}

void perft_divide(Position& pos, int depth) {
    Move moves[256];
    int count = generate_pseudo_legal(pos, moves);

    uint64_t total = 0;
    const char files[] = "abcdefgh";
    const char ranks[] = "12345678";

    for (int i = 0; i < count; i++) {
        Position copy = pos;
        if (make_move(copy, moves[i])) {
            uint64_t nodes = (depth > 1) ? perft(copy, depth - 1) : 1;
            total += nodes;

            // Print move in algebraic notation
            char from_str[3] = {files[moves[i].from % 8], ranks[moves[i].from / 8], '\0'};
            char to_str[3]   = {files[moves[i].to % 8],   ranks[moves[i].to / 8],   '\0'};
            std::cout << from_str << to_str;

            int promo = get_promo_piece(moves[i].flags);
            if (promo >= 0) {
                // PAWN=0, QUEEN=1, KING=2, BISHOP=3, KNIGHT=4, ROOK=5
                const char pc[] = {' ', 'q', ' ', 'b', 'n', 'r'};
                std::cout << pc[promo];
            }
            std::cout << ": " << nodes << "\n";
        }
    }

    std::cout << "\nTotal: " << total << "\n";
}
