#include <iostream>
#include <string>
#include "utils.h"


uint64_t KNIGHT_ATTACKS[64];
uint64_t WHITE_PAWN_ATTACKS[64];
uint64_t BLACK_PAWN_ATTACKS[64];
uint64_t KING_ATTACKS[64];
uint64_t FILE_MASKS[8];
uint64_t RANK_MASKS[8];

/*0=N, 1=S, 2=E, 3=W, 4=NE, 5=NW, 6=SE, 7=SW

This means that bishops go 4->7, rooks 0->3, queens 0->7
*/
uint64_t RAY_MASKS[64][8];

// note:
// index = color * 6 + piece
// white pawn = 1, black queen = 0 + 1 = 1
// getting all white knights = boards[(WHITE * 6) + KNIGHT]


// knight_attacks, rank masks, file masks, king attacks
void init_lookup_tables() {
    // file and rank
    for (int i = 0; i < 8; i++) {
        FILE_MASKS[i] = 0x0101010101010101ULL << i;
        RANK_MASKS[i] = 0x00000000000000FFULL << (i * 8);
    }

    for (int sq = 0; sq < 64; sq++) {
        int file = sq % 8;
        int rank = sq / 8;

        KNIGHT_ATTACKS[sq] = 0ULL;
        KING_ATTACKS[sq] = 0ULL;

        int knight_moves[8][2] = {{1, 2}, {-1, 2}, {1, -2}, {-1, -2}, {2, 1}, {-2, 1}, {2, -1}, {-2,-1}};
        for (auto&move : knight_moves) {
            int add_f = file + move[0];
            int add_r = rank + move[1];

            // check for bounds
            if (add_f >= 0 && add_f < 8 && add_r >= 0 && add_r < 8) {
                int set_sq = add_r * 8 + add_f;
                KNIGHT_ATTACKS[sq] |= (1ULL << set_sq);
            }

        }

        int king_moves[8][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}, {-1, 0}, {0, -1}, {-1, -1}, {-1, 1}};

        for (auto& move : king_moves) {
            int add_f = file + move[0];
            int add_r = rank + move[1];

            if (add_f >= 0 && add_f < 8 && add_r >= 0 && add_r < 8) {
                int set_sq = add_r * 8 + add_f;
                KING_ATTACKS[sq] |= (1ULL << set_sq);
            }
        }
    }
}

// ray_masks are only half the battle. need occupancy board
void init_ray_masks() {
    for (int sq = 0; sq < 64; sq++) {
        int file = sq % 8;
        int row = sq / 8;

        for (int dir = 0; dir < 8; dir++) {
            RAY_MASKS[sq][dir] = 0ULL;
        }

        // north
        for (int i = row + 1; i < 8; ++i) 
            RAY_MASKS[sq][0] |= (1ULL << (i * 8 + file));
        
        // south
        for (int i = row - 1; i >= 0; --i) 
            RAY_MASKS[sq][1] |= (1ULL << (i * 8 + file));
        
        // east
        for (int i = file + 1; i < 8; ++i) 
            RAY_MASKS[sq][2] |= (1ULL << (row * 8 + i));
        
        // west
        for (int i = file - 1; i >= 0; --i) 
            RAY_MASKS[sq][3] |= (1ULL << (row * 8 + i));

        // NE
        for (int i = 1; row + i < 8 && file + i < 8; ++i) 
            RAY_MASKS[sq][4] |= (1ULL << ((row + i) * 8 + (file + i)));
            
        // NW
        for (int i = 1; row + i < 8 && file - i >= 0; ++i) 
            RAY_MASKS[sq][5] |= (1ULL << ((row + i) * 8 + (file - i)));
            
        // SE
        for (int i = 1; row - i >= 0 && file + i < 8; ++i) 
            RAY_MASKS[sq][6] |= (1ULL << ((row - i) * 8 + (file + i)));
            
        // SW
        for (int i = 1; row - i >= 0 && file - i >= 0; ++i) 
            RAY_MASKS[sq][7] |= (1ULL << ((row - i) * 8 + (file - i)));
    }
}

uint64_t get_occupancy_board(const BoardState &board) {
    uint64_t occ = 0ULL;
    for (int i = 0; i < 12; i++) {
        occ |= board.bitboards[i];
    }
    return occ;
}
// finally able to now get all of the sliding attacks
uint64_t get_sliding_attacks(int sq, uint64_t occ, bool is_diagonal, bool is_orthogonal) {
    uint64_t attacks = 0ULL;
    //north is add, south is sub etc
    int pos_rays[4] = {0, 2, 4, 5}; // north, east, northeast, northwest
    int neg_rays[4] = {1, 3, 6, 7}; // south, west, southeast, southwest

    // north, west, south, east only
    if (is_orthogonal) {
        for (int i = 0; i < 2; i++) {
            int dir = pos_rays[i];
            uint64_t ray = RAY_MASKS[sq][dir];
            uint64_t blockers = ray & occ;

            if (blockers) {
                int blockers_sq = __builtin_ctzll(blockers); // find the lowest set bit
                ray ^= RAY_MASKS[blockers_sq][dir]; // wipe the shadow
            }
            attacks |= ray;
        }

        for (int i = 0; i < 2; i++) {
            int dir = neg_rays[i];
            uint64_t ray = RAY_MASKS[sq][dir];
            uint64_t blockers = ray & occ; 

            if (blockers) {
                int blocker_sq = 63 - __builtin_clzll(blockers);
                ray ^= RAY_MASKS[blocker_sq][dir];
            }
            attacks |= ray;
        }
    }

    if (is_diagonal) {
        for (int i = 2; i < 4; i++) {
            int dir = pos_rays[i];
            uint64_t ray = RAY_MASKS[sq][dir];
            uint64_t blockers = ray & occ;
            if (blockers) {
                int blocker_sq = __builtin_ctzll(blockers);
                ray ^= RAY_MASKS[blocker_sq][dir];
            }
            attacks |= ray;
        }
        for (int i = 2; i < 4; ++i) {
            int dir = neg_rays[i];
            uint64_t ray = RAY_MASKS[sq][dir];
            uint64_t blockers = ray & occ;
            if (blockers) {
                int blocker_sq = 63 - __builtin_clzll(blockers);
                ray ^= RAY_MASKS[blocker_sq][dir];
            }
            attacks |= ray;
        }
    }
    return attacks;
}

void init_pawn_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        WHITE_PAWN_ATTACKS[sq] = 0ULL;
        BLACK_PAWN_ATTACKS[sq] = 0ULL;

        int f = sq % 8;
        int r = sq / 8;

        // white goes nw and ne
        if (r < 7) {
            if (f > 0) {
                WHITE_PAWN_ATTACKS[sq] |= (1ULL << (sq + 7)); //NW
            }

            if (f < 7) {
                WHITE_PAWN_ATTACKS[sq] |= (1ULL << (sq + 9)); //NE
            }
        }
        // black captures sw and se
        if (r > 0) {
            if (f > 0) BLACK_PAWN_ATTACKS[sq] |= (1ULL << (sq - 9)); // SW
            if (f < 7) BLACK_PAWN_ATTACKS[sq] |= (1ULL << (sq - 7)); // SE
        }
    }
}
void set_starting_position(BoardState &board) {  
    for (int i = 0; i < 14; i++) {
        board.bitboards[i] = 0ULL;
    }

    // white pieces
    board.bitboards[(WHITE * 6) + PAWN] = 0x000000000000FF00ULL;
    board.bitboards[(WHITE * 6) + KNIGHT] = 0x0000000000000042ULL;
    board.bitboards[(WHITE * 6) + BISHOP] = 0x0000000000000024ULL;
    board.bitboards[(WHITE * 6) + ROOK] = 0x0000000000000081ULL;
    board.bitboards[(WHITE * 6) + QUEEN] = 0x0000000000000008ULL;
    board.bitboards[(WHITE * 6) + KING] = 0x0000000000000010ULL;

    // black pieces
    board.bitboards[(BLACK * 6) + PAWN] = 0x00FF000000000000ULL;
    board.bitboards[(BLACK * 6) + KNIGHT] = 0x4200000000000000ULL;
    board.bitboards[(BLACK * 6) + BISHOP] = 0x2400000000000000ULL;
    board.bitboards[(BLACK * 6) + ROOK] = 0x8100000000000000ULL;
    board.bitboards[(BLACK * 6) + QUEEN] = 0x0800000000000000ULL;
    board.bitboards[(BLACK * 6) + KING] = 0x1000000000000000ULL;

}