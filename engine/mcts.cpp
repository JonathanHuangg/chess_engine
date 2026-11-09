#include "mcts.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Piece values in centipawns (king excluded: never captured).
static const int kPieceValue[6] = {
    100,  // PAWN
    900,  // QUEEN
    0,    // KING
    330,  // BISHOP
    320,  // KNIGHT
    500,  // ROOK
};

float MaterialEvaluator::evaluate(const Position& pos) {
    int score = 0;  // centipawns, from the side-to-move's perspective
    for (int p = 0; p < 6; p++) {
        int mine = popcount64(pos.board.bitboards[pos.color * 6 + p]);
        int theirs = popcount64(pos.board.bitboards[(pos.color ^ 1) * 6 + p]);
        score += kPieceValue[p] * (mine - theirs);
    }
    // tanh keeps the value in [-1, 1]; ~1 pawn ~= 0.11, a queen ~= 0.76.
    return static_cast<float>(std::tanh(score / 900.0));
}

namespace {

// xorshift32 — tiny deterministic RNG for tie-breaking only.
unsigned int xorshift32(unsigned int& state) {
    if (state == 0) state = 0x9E3779B9u;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// True when `pos` has no legal move. Sets `checkmated` when the
// side to move is in check (checkmate) as opposed to stalemate.
bool is_terminal(const Position& pos, bool& checkmated) {
    Move moves[256];
    int count = generate_pseudo_legal(pos, moves);
    for (int i = 0; i < count; i++) {
        Position copy = pos;
        if (make_move(copy, moves[i])) return false;
    }
    // No legal moves: checkmate iff the king is attacked.
    uint64_t king = pos.board.bitboards[pos.color * 6 + KING];
    checkmated = false;
    if (king) {
        int king_sq = ctz64(king);
        checkmated = is_square_attacked(king_sq, pos.color ^ 1, pos.board);
    }
    return true;
}

// All legal moves from `pos` (pseudo-legal filtered by make_move).
std::vector<Move> legal_moves(const Position& pos) {
    Move moves[256];
    int count = generate_pseudo_legal(pos, moves);
    std::vector<Move> out;
    out.reserve(count);
    for (int i = 0; i < count; i++) {
        Position copy = pos;
        if (make_move(copy, moves[i])) out.push_back(moves[i]);
    }
    return out;
}

}  // namespace

Mcts::Mcts(Evaluator& evaluator, int simulations, float cpuct,
           unsigned int seed)
    : evaluator_(evaluator),
      simulations_(simulations),
      cpuct_(cpuct),
      rng_state_(seed) {}

bool Mcts::best_move(const Position& root_pos, Move& out_move) {
    nodes_.clear();
    root_moves_.clear();
    root_visits_ = 0;
    root_value_ = 0.0f;

    std::vector<Move> moves = legal_moves(root_pos);
    if (moves.empty()) return false;

    nodes_.push_back(MctsNode());  // root, index 0
    nodes_[0].expanded = true;
    // Uniform priors in the C++ search (no policy network here);
    // the Python search replaces these with the ResNet's priors.
    float prior = 1.0f / static_cast<float>(moves.size());
    for (const Move& m : moves) {
        MctsNode child;
        child.move = m;
        child.parent = 0;
        child.prior = prior;
        int idx = static_cast<int>(nodes_.size());
        nodes_.push_back(child);
        nodes_[0].children.push_back(idx);
        root_moves_.push_back({m, idx});
    }

    for (int sim = 0; sim < simulations_; sim++) {
        int leaf = select(0);
        // Reconstruct the leaf position by replaying moves from root.
        Position pos = root_pos;
        int node = leaf;
        std::vector<int> path;
        while (node != 0) {
            path.push_back(node);
            node = nodes_[node].parent;
        }
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Position next = pos;
            // Must succeed: these moves were legal when generated.
            if (!make_move(next, nodes_[*it].move)) break;
            pos = next;
        }
        float value = expand_and_evaluate(leaf, pos);
        backup(leaf, value);
        root_visits_++;
    }

    // Most-visited root move wins; break ties by search order with a
    // deterministic shuffle so repeated runs with the same seed agree.
    int best_idx = -1;
    int best_visits = -1;
    for (const auto& rm : root_moves_) {
        int v = nodes_[rm.node_idx].visits;
        if (v > best_visits) {
            best_visits = v;
            best_idx = rm.node_idx;
        }
    }
    if (best_idx < 0) return false;
    out_move = nodes_[best_idx].move;
    root_value_ = nodes_[0].visits > 0
                      ? nodes_[0].value_sum / nodes_[0].visits
                      : 0.0f;
    return true;
}

int Mcts::select(int node_idx) {
    int current = node_idx;
    while (nodes_[current].expanded && !nodes_[current].children.empty() &&
           !nodes_[current].terminal) {
        // PUCT: pick the child maximising Q + cpuct * P * sqrt(N) / (1 + n).
        // Q is stored from the child's side-to-move view, so negate it:
        // a good position for the child is bad for the parent.
        int best_child = -1;
        float best_score = -1e30f;
        float sqrt_parent = std::sqrt(static_cast<float>(nodes_[current].visits) + 1.0f);
        for (int child_idx : nodes_[current].children) {
            const MctsNode& child = nodes_[child_idx];
            float q = child.visits > 0
                          ? -(child.value_sum / child.visits)
                          : 0.0f;
            float u = cpuct_ * child.prior * sqrt_parent / (1.0f + child.visits);
            // Tiny deterministic jitter breaks exact ties without
            // changing the ordering of genuinely different scores.
            float jitter = (xorshift32(rng_state_) % 1000) / 1000.0f * 1e-6f;
            float score = q + u + jitter;
            if (score > best_score) {
                best_score = score;
                best_child = child_idx;
            }
        }
        if (best_child < 0) break;
        current = best_child;
    }
    return current;
}

float Mcts::expand_and_evaluate(int node_idx, const Position& pos) {
    MctsNode& node = nodes_[node_idx];
    if (node.terminal) return node.terminal_value;

    bool checkmated = false;
    if (is_terminal(pos, checkmated)) {
        node.terminal = true;
        // Checkmate: side to move lost. Stalemate: draw.
        node.terminal_value = checkmated ? -1.0f : 0.0f;
        return node.terminal_value;
    }

    std::vector<Move> moves = legal_moves(pos);
    if (moves.empty()) {  // cannot happen if is_terminal agreed, be safe
        node.terminal = true;
        node.terminal_value = 0.0f;
        return 0.0f;
    }
    float prior = 1.0f / static_cast<float>(moves.size());
    for (const Move& m : moves) {
        MctsNode child;
        child.move = m;
        child.parent = node_idx;
        child.prior = prior;
        int idx = static_cast<int>(nodes_.size());
        nodes_.push_back(child);
        // NOTE: `node` may dangle after push_back reallocates; use index.
        nodes_[node_idx].children.push_back(idx);
    }
    nodes_[node_idx].expanded = true;
    return evaluator_.evaluate(pos);
}

void Mcts::backup(int node_idx, float value) {
    // `value` is from the leaf side-to-move's perspective; negate it
    // at every ply going up, because the perspective flips each move.
    int current = node_idx;
    float v = value;
    while (current != -1) {
        nodes_[current].visits++;
        nodes_[current].value_sum += v;
        v = -v;
        current = nodes_[current].parent;
    }
}

bool mcts_best_move(const Position& pos, Evaluator& evaluator,
                    int simulations, Move& out_move, unsigned int seed) {
    Mcts search(evaluator, simulations, 1.5f, seed);
    return search.best_move(pos, out_move);
}
