#ifndef MCTS_H
#define MCTS_H

#include "movegen.h"
#include <vector>

// ============================================================
// Phase 4 — Monte-Carlo Tree Search over the engine's own
// move generator (engine/movegen.cpp).
//
// The search is evaluator-agnostic: any object that scores a
// position from the side-to-move's perspective in [-1, 1]
// (1 = side to move is winning) can drive it.
//
//   * MaterialEvaluator (below) is a dependency-free heuristic
//     fallback, so the C++ engine plays standalone.
//   * The trained network drives the search from Python — see
//     part1/inference.py, which loads part1/chessbrain_trained.pt
//     and runs the same PUCT search with the ResNet's policy
//     priors and value head.
// ============================================================

class Evaluator {
public:
    virtual ~Evaluator() = default;
    // Score `pos` from the side-to-move's perspective, in [-1, 1].
    virtual float evaluate(const Position& pos) = 0;
};

// Cheap material evaluator: tanh(material advantage), so the
// C++ engine can play without the neural network. Deliberately
// simple — the network is the real evaluator (inference.py).
class MaterialEvaluator : public Evaluator {
public:
    float evaluate(const Position& pos) override;
};

// One search tree node. Nodes live in an arena (vector<Node>)
// inside Mcts; children are indices into that arena.
struct MctsNode {
    Move move;               // move from parent to this node (root: FLAG_NONE)
    int parent = -1;
    std::vector<int> children;
    float prior = 0.0f;      // prior probability of this move (uniform in C++)
    int visits = 0;
    float value_sum = 0.0f;  // accumulated value, from this node's side-to-move view
    bool expanded = false;
    bool terminal = false;
    float terminal_value = 0.0f;  // valid when terminal: side-to-move result
};

class Mcts {
public:
    Mcts(Evaluator& evaluator, int simulations, float cpuct = 1.5f,
         unsigned int seed = 0);

    // Run the search from `root_pos` and return the most-visited move.
    // Returns false when there is no legal move (checkmate / stalemate).
    bool best_move(const Position& root_pos, Move& out_move);

    // Diagnostics from the last search.
    int root_visits() const { return root_visits_; }
    float root_value() const { return root_value_; }

private:
    struct RootMoveInfo {
        Move move;
        int node_idx;
    };

    int select(int node_idx);
    // Expand node `node_idx` (position `pos`); returns leaf value
    // from the side-to-move's perspective at that node.
    float expand_and_evaluate(int node_idx, const Position& pos);
    void backup(int node_idx, float value);

    Evaluator& evaluator_;
    int simulations_;
    float cpuct_;
    unsigned int rng_state_;

    std::vector<MctsNode> nodes_;
    std::vector<RootMoveInfo> root_moves_;
    int root_visits_ = 0;
    float root_value_ = 0.0f;
};

// Convenience wrapper: run a search and return the best move.
// Returns false when no legal move exists.
bool mcts_best_move(const Position& pos, Evaluator& evaluator,
                    int simulations, Move& out_move, unsigned int seed = 0);

#endif
