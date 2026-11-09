#!/usr/bin/env python3
"""Phase 4 — neural inference engine: PUCT Monte-Carlo tree search
driven by the trained ChessBrain ResNet (part1/chessbrain_trained.pt).

This is the Python half of the inference engine. The C++ half
(engine/mcts.cpp + engine/play.cpp) runs the same search over the
same move generator with a material evaluator, so the engine plays
standalone; this script swaps in the real network for policy priors
and the value head.

Board encoding (must match training exactly — see
part1/pgn_processor.cpp, "turn to trainingsample"):
  * planes 0-5  = side-to-move's pieces, in order
    PAWN, QUEEN, KING, BISHOP, KNIGHT, ROOK
  * planes 6-11 = opponent's pieces, same order
  * plane 12    = all 1s if White to move, else all 0s
  * planes 13-16 = my kingside, my queenside, enemy kingside,
    enemy queenside castling rights (all 1s / all 0s)
  * plane 17    = en-passant square bit (or 0)
  * when Black is to move, every piece/ep plane is flipped
    vertically (byte-swap: rank r -> rank 7-r), so the network
    always sees the board from the side-to-move's perspective.
  * the value label is likewise from the side-to-move's view.

Policy index for a move is utils.encode_move_flipped():
  index = src_sq * 73 + move_type, with src/dest squares flipped
  (sq ^ 56) when Black is to move. Queen promotions encode as a
  normal pawn move; knight/bishop/rook promotions get move types
  64..72. See encode_move_index() below, a direct port.

Usage:
  python inference.py [--checkpoint part1/chessbrain_trained.pt]
                      [--sims 64] [--plies 8] [--fen FEN]
                      [--seed 0] [--self-play] [--show-top-moves N]

  --self-play plays both sides from the start position (or --fen)
  for --plies plies and prints one UCI move per line on stdout.
  Without --self-play, prints the single best move for the given
  position. Diagnostics go to stderr.
"""

import argparse
import math
import os
import random
import struct
import sys

import numpy as np

# ----------------------------------------------------------------
# Encoding helpers (port of utils/utils.h + pgn_processor.cpp)
# ----------------------------------------------------------------

# utils.h piece ids: PAWN=0, QUEEN=1, KING=2, BISHOP=3, KNIGHT=4, ROOK=5
PIECE_ORDER = ["pawn", "queen", "king", "bishop", "knight", "rook"]

# python-chess piece_type values: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4,
# QUEEN=5, KING=6. Map them onto utils.h ids.
CHESS_TO_UTILS = {1: 0, 5: 1, 6: 2, 3: 3, 2: 4, 4: 5}
UTILS_TO_CHESS = {v: k for k, v in CHESS_TO_UTILS.items()}


def _bswap64(x: int) -> int:
    """Vertical board flip: rank r -> rank 7-r (utils flip_bitboard_vertical)."""
    return struct.unpack("<Q", struct.pack(">Q", x & 0xFFFFFFFFFFFFFFFF))[0]


def encode_move_index(from_sq: int, to_sq: int, color_is_white: bool,
                      promo_utils_piece: int) -> int:
    """Direct port of encode_move_flipped() in utils/utils.h.

    promo_utils_piece: utils.h piece id of the promotion piece, or -1.
    Returns the 0..4671 policy index (src_sq * 73 + move_type).
    """
    src_sq, dest_sq = from_sq, to_sq
    if not color_is_white:
        src_sq ^= 56
        dest_sq ^= 56
    src_x, src_y = src_sq % 8, src_sq // 8
    dest_x, dest_y = dest_sq % 8, dest_sq // 8
    dx, dy = dest_x - src_x, dest_y - src_y

    # KNIGHT=4, BISHOP=3, ROOK=5 in utils.h ids.
    if promo_utils_piece in (4, 3, 5):
        dir_idx = 0
        if dx == -1:
            dir_idx = 1
        elif dx == 1:
            dir_idx = 2
        p_idx = 0
        if promo_utils_piece == 3:
            p_idx = 1
        elif promo_utils_piece == 5:
            p_idx = 2
        move_type = 64 + (p_idx * 3) + dir_idx
    else:
        abs_dx, abs_dy = abs(dx), abs(dy)
        if abs_dx > 0 and abs_dy > 0 and abs_dx + abs_dy == 3:
            knight_map = {(1, 2): 56, (2, 1): 57, (2, -1): 58, (1, -2): 59,
                          (-1, -2): 60, (-2, -1): 61, (-2, 1): 62, (-1, 2): 63}
            move_type = knight_map[(dx, dy)]
        else:
            if dx == 0 and dy > 0:
                dir_idx = 0
            elif dx > 0 and dy > 0:
                dir_idx = 1
            elif dx > 0 and dy == 0:
                dir_idx = 2
            elif dx > 0 and dy < 0:
                dir_idx = 3
            elif dx == 0 and dy < 0:
                dir_idx = 4
            elif dx < 0 and dy < 0:
                dir_idx = 5
            elif dx < 0 and dy == 0:
                dir_idx = 6
            else:  # dx < 0 and dy > 0
                dir_idx = 7
            dist = max(abs_dx, abs_dy)
            move_type = dir_idx * 7 + (dist - 1)
    return src_sq * 73 + move_type


def board_to_bitboards(board) -> np.ndarray:
    """Encode a chess.Board as the (18,) int64 raw-bitboard vector the
    ChessBrainResNet expects (its GPUUnpacker expands it to (18, 8, 8)).

    Square numbering matches the C++ code: bit = rank * 8 + file,
    rank 0 = rank 1. python-chess uses the same convention.
    """
    import chess

    white_to_move = board.turn == chess.WHITE
    planes = np.zeros(18, dtype=np.uint64)
    for utils_id, name in enumerate(PIECE_ORDER):
        chess_pt = UTILS_TO_CHESS[utils_id]
        mine = board.pieces_mask(chess_pt, board.turn)
        theirs = board.pieces_mask(chess_pt, not board.turn)
        if white_to_move:
            planes[utils_id] = mine
            planes[6 + utils_id] = theirs
        else:
            planes[utils_id] = _bswap64(mine)
            planes[6 + utils_id] = _bswap64(theirs)

    planes[12] = 0xFFFFFFFFFFFFFFFF if white_to_move else 0
    # my K, my Q, enemy K, enemy Q castling rights
    planes[13] = 0xFFFFFFFFFFFFFFFF if board.has_kingside_castling_rights(board.turn) else 0
    planes[14] = 0xFFFFFFFFFFFFFFFF if board.has_queenside_castling_rights(board.turn) else 0
    planes[15] = 0xFFFFFFFFFFFFFFFF if board.has_kingside_castling_rights(not board.turn) else 0
    planes[16] = 0xFFFFFFFFFFFFFFFF if board.has_queenside_castling_rights(not board.turn) else 0
    planes[17] = 0
    if board.ep_square is not None:
        ep = 1 << board.ep_square
        planes[17] = ep if white_to_move else _bswap64(ep)
    # GPUUnpacker takes int64; bit 63 set means a negative int64.
    return planes.view(np.int64)


# ----------------------------------------------------------------
# Network wrapper
# ----------------------------------------------------------------

class ChessBrain:
    """Loads a trained checkpoint and evaluates positions."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch

        from model import ChessBrainResNet

        self.torch = torch
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        num_blocks = ckpt.get("num_blocks", 8)
        channels = ckpt.get("channels", 128)
        vocab_size = ckpt.get("vocab_size", 4672)
        self.model = ChessBrainResNet(num_blocks=num_blocks,
                                      channels=channels,
                                      vocab_size=vocab_size)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.calls = 0

    @property
    def info(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"{n:,} parameters"

    def evaluate(self, board):
        """Return (policy_logits np.array[4672], value float) for board.

        Value is from the side-to-move's perspective, in [-1, 1].
        """
        import torch

        raw = board_to_bitboards(board)
        with torch.no_grad():
            x = torch.from_numpy(raw).unsqueeze(0).to(self.device)
            logits, value = self.model(x)
        self.calls += 1
        return (logits[0].detach().cpu().numpy(),
                float(value[0, 0].detach().cpu().numpy()))


# ----------------------------------------------------------------
# PUCT MCTS (AlphaZero-style)
# ----------------------------------------------------------------

class Node:
    __slots__ = ("board", "move", "parent", "children", "prior",
                 "visits", "value_sum", "expanded", "terminal_value")

    def __init__(self, board, move=None, parent=None, prior=0.0):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = {}       # chess.Move -> Node
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.expanded = False
        self.terminal_value = None

    def q(self) -> float:
        # Value from the *parent's* side-to-move view: negate, because
        # value_sum is accumulated from this node's side-to-move view.
        return -self.value_sum / self.visits if self.visits else 0.0


class Mcts:
    def __init__(self, brain: ChessBrain, simulations: int = 64,
                 cpuct: float = 1.5, seed: int = 0):
        self.brain = brain
        self.simulations = simulations
        self.cpuct = cpuct
        self.rng = random.Random(seed)

    def search(self, board):
        """Run the search from `board`; return (best_move, root_node)."""
        import chess

        root = Node(board.copy())
        # Expand the root immediately so priors exist.
        self._expand(root)
        for _ in range(self.simulations):
            node = root
            # Selection: descend by PUCT until an unexpanded node.
            while node.expanded and node.children:
                node = self._select_child(node)
            value = self._expand(node)
            self._backup(node, value)
        # Most-visited child wins.
        best_move, best_child = max(root.children.items(),
                                    key=lambda kv: kv[1].visits)
        return best_move, root

    def _select_child(self, node: Node) -> Node:
        sqrt_n = math.sqrt(node.visits + 1)
        best, best_score = None, -1e30
        for child in node.children.values():
            score = child.q() + self.cpuct * child.prior * sqrt_n / (1 + child.visits)
            score += self.rng.random() * 1e-6  # deterministic tie-break
            if score > best_score:
                best, best_score = child, score
        return best

    def _expand(self, node: Node) -> float:
        """Evaluate node.board with the network (or terminal rules) and
        create its children. Returns the value from the side-to-move's
        perspective at this node."""
        import chess

        board = node.board
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                node.terminal_value = 0.0
            elif outcome.winner == board.turn:
                node.terminal_value = 1.0
            else:
                node.terminal_value = -1.0
            node.expanded = True
            return node.terminal_value

        logits, value = self.brain.evaluate(board)
        legal = list(board.legal_moves)
        # Softmax over legal moves only, using the trained policy head.
        idxs = []
        for m in legal:
            promo = -1
            if m.promotion is not None:
                promo = CHESS_TO_UTILS[m.promotion]
            idxs.append(encode_move_index(m.from_square, m.to_square,
                                          board.turn == chess.WHITE, promo))
        idxs = np.asarray(idxs)
        masked = logits[idxs]
        masked = masked - masked.max()
        exp = np.exp(masked)
        priors = exp / exp.sum()
        for m, p in zip(legal, priors):
            child_board = board.copy()
            child_board.push(m)
            node.children[m] = Node(child_board, move=m, parent=node,
                                    prior=float(p))
        node.expanded = True
        return value

    def _backup(self, node: Node, value: float):
        # Negate at every ply: the perspective flips each move.
        cur, v = node, value
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = -v
            cur = cur.parent


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ChessBrain phase-4 inference: neural MCTS move picker.")
    parser.add_argument("--checkpoint",
                        default=os.path.join(os.path.dirname(__file__),
                                             "chessbrain_trained.pt"),
                        help="trained checkpoint (default: chessbrain_trained.pt)")
    parser.add_argument("--sims", type=int, default=64,
                        help="MCTS simulations per move (default: 64)")
    parser.add_argument("--plies", type=int, default=8,
                        help="plies to play in --self-play mode (default: 8)")
    parser.add_argument("--fen", default=None,
                        help="FEN to start from (default: start position)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-play", action="store_true",
                        help="play both sides for --plies plies")
    parser.add_argument("--show-top-moves", type=int, default=0,
                        help="print the N most-visited root moves to stderr")
    args = parser.parse_args(argv)

    try:
        import chess
    except ImportError:
        print("error: python-chess is required (pip install python-chess)",
              file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401
    except ImportError:
        print("error: torch is required (pip install torch)", file=sys.stderr)
        return 2

    if not os.path.exists(args.checkpoint):
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    print(f"loading {args.checkpoint} ...", file=sys.stderr)
    brain = ChessBrain(args.checkpoint)
    print(f"model ready: {brain.info}", file=sys.stderr)
    mcts = Mcts(brain, simulations=args.sims, seed=args.seed)

    board = chess.Board(args.fen) if args.fen else chess.Board()
    plies = args.plies if args.self_play else 1
    for ply in range(1, plies + 1):
        if board.is_game_over():
            print(f"game over: {board.outcome()}", file=sys.stderr)
            break
        best, root = mcts.search(board)
        san = board.san(best)
        value = root.value_sum / root.visits if root.visits else 0.0
        print(best.uci())
        sys.stdout.flush()
        print(f"ply {ply} ({'white' if board.turn else 'black'}): "
              f"{san} ({best.uci()}) visits={root.children[best].visits} "
              f"root_value={value:+.3f} net_calls={brain.calls}",
              file=sys.stderr)
        if args.show_top_moves:
            ranked = sorted(root.children.items(),
                            key=lambda kv: -kv[1].visits)[:args.show_top_moves]
            for m, child in ranked:
                print(f"    {board.san(m):>6} visits={child.visits:4d} "
                      f"prior={child.prior:.3f} q={child.q():+.3f}",
                      file=sys.stderr)
        board.push(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
