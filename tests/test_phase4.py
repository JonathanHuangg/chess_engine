#!/usr/bin/env python3
"""Phase 4 smoke tests — run from the repo root:

    python3 tests/test_phase4.py [--skip-neural]

Checks, in order:
  1. The C++ engine builds (engine/Makefile -> perft + play).
  2. Perft passes at all depths 1-6 (the move-generator regression).
  3. The C++ movegen's root move set matches python-chess exactly
     (perft --divide 1 vs. chess.Board().legal_moves).
  4. The C++ MCTS (./play) produces only legal moves.
  5. The Python encoding matches training (bitboard planes +
     policy indices) — pure-Python checks, no torch needed.
  6. [neural] The real checkpoint loads, evaluates the start
     position, and neural MCTS self-play produces only legal moves.

Exit code 0 = all checks passed. A failure prints which check
failed and exits non-zero. Neural checks are skipped when torch
is unavailable (or with --skip-neural) and reported as SKIP.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine")
PART1 = os.path.join(ROOT, "part1")
sys.path.insert(0, PART1)

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        FAILURES.append((name, exc))
        print(f"  FAIL {name}: {exc}")
        return
    print(f"  PASS {name}")


def run(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def test_build():
    r = run(["make", "clean"], cwd=ENGINE)
    assert r.returncode == 0, r.stderr[-500:]
    r = run(["make"], cwd=ENGINE)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "error:" not in r.stderr.lower(), r.stderr[-2000:]
    assert os.path.exists(os.path.join(ENGINE, "perft"))
    assert os.path.exists(os.path.join(ENGINE, "play"))


def test_perft():
    r = run(["./perft"], cwd=ENGINE, timeout=300)
    assert r.returncode == 0, r.stdout[-1000:] + r.stderr[-500:]
    assert "All perft tests PASSED" in r.stdout, r.stdout[-1000:]
    for depth, expected in [(1, 20), (2, 400), (3, 8902), (4, 197281),
                            (5, 4865609), (6, 119060324)]:
        assert re.search(rf"Depth {depth}: {expected} .* PASS", r.stdout), \
            f"depth {depth} line missing in:\n{r.stdout}"


def test_movegen_matches_python_chess():
    import chess

    r = run(["./perft", "--divide", "1"], cwd=ENGINE)
    assert r.returncode == 0, r.stderr[-500:]
    cpp_moves = set()
    for line in r.stdout.splitlines():
        m = re.match(r"^([a-h][1-8][a-h][1-8][qrbn]?): \d+$", line.strip())
        if m:
            cpp_moves.add(m.group(1))
    py_moves = {m.uci() for m in chess.Board().legal_moves}
    assert cpp_moves == py_moves, (
        f"C++ movegen != python-chess at startpos: "
        f"only C++: {sorted(cpp_moves - py_moves)}, "
        f"only python-chess: {sorted(py_moves - cpp_moves)}")
    assert len(cpp_moves) == 20


def test_cpp_mcts_plays_legal_moves():
    import chess

    r = run(["./play", "--sims", "100", "--plies", "10", "--seed", "7"],
            cwd=ENGINE, timeout=120)
    assert r.returncode == 0, r.stderr[-1000:]
    moves = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    assert len(moves) == 10, f"expected 10 moves, got {moves}"
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"illegal move {uci} in {board.fen()}"
        board.push(move)


def test_encoding_matches_training():
    import chess

    from inference import board_to_bitboards, encode_move_index

    # NOTE: planes are int64 (the GPUUnpacker takes int64), so any
    # plane with bit 63 set reads back negative. Compare unsigned.
    def u64(v):
        return int(v) & 0xFFFFFFFFFFFFFFFF

    board = chess.Board()
    planes = board_to_bitboards(board)
    assert planes.shape == (18,)
    # White pawn plane: rank 2 -> bits 8..15 = 0xFF00.
    assert u64(planes[0]) == 0xFF00, hex(u64(planes[0]))
    # White queen plane (QUEEN id 1): d1 -> bit 3.
    assert u64(planes[1]) == 0x08
    # Black rook plane (enemy ROOK id 5 -> plane 11): a8+h8.
    assert u64(planes[11]) == 0x8100000000000000
    # Turn plane: all 1s (int64 -1) for White to move.
    assert u64(planes[12]) == 0xFFFFFFFFFFFFFFFF
    # Castling planes 13-16 all set at the start position.
    assert all(u64(planes[i]) == 0xFFFFFFFFFFFFFFFF for i in (13, 14, 15, 16))
    # No en-passant square.
    assert u64(planes[17]) == 0

    # Black to move: planes flip vertically, turn plane is 0.
    board.push_san("e4")
    planes = board_to_bitboards(board)
    assert u64(planes[12]) == 0
    # Side-to-move (black) pawn plane: black's pawns are all still on
    # rank 7, flipped to rank 2 -> 0xFF00. (The pawn that moved was
    # White's; it shows up in the *enemy* pawn plane, flipped from
    # rank 2 to rank 7 (bits 48-55, minus the e-file pawn) and e4
    # flipped to e5, bit 36.)
    assert u64(planes[0]) == 0xFF00
    assert u64(planes[6]) == (0xFF000000000000 & ~(1 << 52)) | (1 << 36)
    # Policy indices, hand-computed from utils/encode_move_flipped:
    # e2e4: from 12, to 28, N, dist 2 -> type 1 -> 12*73+1 = 877.
    assert encode_move_index(12, 28, True, -1) == 877
    # g1f3: from 6, to 21, knight (dx -1, dy 2) -> type 63 -> 6*73+63 = 501.
    assert encode_move_index(6, 21, True, -1) == 501
    # e7e8=Q (white promo, queen encodes as a normal push):
    # from 52, to 60, N dist 1 -> type 0 -> 52*73 = 3796.
    assert encode_move_index(52, 60, True, 1) == 3796
    # e7e8=N (knight promo, straight): type 64 -> 52*73+64 = 3860.
    assert encode_move_index(52, 60, True, 4) == 3860
    # Black's e7e5 encodes exactly like White's e2e4 (the flip is the
    # whole point of encode_move_flipped).
    assert encode_move_index(52, 36, False, -1) == 877


def test_neural_checkpoint_and_self_play():
    import chess

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  SKIP neural checks: torch not installed")
        return
    from inference import ChessBrain

    ckpt = os.path.join(PART1, "chessbrain_trained.pt")
    assert os.path.exists(ckpt), "checkpoint missing"
    brain = ChessBrain(ckpt)
    logits, value = brain.evaluate(chess.Board())
    assert logits.shape == (4672,), logits.shape
    assert -1.0 <= value <= 1.0, value
    assert brain.calls == 1

    r = run([sys.executable, "inference.py", "--self-play", "--sims", "8",
             "--plies", "6", "--seed", "0"], cwd=PART1, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    moves = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    assert len(moves) == 6, f"expected 6 moves, got {moves}: {r.stderr[-500:]}"
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"illegal move {uci} in {board.fen()}"
        board.push(move)
    print(f"  neural self-play: {' '.join(moves)}")


def main():
    skip_neural = "--skip-neural" in sys.argv
    print("== phase 4 smoke tests ==")
    check("engine builds (make perft + play)", test_build)
    check("perft depths 1-6 pass", test_perft)
    check("C++ movegen == python-chess at startpos",
          test_movegen_matches_python_chess)
    check("C++ MCTS plays 10 legal plies", test_cpp_mcts_plays_legal_moves)
    check("encoding matches training (planes + policy idx)",
          test_encoding_matches_training)
    if skip_neural:
        print("  SKIP neural checks: --skip-neural")
    else:
        check("neural checkpoint + MCTS self-play (real model)",
              test_neural_checkpoint_and_self_play)
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for name, exc in FAILURES:
            print(f"  - {name}: {exc}")
        return 1
    print("ALL PHASE-4 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
