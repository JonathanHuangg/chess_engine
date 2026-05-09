"""
Standalone ONNX export script.

Loads a saved checkpoint and exports to ONNX, working around the
int64 bitwise-op tracing limitation in torch.onnx.

The trick: we split the model into two parts for export:
  1. The GPUUnpacker (bitwise ops) is replaced with a no-op identity —
     the ONNX model accepts pre-unpacked float32 (batch, 18, 8, 8) input.
  2. The C++ inference engine does the bitboard unpacking itself
     (or we add a tiny pre-processing ONNX graph).

This is the standard approach used by Lc0 and similar engines.
"""

import argparse
import torch
import torch.nn as nn
from model import ChessBrainResNet


class ChessBrainONNX(nn.Module):
    """
    Wrapper that accepts already-unpacked float32 board planes
    instead of raw int64 bitboards.  This sidesteps the ONNX
    tracing crash on bitwise ops.

    Input:  (batch, 18, 8, 8) float32   ← board planes
    Output: policy (batch, 4672), value (batch, 1)
    """

    def __init__(self, source_model: ChessBrainResNet):
        super().__init__()
        # Copy everything EXCEPT the unpacker
        self.initial_conv = source_model.initial_conv
        self.res_blocks   = source_model.res_blocks
        self.policy_head  = source_model.policy_head
        self.value_head   = source_model.value_head

    def forward(self, board_planes: torch.Tensor):
        # board_planes: (B, 18, 8, 8) float32 — already unpacked
        x = self.initial_conv(board_planes)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


def export(checkpoint_path: str, onnx_path: str, num_blocks: int, channels: int, vocab_size: int):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild the original model and load weights
    model = ChessBrainResNet(num_blocks=num_blocks, channels=channels, vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create the ONNX-safe wrapper (no bitwise ops)
    onnx_model = ChessBrainONNX(model)
    onnx_model.eval()

    # Dummy input: already-unpacked board planes
    dummy = torch.zeros((1, 18, 8, 8), dtype=torch.float32)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        onnx_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["board_planes"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board_planes": {0: "batch_size"},
            "policy":       {0: "batch_size"},
            "value":        {0: "batch_size"},
        },
    )
    print(f"✓ ONNX export successful: {onnx_path}")
    print()
    print("NOTE: The ONNX model expects pre-unpacked float32 (batch, 18, 8, 8) input.")
    print("Your C++ inference engine should unpack bitboards → 18×8×8 planes before feeding ONNX.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained ChessBrain checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="chessbrain_epoch2.pt",
                        help="Path to .pt checkpoint file (default: chessbrain_epoch2.pt)")
    parser.add_argument("--output", default="chessbrain.onnx",
                        help="Output ONNX file path (default: chessbrain.onnx)")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4672)
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.num_blocks, args.channels, args.vocab_size)
