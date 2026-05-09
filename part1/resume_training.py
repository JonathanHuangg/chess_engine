"""
Resume training from a saved checkpoint.

Usage:
    python resume_training.py                           # resumes from epoch 2, trains epoch 3
    python resume_training.py --checkpoint chessbrain_epoch1.pt --start-epoch 2
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from pytorch_Dataloader import ChessChunkedDataset, fast_collate
from model import ChessBrainResNet, train_loop


def main():
    parser = argparse.ArgumentParser(description="Resume ChessBrain training from checkpoint")
    parser.add_argument("--checkpoint", default="chessbrain_epoch2.pt",
                        help="Checkpoint to resume from (default: chessbrain_epoch2.pt)")
    parser.add_argument("--start-epoch", type=int, default=4,
                        help="Epoch number to start from (default: 4)")
    parser.add_argument("--total-epochs", type=int, default=5,
                        help="Total epochs to reach (default: 5)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU required"

    # ── Same hyperparameters as main.py ──
    NUM_BLOCKS       = 8
    CHANNELS         = 128
    BATCH_SIZE       = 2048
    MAX_BATCHES      = 20000
    VALUE_WEIGHT     = 1
    DATALOADER_WORKERS = 4

    remaining_epochs = args.total_epochs - args.start_epoch
    if remaining_epochs <= 0:
        print(f"Already trained {args.total_epochs} epochs. Nothing to do.")
        return

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # ── Rebuild model ──
    model = ChessBrainResNet(num_blocks=NUM_BLOCKS, channels=CHANNELS, vocab_size=4672)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Restored model from epoch {ckpt['epoch']}")
    print(f"  Avg value loss:  {ckpt.get('avg_value_loss', 'N/A')}")
    print(f"  Avg policy loss: {ckpt.get('avg_policy_loss', 'N/A')}")

    # ── DataLoader (same as main.py) ──
    etl_workers = os.cpu_count() or 16
    binaries = [f"chunk_{i}.bin" for i in range(etl_workers)]
    binaries = [f for f in binaries if os.path.exists(f)]

    if not binaries:
        print("ERROR: No chunk_*.bin files found.")
        return

    print(f"Found {len(binaries)} binary chunk files")

    dataset = ChessChunkedDataset(binaries, num_workers=DATALOADER_WORKERS, ram_frac=0.25)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=fast_collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    print("Dataloader is ready")

    # ── Resume training ──
    
    import torch.nn.functional as F
    import torch.optim as optim
    import time

    device = torch.device("cuda")
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = MAX_BATCHES * args.total_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Restored optimizer & scheduler state")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nResuming training: epochs {args.start_epoch} → {args.total_epochs - 1}")
    print(f"{'='*60}\n")

    for epoch in range(args.start_epoch, args.total_epochs):
        model.train()

        epoch_start = time.time()
        samples_processed = 0
        running_vloss = 0.0
        running_ploss = 0.0

        for batch_idx, (bitboards, results, move_idxs) in enumerate(dataloader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break

            bitboards = bitboards.to(device, non_blocking=True)
            results   = results.to(device, non_blocking=True)
            move_idxs = move_idxs.to(device, non_blocking=True)

            batch_size = bitboards.size(0)
            samples_processed += batch_size

            optimizer.zero_grad(set_to_none=True)

            policy_logits, values = model(bitboards)

            value_loss  = F.mse_loss(values.squeeze(-1), results)
            policy_loss = F.cross_entropy(policy_logits, move_idxs)
            loss = (VALUE_WEIGHT * value_loss) + policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_vloss += value_loss.item()
            running_ploss += policy_loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                throughput = samples_processed / elapsed if elapsed > 0 else 0
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | Batch {batch_idx:>6d} | "
                      f"V: {value_loss.item():.4f} | P: {policy_loss.item():.4f} | "
                      f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                      f"{throughput:.0f} samp/s")

        epoch_time = time.time() - epoch_start
        avg_throughput = samples_processed / epoch_time if epoch_time > 0 else 0
        batches_done = min(batch_idx + 1, MAX_BATCHES)
        avg_vloss = running_vloss / batches_done if batches_done > 0 else 0
        avg_ploss = running_ploss / batches_done if batches_done > 0 else 0

        print(f"\n--- Epoch {epoch} Complete ---")
        print(f"  Time:           {epoch_time:.1f}s ({epoch_time/3600:.2f}h)")
        print(f"  Batches:        {batches_done:,}")
        print(f"  Samples:        {samples_processed:,}")
        print(f"  Avg Throughput: {avg_throughput:,.0f} samples/s")
        print(f"  Avg Value Loss: {avg_vloss:.4f}")
        print(f"  Avg Policy Loss:{avg_ploss:.4f}")
        print(f"  Final LR:       {optimizer.param_groups[0]['lr']:.2e}\n")

        # Save checkpoint
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        ckpt_path = f"chessbrain_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'avg_value_loss': avg_vloss,
            'avg_policy_loss': avg_ploss,
        }, ckpt_path)
        print(f"  ✓ Checkpoint saved: {ckpt_path}")

    # ── Save final PyTorch checkpoint ──
    final_path = "chessbrain_trained.pt"
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model_state_dict': raw.state_dict(),
        'num_blocks': NUM_BLOCKS,
        'channels': CHANNELS,
        'vocab_size': 4672,
        'epochs_trained': args.total_epochs,
    }, final_path)
    print(f"\n✓ Final checkpoint saved: {final_path}")

    # ── ONNX export ──
    print("\n--- Exporting ONNX ---")
    try:
        from export_onnx import export
        export(final_path, "chessbrain.onnx", NUM_BLOCKS, CHANNELS, 4672)
    except Exception as e:
        print(f"\n⚠ ONNX export failed: {e}")
        print(f"Your weights are safe in '{final_path}'")
        print("Run manually: python export_onnx.py --checkpoint chessbrain_trained.pt")


if __name__ == "__main__":
    main()
