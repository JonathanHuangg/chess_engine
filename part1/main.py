import os
import torch 
from torch.utils.data import DataLoader 

from pytorch_Dataloader import ChessChunkedDataset, fast_collate 
from model import ChessBrainResNet, train_loop, export_to_onnx

def main():
    assert torch.cuda.is_available() or torch.backends.mps.is_available(), "GPU does not exist"

    # ================================================================
    # TUNING KNOBS — adjust these for your 12-hour compute window
    # ================================================================
    NUM_BLOCKS       = 8       # ResNet depth (was 15, reduced for Pascal)
    CHANNELS         = 128     # channel width (was 256, reduced for Pascal)
    BATCH_SIZE       = 2048    # larger batch = better GPU utilization
    EPOCHS           = 4       # target 3-4 epochs in 12 hours
    MAX_BATCHES      = 20000   # cap per epoch for faster iteration (set None for full data)
    VALUE_WEIGHT     = 0.01    # prevent value gradients from dominating policy
    DATALOADER_WORKERS = 4     # 4-8 is optimal, os.cpu_count() causes I/O contention
    # ================================================================

    # Binary files produced by the C++ ETL pipeline
    # The ETL writes one chunk per thread; we read them all
    etl_workers = os.cpu_count() or 16
    binaries = [f"chunk_{i}.bin" for i in range(etl_workers)]
    binaries = [f for f in binaries if os.path.exists(f)]

    if not binaries:
        print("ERROR: No chunk_*.bin files found. Run the C++ ETL pipeline first.")
        return

    print(f"Found {len(binaries)} binary chunk files")

    # Dataset statistics
    total_bytes = sum(os.path.getsize(f) for f in binaries)
    total_samples = total_bytes // 152  # 152 bytes per TrainingSample
    total_batches = total_samples // BATCH_SIZE

    print(f"Dataset Size: {total_bytes / (1024**3):.2f} GB")
    print(f"Total Samples: {total_samples:,}")
    print(f"Batches per Full Epoch: {total_batches:,}")
    if MAX_BATCHES:
        effective_samples = MAX_BATCHES * BATCH_SIZE
        print(f"Capped at: {MAX_BATCHES:,} batches/epoch ({effective_samples:,} samples, "
              f"{effective_samples * 152 / (1024**3):.2f} GB)")

    # DataLoader setup
    dataset = ChessChunkedDataset(binaries, num_workers=DATALOADER_WORKERS, ram_frac=0.25)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS,
        collate_fn=fast_collate,
        pin_memory=True,
        prefetch_factor=4,           # keep 4 batches ready (was 2)
        persistent_workers=True      # don't respawn workers between epochs
    )
    print("Dataloader is ready")

    print("--- setting up model architecture --- ")

    model = ChessBrainResNet(num_blocks=NUM_BLOCKS, channels=CHANNELS, vocab_size=4672)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {NUM_BLOCKS} blocks × {CHANNELS} channels = {param_count:,} parameters")

    print(f"Running {EPOCHS} epochs with value_weight={VALUE_WEIGHT}")
    
    train_loop(dataloader, model, EPOCHS, VALUE_WEIGHT, max_batches=MAX_BATCHES)

    print("---Exporting ONNX---")  
    export_to_onnx(model, "chessbrain.onnx")
    print("Done! Ready for inference")

if __name__ == '__main__':
    main()