import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import time
from gpu_unpacker import GPUUnpacker
"""
- we pad by 1 to make 9 divisible by 3
- no bias because BatchNorm2d normalizes by subtracting the mean of the batch so it 
would have just been cancelled out regardless

"""
class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    # f(x) = f(x) + x
    def forward(self, x):
        residual = x 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual 
        return self.relu(out)

"""
GPUUnpacker outputs (Batch, 18, 8, 8)
Project that into a channel feature space then pass through ResidualBlocks.
channels is configurable: 128 for GTX 1080, 256 for AlphaZero scale.
"""
class ChessBrainResNet(nn.Module):

    # 64 x 64 = 4096. But vocab size is 4672 set by alphazero
    def __init__(self, num_blocks=8, channels=128, vocab_size=4672):
        super().__init__()
        self.unpacker = GPUUnpacker()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2*8*8, vocab_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, raw_bitboards):
        x = self.unpacker(raw_bitboards)
        x = self.initial_conv(x)

        for block in self.res_blocks:
            x = block(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

def train_loop(dataloader, model, epochs, value_weight=1.0, max_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # use the cuDNN autotuner to find fastest convolution algorithms
    torch.backends.cudnn.benchmark = True 

    # AdamW decouples weight decay from the gradient update, better generalization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing decays LR from 1e-3 → near 0 across all epochs
    # prevents plateauing at a suboptimal local minimum
    total_steps = max_batches * epochs if max_batches else None
    if total_steps:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    else:
        # fallback: step down each epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Device:            {device}")
    print(f"  Epochs:            {epochs}")
    print(f"  Max batches/epoch: {max_batches if max_batches else 'unlimited'}")
    print(f"  Value weight:      {value_weight}")
    print(f"  Optimizer:         AdamW (lr=1e-3, wd=1e-4)")
    print(f"  Scheduler:         CosineAnnealingLR")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters:  {param_count:,}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        
        epoch_start = time.time()
        samples_processed = 0
        running_vloss = 0.0
        running_ploss = 0.0

        for batch_idx, (bitboards, results, move_idxs) in enumerate(dataloader):
            # cap batches per epoch for faster iteration cycles
            if max_batches and batch_idx >= max_batches:
                break

            # async transfer to GPU
            bitboards = bitboards.to(device, non_blocking=True)
            results = results.to(device, non_blocking=True)
            move_idxs = move_idxs.to(device, non_blocking=True)
            
            batch_size = bitboards.size(0)
            samples_processed += batch_size

            optimizer.zero_grad(set_to_none=True)

            policy_logits, values = model(bitboards)

            value_loss = F.mse_loss(values.squeeze(-1), results)
            policy_loss = F.cross_entropy(policy_logits, move_idxs)

            # weight value loss to prevent policy gradients from dominating
            loss = (value_weight * value_loss) + policy_loss 

            loss.backward()

            # gradient clipping prevents exploding gradients during early training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # step the LR scheduler every batch for smooth cosine decay
            if total_steps:
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

        # step scheduler per-epoch if not stepping per-batch
        if not total_steps:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_throughput = samples_processed / epoch_time if epoch_time > 0 else 0
        batches_done = min(batch_idx + 1, max_batches) if max_batches else batch_idx + 1
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

    
def export_to_onnx(model, onnx_file_path="chessbrain.onnx"):

    # .eval comes from nn.Module(). Basically puts the network in inference. Switches BatchNorm off
    # unwrap torch.compile wrapper if present
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.eval()

    # tensorRT traces the graph to understand memory geometry
    dummy_input = torch.zeros((1, 18), dtype=torch.int64)

    # put the dummy tensor with the model weights
    device = next(raw_model.parameters()).device 
    dummy_input = dummy_input.to(device)

    torch.onnx.export(
        raw_model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['Raw_bitboards'],
        output_names=['policy', 'value'],

        # dynamic batching
        dynamic_axes = {
            'Raw_bitboards': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    print(f"Export successful:  {onnx_file_path}")


    