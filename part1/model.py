import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
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
Project that into 256-channel feature space
then pass it through 15 ResidualBlocks
"""
class ChessBrainResNet(nn.Module):

    # 64 x 64 = 4096. But vocab size is 4672 set by alphazero
    def __init__(self, num_blocks=15, vocab_size=4672):
        super().__init__()
        self.unpacker = GPUUnpacker()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(18, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # pass through the 15 block residual layer
        self.res_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(num_blocks)])


        # policy and value extract whatever we need on top

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2*8*8, vocab_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
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

def train_loop(dataloader, model, epochs, value_weight=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # use the cuDNN autotuner
    torch.backends.cudnn.benchmark = True 

    optimizer = optim.Adam(model.parameters(), lr=1e-3, fused=True)

    for epoch in range(epochs):
        model.train()

        for batch_idx, (bitboards, results, move_idxs) in enumerate(dataloader):
            # async transfer to GPU
            bitboards = bitboards.to(device, non_blocking=True)
            results = results.to(device, non_blocking=True)
            move_idxs = move_idxs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            policy_logits, values = model(bitboards)

            value_loss = F.mse_loss(values.squeeze(-1), results)
            policy_loss = F.cross_entropy(policy_logits, move_idxs)

            # weight value loss to prevent policy gradients from dominating
            loss = (value_weight * value_loss) + policy_loss 

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch} | Batch: {batch_idx} | V: {value_loss.item():.4f} | P: {policy_loss.item():.4f} | Loss: {loss.item():.4f}")