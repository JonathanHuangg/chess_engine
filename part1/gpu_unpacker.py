import torch
import torch.nn as nn

class GPUUnpacker(nn.Module):
    def __init__(self):
        super().__init__()
        # create the 8x8 matrix
        shifts = torch.arange(64, dtype=torch.int64).view(8, 8)

        # buffer is not updated by optimizer during backprop. automatically moved
        # to the same device when you do .to(device)
        self.register_buffer('shifts', shifts)
    
    def forward(self, bitboards):
        # input: (batch, 18) of int64. First add height and width
        boards = bitboards.view(bitboards.shape[0], 18, 1, 1)

        # self.shifts is a grid 0-63. when you right shift with the board
        # [0,0] shifts by 0. [7, 7] shifts by 63

        # IMPORTANT: PyTorch has no uint64. These are int64 (signed).
        # Right-shifting a negative int64 (bit 63 set) does ARITHMETIC shift,
        # copying the sign bit downward. The & 1 mask is CRITICAL: it isolates
        # only the least significant bit, discarding the replicated sign bits.
        # If you ever remove the & 1, the sign extension will silently corrupt
        # every bitboard that has the h8 square (bit 63) occupied.
        unpacked = (boards >> self.shifts) & 1

        # convert to float32
        return unpacked.to(torch.float32)