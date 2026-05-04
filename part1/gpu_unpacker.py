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

        # &1 just checks if the number is asserted
        # this is the equivalent is saying (rook here (1), rook not here (0))
        unpacked = (boards >> self.shifts) & 1

        # convert to float32
        return unpacked.to(torch.float32)