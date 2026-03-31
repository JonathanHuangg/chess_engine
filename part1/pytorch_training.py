"""

pytorch training loops
goal: 100% GPU utilization

1) Dataset creatin via numpy.memmap()
around 40 GB of chess data so need to make our own memory map w/ lazy loading
The format datatype is one TrainingSample w/ 112 bytes bitboards (14), 4 byte result, 2 byte 
move idx, 2 byte padding. 

from tests, there are 4646902 games. We can double check everything is perfect
by doing size of filepath // 120. 

a) idea 1: ussing getitem if self.data is none
quick thing: we will have an init function and get_item. Because this is multithreaded,
it is better to initialize in the gett_item because every worker should have 
an independent view. If they all use the same memory map, there is i/o traffic.

given 4.6 million games, with 100 epochs, this will be called 460 million times. Doing a null
check on self.data still requires 460 million branch predictions. In addition, workers
are created using fork (linux) or spawn, if this is done in init, fork would copy the file descriptor. 

In addition, with 15 binary files, gett_item is difficult. 

b) idea 2: move all the work into worker_init_fn. 
Puting the idea in worker_init_fn, each worker will have its own unique virtual mapping. 
worker_init_fn can pre-open all of the file handles. When memmap is called, it maps the addresses
so everytime we acess, page fault.

2) Custom collation function.

We can create a function above that does 1 game at a time. GPU can process n games at a time. 
Collation ties it into a batched tensor. Pytorch has collate_fn but it's slow. We have custom 
numpy tuples so we can just build a custom one. 
create a custom collation that takes the tuples, do numpy stack, and convert to a tensor

3) Bit unpacker - we grab 14 integers/bitboards. Resnet doesn't know what that is. 
It needs a 2D spatial grid of floating point numbers. We can reshape in a 8x8 float tensor. 
We can dump this to the GPU and use bitwise math. 
no need for sparse overhead or anything too crazy. 

4) Dataloader from .bin files -> resnet
Feeds the actual data into the GPU. num_workers should be set to CPU core count. Make sure to pin memory.
Make sure you keep the persistent workers so the memory map doesn't need to keep opening.

5) dual head resnet - must accept batch 14, 8, 8 floats
0-5 your move pieces
6-11 opponent move pieces
12 - which side can move
13 - metadata with castling rights, em passant, etc

7) GPU unpacker -> bitwise logic as a separate entity from the resnet

Backbone: 15 Residual Blocks. Each block: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> Add Residual -> ReLU.

Policy Head: Conv(1x1, 2 channels) -> BN -> ReLU -> Flatten -> Linear(128, 4168).

Value Head: Conv(1x1, 1 channel) -> BN -> ReLU -> Flatten -> Linear(64, 256) -> ReLU -> Linear(256, 1) -> Tanh

8) training loop, pushing to GPU, unpacking, forward pass, loss calculation, backprop, optimizer step

8.5) make sure you implement checkpointing. only save the chessbrain weights, not the unpacker

9) ONNX export

create a dummy tensor (1, 14, 8, 8). export the model and run validation

"""
import os
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChessMemmapDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths 

        self.dtype = np.dtype([
            ('bitboards', np.uint64, (14,)),
            ('result', np.float32),
            ('move_idx', np.int16),
            ('padding', np.int16)
        ])

        self.cumulative_sizes = []
        self.memaps = None 

        total_samples = 0 
        
