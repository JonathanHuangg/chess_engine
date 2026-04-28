"""
DataLoader

1) Pytorch tells OS to spawn independent worker processes based on num_threads
2) Get system RAM. Allocate so threads process enough RAM without thrashing
3) Inside each worker process, __iter__ runs. Every worker loads NGB into RAM, shuffles, etc
4) Pytorch waits for IteratorDataset to get 1024 tuples at a time
5) Hands batches of 1024 tuples to fast_collate
6) fast_collate builds the Pytorch tensors
7) pin_memory=True. Once tensors are created, they are copied to pinned RAM. 
8) Ready for GPU to read this now.

"""
import os
import bisect
import torch
import psutil
import math
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
 
class ChessChunkedDataset(IterableDataset):
    # usually 0.5, but we need half the RAM to copy and scrambling
    def __init__(self, file_paths, num_workers, ram_frac=0.25):
        super().__init__()
        self.file_paths = file_paths 

        # 18 bitbaords + result + move_idx + padding = 1216 bits = 152 bytes
        # this is divisible by 8 so block size should be fine
        self.dtype = np.dtype([
            ('bitboards', np.uint64, (18,)),
            ('result', np.float32),
            ('move_idx', np.int16),
            ('padding', np.int16)
        ])

        self.sample_bytes = self.dtype.itemsize # sample_byes = 152 - a single move in a game
        self.num_workers = num_workers
        # RAM BUDGETING
        total_ram_byes= psutil.virtual_memory().total
        total_ram_gb = total_ram_byes / (1024**3)
        ram_budget = total_ram_gb * ram_frac
    
        chunk_size_gb = ram_budget / self.num_workers 
        num_bytes = int(chunk_size_gb * 1024**3)

        # Number of board states in xGB of RAM
        self.samples_per_chunk = num_bytes // self.sample_bytes

        print(f"System RAM: {total_ram_gb:.1f}GB | Budget: {ram_budget:.1f}GB")
        print(f"Workers: {self.num_workers} | Chunk Size: {chunk_size_gb:.2f}GB per worker")

    """
    Every worker calls this independently
    """
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Pre allocate physical RAM buffer

        chunk_bytes = self.samples_per_chunk * self.sample_bytes
        ram_buffer = bytearray(chunk_bytes)

        for path in self.file_paths:
            file_size = os.path.getsize(path)
            total_samples = file_size // self.sample_bytes
            samples_per_worker = math.ceil(total_samples / num_workers)

            # chunk of data that the worker reads
            start_sample_worker_idx = worker_id * samples_per_worker 
            end_sample_worker_idx = min(start_sample_worker_idx + samples_per_worker, total_samples)

            if start_sample_worker_idx > end_sample_worker_idx:
                continue
            
            current_sample_idx = start_sample_worker_idx
            with open(path, "rb") as f:
                while current_sample_idx < end_sample_worker_idx:
                    chunk_end = min(current_sample_idx + self.samples_per_chunk, end_sample_worker_idx)
                    chunk_length_samples = chunk_end - current_sample_idx 
                    chunk_length_bytes = chunk_length_samples * self.sample_bytes
                    byte_offset = current_sample_idx * self.sample_bytes

                    # get the direct offset on RAM
                    f.seek(byte_offset)

                    # mem copy from OS cache to RAM
                    view = memoryview(ram_buffer)[:chunk_length_bytes]
                    bytes_read = f.readinto(view)

                    if bytes_read != chunk_length_bytes:
                        raise RuntimeError(f"Expected {chunk_length_bytes} bytes, got {bytes_read}")

                    chunk_arr = np.ndarray(
                        shape=(chunk_length_samples,),
                        dtype=self.dtype,
                        buffer=ram_buffer
                    )

                    np.random.shuffle(chunk_arr)

                    for sample in chunk_arr:
                        yield sample['bitboards'], sample['result'], sample['move_idx']
                    
                    current_sample_idx = chunk_end


            

"""
Turns tuples into memory blocks for GPU DMA.
By the time the data reaches here, Dataloader has list of 1024 tuples yielded by workers

Batch looks like [(bitboard_0, res_0, move_0), (bitboard_1, res_1, move_1)...]
"""
def fast_collate(batch):

    # change this to column format. [bitboard_0, bitboard_1, bitboard_2, ...], 
    # [results_0, results_1, results_2,...]
    bitboards, results, move_indices = zip(*batch)

    # used to be list of pointers. Fit into a 2D array of shape (1024, 18). Colascing
    bitboards_np = np.array(bitboards) #bitborads is a tuple of 1D arrays
    results_np = np.array(results, dtype=np.float32)
    move_idxs_np = np.array(move_indices, dtype=np.int64)

    # wrap them in pytorch tensor objects
    return (
        torch.from_numpy(bitboards_np),
        torch.from_numpy(results_np),
        torch.from_numpy(move_idxs_np)
    )

if __name__ == '__main__':
    binaries = ["chunk_0.bin", "chunk_1.bin", "chunk_2.bin", "chunk_3.bin", 
    "chunk_4.bin", "chunk_5.bin", "chunk_6.bin", "chunk_7.bin",
    "chunk_8.bin", "chunk_9.bin", "chunk_10.bin", "chunk_11.bin", 
    "chunk_12.bin", "chunk_13.bin", "chunk_14.bin", "chunk_15.bin"]
    num_workers = os.cpu_count()
    dataset = ChessChunkedDataset(binaries, num_workers=num_workers, ram_frac=0.25)

    dataloader = DataLoader(
        dataset, 
        batch_size=1024, #1024 
        num_workers=os.cpu_count(),
        collate_fn=fast_collate,
        pin_memory=True,
        prefetch_factor=2
    )

            



        
