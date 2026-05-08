import os
import torch 
from torch.utils.data import DataLoader 

from pytorch_Dataloader import ChessChunkedDataset, fast_collate 
from model import ChessBrainResNet, train_loop, export_to_onnx

def main():
    assert torch.cuda.is_avaiable(), "GPU does not exist"

    print("---getting the binaries that should already be generated---")
    binaries = [f"chunk_{i}.bin" for i in range(16)]

    # get the core count
    num_workers = os.cpu_count or 1
    print("number of workers: ", num_workers)

    # set up the dataloader
    dataset = ChessChunkedDataset(binaries, num_workers=num_workers, ram_frac=0.25)

    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        num_workers=num_workers,
        collate_fn=fast_collate,
        pin_memory=True,
        prefetch_factor=2
    )
    print("Dataloader is ready")

    print("--- setting up model architecture --- ")

    model = ChessBrainResNet(num_blocks=15, vocab_size=4672)
    print("Model initialized")

    epochs = 10
    value_weight = 0.01
    print(f"Running model with {epochs} epochs and value_weight of {value_weight}")
    train_loop(dataloader, model, epochs, value_weight)

    print("---Exporting ONNX---")  
    export_to_onnx(model, "chessbrain.onnx")
    print("Done! Ready for inference")

if __name__ == '__main__':
    main()