# ChessBrain: High-Performance GM Game Learning

During the World Chess Championship, I was motivated to create a chess-engine - merging AI stuff with HPC fundamentals.

The goal of ChessBrain is to learn GM games while minimizing CPU and GPU starvation between getting the data and having a chess engine that can output moves. All of this was built for a Ryzen 7 2700x and GTX 1080 which actually is my current gaming rig as I wanted to work with something "tangible."

The project is separated into 4 distinct phases: 
1. **An ETL pipeline** to turn PGN to binary. 
2. **PyTorch Dataloader** for high-throughput streaming.
3. **Dual-Head ResNet + Training Loop**. 
4. **Inference Engine** using monte-carlo tree search to output chess moves.  

---

### 1) The ETL Pipeline
Almost all of the code can be found in `/part1/pgn_processor`, `/utils/utils.cpp`, `/utils/utils.h`. 

After manually downloading the PGN files, we assign workers based on the number of threads on the CPU. The script then goes through all of the files, simply looking at file sizes, and divides up the exact binary segments to read for each worker thread so each have a similar amount of work [^1]. 

The simple-ish part of the script is raw string manipulation - being able to parse PGNs for moves with edge cases like castling, checking, or general unpredicted spacing. 

To simulate the games, I used `uint64_t` bitboards. As a chessboard is 8x8 = 64, for every type of piece, you can assert a bit if a piece exists on that square. In other words, given 6 different types of pieces (pawn, queen, knight, rook, king, bishop) as well as their opposite colored counterparts, with the addition of bitboards for special moves (en passant, castling rights), we are managing a total of **18 bitboards** or 18 `uint64_t` numbers for a total of 144 bytes [^2]. This is much smaller than a traditional class where every square has an `int` (64 squares * 4 bytes = 256 bytes).

On top of that, for a CNN or ResNet, the one-hot encoding from bitboards is ideal for training. Because they use dot products, if integers were assigned to pieces (1 = pawn, 2 = knight, etc.), a network during learning would assume the pieces are *n* times mathematically larger depending on the number. Instead, we use `0/1` to show presence, which allows networks to easily learn movement patterns. 

That was only the basics. To actually parse and turn the PGN string into bitboards, I had to write a state machine to disambiguate every piece move. For instance, if a knight jumps to c3, that is `Nc3` (not Kc3). However, if there are two knights, which knight jumps? With eyes, it's obvious. But textually, it's a little harder. I wrote attack masks for every piece and did a bitwise AND (`&`) over the current board state. That simulated which pieces can reach the state. In cases where both pieces can (e.g. both knights can jump to c3), I implemented a rank and file mask to find the correct piece that moves. 

With all of that out of the way, the bitboards also had to be prepped for the ResNet. This meant needing to flip the bitboard for every black move (a ResNet would only know from the perspective of a single player) with the attached game result. 

Packaging all the bitboards and metadata now, each worker thread has its own write buffer [^3] where the data is pushed. 

> **Note:** When I ran this, it generated 50GB of data, which means that if using the pipeline, you will need 50GB of free disk space. 

---

### 3) Dual-Head ResNet
The biggest thing I thought about here was "sacrifice." I had built this for the Ryzen 7 2700x and GTX 1080 because I wanted to. Trying to match AlphaZero is basically impossible. 

Regardless, the idea behind the dual-head stems from AlphaZero. In their case, they used a **value head** and a **policy head**. The value head is used to calculate who's winning and losing, and the policy head is used to output good/bad moves (more on this later).

Given `N x 18 x 8 x 8` tensor blocks to be fed into the network, I originally planned to stack 15 residual blocks in this order: `Conv2d(3x3) -> BatchNorm -> ReLU -> Conv2D(3x3) -> BatchNorm -> Skip -> ReLU`. For reference, AlphaZero used 40 blocks. 

After going through the 15 residual blocks, the understanding of the game is generally already captured in the `256 x 8 x 8` tensor, where 256 is the number of channels. 

* **The Value Head:** I want to get a global view of who is winning. If I flattened a `256 x 8 x 8` tensor and fed it into a dense layer where every input matches the output, the weight matrix would be massive - and for many games, near impossible to be done quickly. Thus, I ran a `1x1` convolution (the goal is a local view) to squash 256 channels down to 1 channel. Now each square has a single number representing its value contribution. 
* **The Policy Head:** The 256 channels are squashed to 2 channels, which is meant to represent pieces moving "away from" and "to" the square. 

Both the value and policy head must now be fed into a linear layer. Prior to this, I used convolutions which, by construction, yield local understanding. By passing it into a linear layer, the network has to look at all 64 squares simultaneously. More specifically, the value head looks at the threat level of all the squares and outputs a single scalar value -> *"white is winning by +0.67"*. For the policy head, I had the `(2 x 8 x 8)` tensor fed into a linear layer of `[4672, 128]` to output the best move out of 4,672 potential moves. 

A big question was the vocab size. Normally, we know that we have 64 starting squares and then 64 destination squares, so the vocabulary is 4096. However, AlphaZero understands this differently. Instead of looking at the board as *Square A to Square B*, this is encoded as **“Starting Square + Move Type.”** For each of the 64 squares, there are 73 theoretical move types:
* **Normal moves** (up, down, left, right, diagonals). 8 directions x 7 distances = 56 moves. 
* **Knight moves** (8 moves max from a single square).
* **Underpromotions**. A pawn can move straight forward, capture left, capture right (3 moves). It can turn into a knight, bishop, or rook. (Queen is not included because a queen promotion is already considered a normal move). 3 directions x 3 piece types = 9 moves.
* **Total:** 56 + 8 + 9 = 73 moves per square (73 * 64 = 4672 total). 

#### Implementation-wise
We have 2 classes: `ResidualBlock` and `ChessBrainResNet`. `ResidualBlock` acts as the basic building block and `ChessBrainResNet` is the overarching model. 

For `ResidualBlock`, we have a convolution, batch normalization, ReLU, convolution, and batch normalization. In every forward step, we add a convolution, batch norm, and add a ReLU. We then do it again with a convolution, batch norm, but now we add the residual connection, and then a final ReLU. 

For `ChessBrainResNet`, in `__init__()`, we use the custom `GPUUnpacker` and create a private variable called `initial_conv` which simply expands the 18 bitboards to 256 channels with a batch norm and ReLU. We then create the 15 `ResidualBlock` model called `res_block` along with the policy and value heads. 

* **Policy Head Forward Pass:** As stated above, the policy head takes the 256 channels and converts it to 2. After doing `BatchNorm2d` with a `ReLU`, `nn.Flatten()` unrolls the tensor into a `128` (2*8*8) 1D tensor. Following with `nn.Linear()`, the function creates a 2D weight matrix of shape `[4672, 128]` with a bias vector, and the GPU computes a cuBLAS GEMM. Further understanding this 2D tensor: `[4672, 128]` has 4,672 distinct rows. Each row represents a potential chess move. During the matrix multiplication, you take the row and perform a dot product with the 128 board features to see how strongly the move is supported. We leave this as logits for when we calculate the cross-entropy loss. 
* **Value Head Forward Pass:** Our goal is to calculate who is winning. Given a 256-channel `8x8` tensor of a specific batch size, we run a `1x1` convolution. This takes the 256 channels and squashes it to 1 value for each square. After calling `nn.Flatten()`, we unroll it into a 1D vector of 64 floats. We then pass it to `nn.Linear(64, 256)` which, under the hood, creates a weight matrix of `[256, 64]`. After doing the matrix multiplication, we essentially force the square to interact with every other square (the global view). The second `Linear(256, 1)` creates a weight matrix of `[1, 256]` and does a dot product to merge that global context into a single score. The tensor shape is finally `(batch, 1)`. We finally run `tanh()` to bound the winning advantage between `[-1, 1]`. 

---

#### Footnotes
[^1]: **MESI Protocol Optimization:** With multithreading, something like work tracking across threads posed an interesting challenge. With 16 threads updating a shared global counter, the CPU cores would often invalidate each other's L1/L2 caches (cache line bouncing). The solution was to run an atomic `fetch_add` with `std::memory_order_relaxed` only once every 4,096 games. 

[^2]: **Bitboard Layout:** Bitboards 0-11 are for the piece types for the player and opponent. Bitboard 12 is the turn indicator (flooded with 1s if white, 0s if black. Because a CNN uses 3x3 sliding windows, it needs to always be able to see whose turn it is locally). Bitboards 13-16 are castling rights (player kingside/queenside and opponent kingside/queenside, also flooded). Bitboard 17 is the en passant square. 

[^3]: **Bulk Binary Writes:** Referencing the bitboards, I had a `struct` called `TrainingSample` that had all the data I needed. I used a `std::vector` that accumulated 131,072 items (~19 MB of data) per chunk before doing a massive bulk memory dump into the NVMe drive. The result of this implementation was a consistent **770 MB/s** write speed. Noting that NVMe drives have a burst cache speed of around 2500 MB/s with the SLC cache, the focus for a 50GB dataset was the sustained TLC NAND write limit, which NVMe's can only do at around 900 MB/s. 