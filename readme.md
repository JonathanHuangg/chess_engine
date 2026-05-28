During the World Chess Championship, I was motivated to create a chess-engine - merging AI stuff with HPC fundamentals.

The goal of ChessBrain is to learn GM games while minimizing CPU and GPU starvation between getting the data and having a chess engine that can output moves. All of this was built for a Ryzen 7 2700x and GTX 1080 which actually is my current gaming rig as I wanted to work with something "tangible."

The phases are separated into 3 distinct phases: 1) An ETL pipeline to turn PGN to binary. 2) Dual-Head ResNet 3) Pytorch Dataloader + Training Loop 4) Inference with monte-carlo tree search to output chess moves.  

### 1) The ETL Pipeline
Almost all of the code can be found in /part1/pgn_processor, /utils/utils.cpp, /utils/utils.h. 

After manually downloading the pgn files, we assign workers based on the number of threads on the CPU. The script then goes through all of the files, simply looking at file sizes, and divides up the exact binary segments to read for each worker thread so each have a similar amount of work (a). 

The simple-ish part of the script is raw string manipulation - being able to parse PGNs for moves with edge cases like castling, checking, or general unpredicted spacing. 

To simulate the games, I used uint64_t bitboards as on a chessboard, it is 8x8 = 64 so for every type of piece, you can assert a bit if a piece exists on that square. In other words, given 6 different types of pieces (pawn, queen, knight, rook, king, bishop) as well as their opposite colored counterparts, with the addition of bitboards for special moves (em passant, castling rights), we are managing a total of 18 bitboards or 18 uint64_t numbers for a total of 144 bytes (b) -> much smaller with a traditional class where every square has an int so 64 squares * 4 bytes =  256 bytes.

On top of that, for a CNN or Resnet, the one-hot encoding from bitboards is ideal for training. Because they use dot products, if integers were assinged to pieces (1 = pawn, 2 = knight, etc), a network during learning would assume the pieces are n times mathamatically larger depenidng on the number. Instead, we use 0/1 to show presence whih allows netowrks to easily learn movement patterns. 

That was only the basics. To actually parse and turn the PGN string into bitboards, I had to write a state machine to disambuglate every piece move. For instance, if a knight jumps to C3, that is KC3. However, there are two knights, which knight jumps? With eyes, it's obvious. But textually, it's a little harder. I wrote attack masks for every piece and did an AND (&) over the current board state. That simulated which pieces can reach the state. In cases where both pieces can, ie: both knights can jump to C3, I implemented a rank and file mask to find the correct piece that moves. 

With all of that out of the way, the bitboards also had to be prepped for the Resnet This meant needing to flip the bitboard for every black move (a ResNet would only know from the perspective of a single player) with the attached game result. 

Packaging all the bitboards and metadata now, each worker thread has their own write buffer (c)where the data is pushed. 

NOTE: when I ran this, it was 50GB of data which means that if using the data I provide, you will need 50GB of free space. 


(a) With multithreading, something like work tracking across threads posed an interesting challenge: with 16 threads updating a shared global counter, the CPU cores would often invalidate each other's L1/L2 caches. The solution was to run an atomic feth_add with std::memroy_order_relaxed every 4096 games. 

(b) bitboards 0-11 are for the piece types for the player and opponent. bitboard 12 is the turn indicator (flood with 1s if white, 0 if black. This is because CNN uses 3x3 windows it needs to always be able to see who's turn it is). Bitboard 13-16 are castling rights (both colors, player kingside and queenside and opponent kingside and queenside). Bitboard 17 is the emphassant square. 

(c) - Referencing the bitboards, I had a struct called TrainingSample that had all the data I needed. I used a std::vector that held those and after 2048 MB of data, it does a bulk memory dump into the NVMe drive. 

The result of this implementation was a consistent 770 MB/s write speed. Noting that NVMe drives have a burst cache speed of around 2500 MB/s with the SLC cache, with 50GB of data, the focus was a sustained write which NVMe's can only do at around 900 MB/s. 

pip install requirements in part 1:
pip install -r chess\part1\requirements.txt

for gtx 1080:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126