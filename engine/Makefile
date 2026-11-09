CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

UTILS_SRC = ../utils/utils.cpp
MOVEGEN_SRC = movegen.cpp
PERFT_SRC = perft.cpp
MCTS_SRC = mcts.cpp
PLAY_SRC = play.cpp

# Object files
UTILS_OBJ = utils.o
MOVEGEN_OBJ = movegen.o
PERFT_OBJ = perft.o
MCTS_OBJ = mcts.o
PLAY_OBJ = play.o

TARGET = perft
PLAY_TARGET = play

all: $(TARGET) $(PLAY_TARGET)

$(UTILS_OBJ): $(UTILS_SRC) ../utils/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTILS_SRC) -o $(UTILS_OBJ)

$(MOVEGEN_OBJ): $(MOVEGEN_SRC) movegen.h ../utils/utils.h
	$(CXX) $(CXXFLAGS) -c $(MOVEGEN_SRC) -o $(MOVEGEN_OBJ)

$(PERFT_OBJ): $(PERFT_SRC) movegen.h
	$(CXX) $(CXXFLAGS) -c $(PERFT_SRC) -o $(PERFT_OBJ)

$(MCTS_OBJ): $(MCTS_SRC) mcts.h movegen.h
	$(CXX) $(CXXFLAGS) -c $(MCTS_SRC) -o $(MCTS_OBJ)

$(PLAY_OBJ): $(PLAY_SRC) mcts.h movegen.h
	$(CXX) $(CXXFLAGS) -c $(PLAY_SRC) -o $(PLAY_OBJ)

$(TARGET): $(UTILS_OBJ) $(MOVEGEN_OBJ) $(PERFT_OBJ)
	$(CXX) $(CXXFLAGS) $(UTILS_OBJ) $(MOVEGEN_OBJ) $(PERFT_OBJ) -o $(TARGET)

$(PLAY_TARGET): $(UTILS_OBJ) $(MOVEGEN_OBJ) $(MCTS_OBJ) $(PLAY_OBJ)
	$(CXX) $(CXXFLAGS) $(UTILS_OBJ) $(MOVEGEN_OBJ) $(MCTS_OBJ) $(PLAY_OBJ) -o $(PLAY_TARGET)

clean:
	rm -f *.o $(TARGET) $(PLAY_TARGET)

.PHONY: all clean
