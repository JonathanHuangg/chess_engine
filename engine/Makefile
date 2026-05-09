CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

UTILS_SRC = ../utils/utils.cpp
MOVEGEN_SRC = movegen.cpp
PERFT_SRC = perft.cpp

# Object files
UTILS_OBJ = utils.o
MOVEGEN_OBJ = movegen.o
PERFT_OBJ = perft.o

TARGET = perft

all: $(TARGET)

$(UTILS_OBJ): $(UTILS_SRC) ../utils/utils.h
	$(CXX) $(CXXFLAGS) -c $(UTILS_SRC) -o $(UTILS_OBJ)

$(MOVEGEN_OBJ): $(MOVEGEN_SRC) movegen.h ../utils/utils.h
	$(CXX) $(CXXFLAGS) -c $(MOVEGEN_SRC) -o $(MOVEGEN_OBJ)

$(PERFT_OBJ): $(PERFT_SRC) movegen.h
	$(CXX) $(CXXFLAGS) -c $(PERFT_SRC) -o $(PERFT_OBJ)

$(TARGET): $(UTILS_OBJ) $(MOVEGEN_OBJ) $(PERFT_OBJ)
	$(CXX) $(CXXFLAGS) $(UTILS_OBJ) $(MOVEGEN_OBJ) $(PERFT_OBJ) -o $(TARGET)

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean
