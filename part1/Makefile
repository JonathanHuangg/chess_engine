CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread

ifeq ($(OS),Windows_NT)
    export PATH := C:/msys64/ucrt64/bin;$(PATH)
    CXXFLAGS += -static
endif

SRCS = pgn_processor.cpp stats.cpp ../utils/utils.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = pgn_processor

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
