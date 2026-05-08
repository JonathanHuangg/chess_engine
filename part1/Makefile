CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread

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
