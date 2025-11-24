CXX = g++

CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp -foffload=nvptx-none -fcf-protection=none -no-pie -fno-stack-protector

SRC_DIR = source
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp

TARGET = test
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) source/*.o

run:
	./$(TARGET)

.PHONY: all clean run
