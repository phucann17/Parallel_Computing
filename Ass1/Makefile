CXX = g++

CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp

SRC_DIR = source

TARGET = main

SOURCES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp

OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

run:
	OMP_NUM_THREADS=4 ./${TARGET}

.PHONY: run clean all