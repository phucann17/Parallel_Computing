CXX = mpicxx

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

run: 
	mpirun -np 2 ./$(TARGET)
clean:
	rm -f $(TARGET) $(OBJECTS)