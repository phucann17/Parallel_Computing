CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp
SRC_DIR = source
TARGET = main
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

NP ?= 2

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

run_nodes: $(TARGET)
	mpirun -np $$(($(NP) * 6)) --hostfile mpi-hosts.txt ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
