CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O3 -Wall -Iinclude -fopenmp
NVCCFLAGS = -std=c++17 -O3 -Iinclude -Xcompiler -fopenmp -arch=sm_86

SRC_DIR = source
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) main.cpp

# Targets
TARGET_CPU = main
TARGET_GPU = main_gpu

# Object files
OBJECTS_CPU = $(SOURCES:.cpp=.o)
OBJECTS_GPU = $(SOURCES:.cpp=.gpu.o)

# Default: build both CPU and GPU
all: $(TARGET_CPU) $(TARGET_GPU)

# CPU build
$(TARGET_CPU): $(OBJECTS_CPU)
	$(CXX) $(CXXFLAGS) -o $(TARGET_CPU) $(OBJECTS_CPU)

# GPU build
$(TARGET_GPU): $(OBJECTS_GPU)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_GPU) $(OBJECTS_GPU)

# Compile CPU objects
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile GPU objects
%.gpu.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET_CPU) $(TARGET_GPU) $(OBJECTS_CPU) $(OBJECTS_GPU) source/*.o

run_cpu:
	./$(TARGET_CPU)

run_gpu:
	./$(TARGET_GPU)

.PHONY: all clean run_cpu run_gpu
