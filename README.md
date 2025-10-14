# Parallel_Computing Assignment CO3067

This project is an implementation and performance analysis of matrix multiplication algorithms (Naive and Strassen) for the Parallel Computing course (CO3067). It explores different computational paradigms:

*   **Sequential**: A single-threaded implementation serving as a performance baseline.
*   **Shared Memory Parallelism**: Accelerated using OpenMP to leverage multi-core processors on a single machine.
*   **Distributed Memory Parallelism**: Designed for multi-node execution using MPI (*to be implemented*).
*   **Hybrid Model**: A combination of MPI and OpenMP for optimized performance on modern multi-core clusters (*to be implemented*).

---

## 1. Prerequisites

Before you begin, ensure your system has the following tools installed.

*   **C++ Compiler**: `g++` with support for C++17 or later.
*   **Make**: The `make` utility for automating the build process.
*   **OpenMP Library**: Typically included with modern `g++` compilers.
*   **(Optional) MPI Library**: An MPI implementation like [OpenMPI](https://www.open-mpi.org/) is required for the distributed versions.

On **Debian/Ubuntu-based** systems, you can install all dependencies with this command:
```bash
sudo apt-get update
sudo apt-get install build-essential make openmpi-bin libopenmpi-dev
```
For other operating systems, figure it out yourself. Google exists for a reason.

---

## 2. Build and Run

Follow these steps to compile and execute the program.

### Step 1: Clone the Repository

```bash
git clone
cd Ass1
```

### Step 2: Build the Executable

Use the provided `Makefile` to compile the entire project.

```bash
make
```

This command will create an executable file named `main` in the root directory.

To remove all compiled object files and the executable, run:
```bash
make clean
```

### Step 3: Execute the Program

The application takes only one command.

**Syntax:**
```bash
./main
```

## 3. Project Structure

The source code is organized into a modular structure for clarity and maintainability.

```
Ass1
.
├── include/                  # Header files (.h) - Declarations only
│   ├── matrix_operations.h
│   ├── naive_mat_multiply_algo.h
│   └── strassen_mat_multiply_algo.h
├── src/                      # Source files (.cpp) - Function implementations
│   ├── matrix_operations.cpp
│   ├── naive_mat_multiply_algo.cpp
│   └── strassen_mat_multiply_algo.cpp
├── main.cpp                  # Main entry point, orchestration, and timing
├── Makefile                  # Build automation script
└── README.md                 # This file
```

---

## 4. Authors

*   **[Your Name Here]** - Team Lead, Core Architect (Sequential Implementations & Testing Framework)
*   **[Teammate 2 Name]** - OpenMP Specialist (Shared Memory Parallelization & Performance Analysis)
*   **[Teammate 3 Name]** - MPI Specialist (Distributed Memory Parallelization & Scalability Study)
```