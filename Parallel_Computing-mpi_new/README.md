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
make all
```

This command will create an executable file named `main` in the root directory.

To remove all compiled object files and the executable, run:
```bash
make clean
```

### Step 3: Execute the Program

Run on a Single Machine (MPI Local Execution)

Use the run target.

You can configure how many MPI processes to launch using the NP variable.

Default (NP = 2):
```bash
make run
```

Run with custom number of processes:
```bash
make run NP=4
```

*******Memory Usage Warning

If you run the program with matrix size ≥ 10,000, please limit execution to a maximum of 4 MPI processes.

Using more than 4 processes can easily exhaust system memory (including swap), which may cause the program to crash or the OS to terminate the process.

With matrix size 10,000 × 10,000, running with 4 processes already requires ~12 GB of RAM due to data replication and temporary buffers used by the algorithm. This notice is also used for Runing on Multiple Machines.

This will execute:
```bash
mpirun -np 4 ./main
```

Run on Multiple Machines (MPI Cluster Execution)

If you have multiple nodes listed inside mpi-hosts.txt, use the run_nodes target.
```bash
make run_nodes
```

The number of processes is calculated automatically as:
```bash
NP * 5
```

In this assignment, we will use 6 node:
```bash
MPI-node7
MPI-node9
MPI-node15
MPI-node14
MPI-node3
```

(Assuming your cluster has 6 nodes, each launching NP processes.)

For example:
```bash
make run_nodes NP=3
```

Will execute:
```bash
mpirun -np 15 --hostfile mpi-hosts.txt ./main
```

Rewritten Version

If you update or modify the source code, you must rebuild the project and then redeploy it to all nodes.

To redeploy, simply run:
```bash
./send_node.sh
```

Before running the script for the first time, make sure it is executable:
```bash
chmod +x send_node.sh
```

## 3. Project Structure

The source code is organized into a modular structure for clarity and maintainability.

```
Ass1
├── include/                  # Header files (.h) - Declarations only
│   ├── matrix_operations.h
│   ├── naive_mat_multiply_algo.h
│   └── strassen_mat_multiply_algo.h
├── source/                      # Source files (.cpp) - Function implementations
│   ├── matrix_operations.cpp
│   ├── naive_mat_multiply_algo.cpp
│   └── strassen_mat_multiply_algo.cpp
├── main.cpp                  # Main entry point, orchestration, and timing
├── Makefile                  # Build automation script
├── send_node.sh                 # Main entry point, orchestration, and timing
├─  mpi-hosts.txt             #hosts list
README.md                 # This file
```

---

## 4. Authors

*   **[Your Name Here]** - Team Lead, Core Architect (Sequential Implementations & Testing Framework)
*   **[Teammate 2 Name]** - OpenMP Specialist (Shared Memory Parallelization & Performance Analysis)
*   **[Teammate 3 Name]** - MPI Specialist (Distributed Memory Parallelization & Scalability Study)

```