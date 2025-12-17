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

## 2. Authors

*   **[Nguyen Quang Minh]** - Team Lead, Core Architect (Sequential Implementations & Testing Framework)
*   **[Nguyen Phuc An]** - OpenMP Specialist (Shared Memory Parallelization & Performance Analysis)
*   **[Dinh Xuan Quyet]** - MPI Specialist (Distributed Memory Parallelization & Scalability Study)

```

## 3. Project Structure & Usage
The repository is organized by computational paradigms. Below is the file structure:

code
Text
PARALLEL_COMPUTING
├── Parallel_Computing-hybrid       # Hybrid Model (MPI + OpenMP)
├── Parallel_Computing-mpi_new      # Distributed Memory implementation (MPI)
├── Parallel_Computing-openMP       # Shared Memory implementation (OpenMP)
├── Parallel_Computing-openMP_GPU   # OpenMP with GPU offloading
├── draw.ipynb                      # Performance visualization template
└── README.md                       # Main documentation

How to Run

Specific Implementations: To test or benchmark a specific parallel version, simply navigate to the corresponding directory (e.g., cd Parallel_Computing-openMP) and follow the instructions provided in the local README.md file inside that folder.

Visualization: The draw.ipynb file is a Jupyter Notebook template. You can use it to generate comparison charts and plots to visualize the performance differences between the algorithms.