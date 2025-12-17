#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"
#include "strassen_mat_multiply_algo.h"

using namespace std;

bool verify_matrix_multiplication(const matrix& A, const matrix& B, int n) {
  for (int i = 0; i < n; ++i){
    if (A[i] != B[i]){
        return false;
    }
  }
  return true;
}

void print_block(const char* title, double time, bool ok) {
    printf("[%s]\n", title);
    printf("    Time: %.6f s\n", time);
    printf("    Check valid: %s\n", ok ? "OK" : "Not equal!");
    printf("------------------------------------------------\n\n");
}

struct BenchmarkResult {
    int n;
    double seq_naive;
    double seq_strassen;
    double mpi_naive;
    double mpi_strassen;
};

BenchmarkResult run_simulation(int n, int rank, int size) {

    if (rank == 0) {
        printf("==============================================\n");
        printf("Start MPI simulation with matrix size %d x %d\n", n, n);
        printf("==============================================\n\n");
    }

    BenchmarkResult R;
    R.n = n;
    n_global = n;

    matrix A = create_matrix(n);
    matrix B = create_matrix(n);
    matrix C_seq_naive = create_matrix(n);
    matrix C_seq_strassen = create_matrix(n);
    matrix C_mpi_strassen = create_matrix(n);

    // Rank 0 init
    if (rank == 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 100;
                B[i][j] = rand() % 100;
            }
    }

    // Broadcast A, B
    for (int i = 0; i < n; i++) {
        MPI_Bcast(A[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(B[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // === SEQ NAIVE ===
    if (rank == 0) {
        double s = MPI_Wtime();
        sequential_transpose_matrix_multiplication_naive(A, B, C_seq_naive, n);
        double e = MPI_Wtime();
        R.seq_naive = e - s;
        print_block("Sequential transpose naive", R.seq_naive, true);
    }

    // === SEQ STRASSEN ===
    if (rank == 0) {
        double s = MPI_Wtime();
        sequential_matrix_multiplication_strassen(A, B, C_seq_strassen, n);
        double e = MPI_Wtime();
        R.seq_strassen = e - s;

        print_block("Sequential Strassen",
                    R.seq_strassen,
                    verify_matrix_multiplication(C_seq_naive, C_seq_strassen, n));
    }

    MPI_Barrier(MPI_COMM_WORLD);


    // === MPI NAIVE ===
    double s1 = MPI_Wtime();
    mpi_parallel_matrix_multiplication_naive(A, B, C_seq_strassen, n, size, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double e1 = MPI_Wtime();

    if (rank == 0) {
        R.mpi_naive = e1 - s1;
        print_block("MPI Parallel Naive",
                    R.mpi_naive,
                    verify_matrix_multiplication(C_seq_naive, C_seq_strassen, n));
    }


    // === MPI STRASSEN ===
    double s2 = MPI_Wtime();
    mpi_parallel_matrix_multiplication_strassen(A, B, C_mpi_strassen, n, size, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double e2 = MPI_Wtime();

    if (rank == 0) {
        R.mpi_strassen = e2 - s2;
        print_block("MPI Parallel Strassen",
                    R.mpi_strassen,
                    verify_matrix_multiplication(C_seq_strassen, C_mpi_strassen, n));
    }

    if (rank == 0) {
        printf("Finish MPI simulation %d x %d\n", n, n);
        printf("==============================================\n\n");
    }

    return R;
}

void print_table(const vector<BenchmarkResult>& R) {
    printf("+--------+--------------+--------------+--------------+--------------+\n");
    printf("| Size   | Seq Naive    | Seq Strassen | MPI Naive    | MPI Strassen |\n");
    printf("+--------+--------------+--------------+--------------+--------------+\n");

    for (auto &r : R) {
        printf("| %-6d | %-12.6f | %-12.6f | %-12.6f | %-12.6f |\n",
               r.n, r.seq_naive, r.seq_strassen, r.mpi_naive, r.mpi_strassen);
    }
    printf("+--------+--------------+--------------+--------------+--------------+\n");
}

void run_all_benchmarks(int rank, int size) {
    vector<int> sizes = {100, 1000, 5000};
    vector<BenchmarkResult> results;

    for (int n : sizes) {
        BenchmarkResult r = run_simulation(n, rank, size);
        if (rank == 0) {
            results.push_back(r);
            printf("Sleeping 2 seconds before next simulation...\n");
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    if (rank == 0) {
        print_table(results);
    }
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    run_all_benchmarks(rank, size);

    MPI_Finalize();
    return 0;
}
