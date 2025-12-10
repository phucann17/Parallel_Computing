#include <iostream>
#include <omp.h>
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"
#include "strassen_mat_multiply_algo.h"
#include <thread>
using namespace std;

// void free_matrix(matrix m, unsigned int n){
//     for (int i = 0; i < n; ++i){
//         free(m[i]);
//     }
//     free(m);
// }
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
    int n; // size
    double seq_naive;
    double omp_cpu_naive;
    double omp_gpu_naive;
    double seq_strassen;
    double omp_cpu_strassen;
    double omp_gpu_strassen;
};

BenchmarkResult run_simulation(int n) {
    printf("==============================================\n");
    printf("Start simulation with matrix size %d x %d\n", n, n);
    printf("==============================================\n\n");

    BenchmarkResult R;
    R.n = n;
    n_global = n;

    matrix A = create_matrix(n);
    matrix B = create_matrix(n);
    matrix res_naive_seq = create_matrix(n);
    matrix res_naive_parallel = create_matrix(n);
    matrix res_strassen_seq = create_matrix(n);

    // init random
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }

    // [1] Sequential transpose naive
    double start = omp_get_wtime();
    sequential_transpose_matrix_multiplication_naive(A, B, res_naive_seq, n);
    double end = omp_get_wtime();
    R.seq_naive = end - start;
    print_block("Sequential transpose naive",
                R.seq_naive,
                true);   // always valid vs itself

    // [2] Sequential Strassen
    start = omp_get_wtime();
    sequential_matrix_multiplication_strassen(A, B, res_strassen_seq, n);
    end = omp_get_wtime();
    R.seq_strassen = end - start;
    print_block("Sequential Strassen",
                R.seq_strassen,
                verify_matrix_multiplication(res_naive_seq, res_strassen_seq, n));

    // [3] OpenMP CPU naive
    start = omp_get_wtime();
    openMP_transpose_parallel_matrix_multiplication_naive(A, B, res_naive_parallel, n);
    end = omp_get_wtime();
    R.omp_cpu_naive = end - start;
    print_block("OpenMP CPU naive",
                R.omp_cpu_naive,
                verify_matrix_multiplication(res_naive_seq, res_naive_parallel, n));

    // [4] OpenMP GPU naive
    start = omp_get_wtime();
    openMP_gpu_matrix_multiply(A, B, res_naive_parallel);
    end = omp_get_wtime();
    R.omp_gpu_naive = end - start;
    print_block("OpenMP GPU naive",
                R.omp_gpu_naive,
                verify_matrix_multiplication(res_naive_seq, res_naive_parallel, n));

    // [5] OpenMP CPU Strassen
    start = omp_get_wtime();
    openmp_parallel_matrix_multiplication_strassen(A, B, res_naive_parallel, n, 0);
    end = omp_get_wtime();
    R.omp_cpu_strassen = end - start;
    print_block("OpenMP CPU Strassen",
                R.omp_cpu_strassen,
                verify_matrix_multiplication(res_strassen_seq, res_naive_parallel, n));

    // [6] OpenMP GPU Strassen
    start = omp_get_wtime();
    openmp_parallel_matrix_multiplication_strassen(A, B, res_naive_parallel, n, 1);
    end = omp_get_wtime();
    R.omp_gpu_strassen = end - start;
    print_block("OpenMP GPU Strassen",
                R.omp_gpu_strassen,
                verify_matrix_multiplication(res_strassen_seq, res_naive_parallel, n));

    printf("Finish simulation %d x %d\n", n, n);
    printf("==============================================\n\n");

    return R;
}

void print_table(const vector<BenchmarkResult>& R) {
    printf("+--------+--------------+--------------+--------------+--------------+--------------+--------------+\n");
    printf("| Size   | Seq Naive    | CPU Naive    | GPU Naive    | Seq Strassen | CPU Strassen | GPU Strassen |\n");
    printf("+--------+--------------+--------------+--------------+--------------+--------------+--------------+\n");

    for (auto &r : R) {
        printf("| %-6d | %-12.6f | %-12.6f | %-12.6f | %-12.6f | %-12.6f | %-12.6f |\n",
            r.n,
            r.seq_naive,
            r.omp_cpu_naive,
            r.omp_gpu_naive,
            r.seq_strassen,
            r.omp_cpu_strassen,
            r.omp_gpu_strassen
        );
    }

    printf("+--------+--------------+--------------+--------------+--------------+--------------+--------------+\n");
}

void run_all_benchmarks() {
    vector<int> sizes = {100, 1000, 3000};
    vector<BenchmarkResult> results;

    for (int n : sizes) {
        results.push_back(run_simulation(n));

        printf("Sleeping 3 seconds before next simulation...\n");
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    print_table(results);
}

int main() {
    run_all_benchmarks();
    return 0;
}