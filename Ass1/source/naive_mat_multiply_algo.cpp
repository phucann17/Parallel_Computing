#include "naive_mat_multiply_algo.h"
#include "matrix_operations.h"

void sequential_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            res[i][j] = tmp;
        } 
    }
}

void parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    #pragma omp parallel for num_threads(8)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            res[i][j] = tmp;
        } 
    }
}

