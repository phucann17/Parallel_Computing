#pragma once
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"

void sequential_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n);

// You can define the parallel version here. 
//  DEFINE //
void openmp_parallel_matrix_multiplication_strassen_operation(const matrix& A, const matrix& B, matrix& res, unsigned int n, bool useGPU);
void openmp_parallel_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n, bool useGPU);