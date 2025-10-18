#pragma once
#include "matrix_operations.h"
void sequential_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);
void openMP_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);
void openMP_transpose_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);