#pragma once
#include "matrix_operations.h"
#include <mpi.h>

void sequential_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);
void parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);
void mpi_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n);