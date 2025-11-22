#pragma once 
#include <vector>
#include <omp.h>
typedef std::vector<std::vector<int>> matrix;
extern unsigned int n_global;
#define NUM_THREADS 20
#define NUM_TASK 8
#define NUM_FOR (NUM_THREADS - NUM_TASK)
matrix create_matrix(int n);
matrix transpose(const matrix& B);
void write_log(const char* filename, const matrix& m, unsigned int n);
void sequential_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n);
void sequential_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n);
void parallel_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n);
void parallel_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n);