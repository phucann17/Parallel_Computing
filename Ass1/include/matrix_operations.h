#pragma once 
#include <vector>

typedef std::vector<std::vector<int>> matrix;

matrix create_matrix(int n);
void write_log(const char* filename, const matrix& m, unsigned int n);
void sequential_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n);
void sequential_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n);