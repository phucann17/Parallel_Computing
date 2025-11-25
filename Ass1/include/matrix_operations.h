#pragma once 
#include <vector>
using std::vector;
extern unsigned int n_global;
typedef std::vector<std::vector<int>> matrix;
matrix create_matrix(int n);
void sequential_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n);
void sequential_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n);