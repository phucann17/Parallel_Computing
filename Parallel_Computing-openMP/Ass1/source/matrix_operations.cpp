#include "matrix_operations.h"
#include <cstdio>
#include <cstdlib>
unsigned int n_global = 0;
matrix create_matrix(int n) {
    return matrix(n, std::vector<int> (n, 0));
}

void sequential_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
}

void sequential_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = a[i][j] - b[i][j];
        }
    }
}

void parallel_matrix_addition(const matrix& a, const matrix& b, matrix& res, int n) {
    // #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_FOR)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
}

void parallel_matrix_subtraction(const matrix& a, const matrix& b, matrix& res, int n) {
    // #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_FOR)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = a[i][j] - b[i][j];
        }
    }
}

void write_log(const char *filename, const matrix& m, unsigned int n){
    FILE* fd = fopen(filename, "w");

    if (!fd){
        perror("Can not open file to write");
        exit(1);
    }

    for (unsigned int i = 0; i < n; ++i){
        for (unsigned int j = 0; j < n; ++j){
            fprintf(fd, "%d ", m[i][j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
