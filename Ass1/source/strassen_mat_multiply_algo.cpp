#include "strassen_mat_multiply_algo.h"
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"

void sequential_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    if (n <= 16) {
        sequential_matrix_multiplication_naive(A, B, res, n);
        return;
    }

    int halfSize = n / 2;

    matrix A11 = create_matrix(halfSize);
    matrix A12 = create_matrix(halfSize);
    matrix A21 = create_matrix(halfSize);
    matrix A22 = create_matrix(halfSize);
    matrix B11 = create_matrix(halfSize);
    matrix B12 = create_matrix(halfSize);
    matrix B21 = create_matrix(halfSize);
    matrix B22 = create_matrix(halfSize);

    for (int i = 0; i < halfSize; ++i) {
        for (int j = 0; j < halfSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j+halfSize];
            A21[i][j] = A[i+halfSize][j];
            A22[i][j] = A[i+halfSize][j+halfSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j+halfSize];
            B21[i][j] = B[i+halfSize][j];
            B22[i][j] = B[i+halfSize][j+halfSize];
        }
    }

    // cout << "Check here man !!!" << endl;

    matrix M1 = create_matrix(halfSize);
    matrix M2 = create_matrix(halfSize);
    matrix M3 = create_matrix(halfSize);
    matrix M4 = create_matrix(halfSize);
    matrix M5 = create_matrix(halfSize);
    matrix M6 = create_matrix(halfSize);
    matrix M7 = create_matrix(halfSize);

    matrix tmp1 = create_matrix(halfSize);
    matrix tmp2 = create_matrix(halfSize);

    // calculate M1 = (A11 + A22)(B11 + B22)
    sequential_matrix_addition(A11, A22, tmp1, halfSize);
    sequential_matrix_addition(B11, B22, tmp2, halfSize);
    sequential_matrix_multiplication_strassen(tmp1, tmp2, M1, halfSize);

    // calculate M2 = (A21 + A22)B11
    sequential_matrix_addition(A21, A22, tmp1, halfSize);
    sequential_matrix_multiplication_strassen(tmp1, B11, M2, halfSize);

    // calculate M3 = A11(B12 - B22)
    sequential_matrix_subtraction(B12, B22, tmp1, halfSize);
    sequential_matrix_multiplication_strassen(A11, tmp1, M3, halfSize);

    // calculate M4 = A22(B21 - B11)
    sequential_matrix_subtraction(B21, B11, tmp1, halfSize);
    sequential_matrix_multiplication_strassen(A22, tmp1, M4, halfSize);

    // calculate M5 = (A11 + A12)B22
    sequential_matrix_addition(A11, A12, tmp1, halfSize);
    sequential_matrix_multiplication_strassen(tmp1, B22, M5, halfSize);

    // calculate M6 = (A21 - A11)(B11 + B12)
    sequential_matrix_subtraction(A21, A11, tmp1, halfSize);
    sequential_matrix_addition(B11, B12, tmp2, halfSize);
    sequential_matrix_multiplication_strassen(tmp1, tmp2, M6, halfSize);

    // calculate M7 = (A12 - A22)(B21 + B22)
    sequential_matrix_subtraction(A12, A22, tmp1, halfSize);
    sequential_matrix_addition(B21, B22, tmp2, halfSize);
    sequential_matrix_multiplication_strassen(tmp1, tmp2, M7, halfSize);

    // C12 = M3 + M5
    // C21 = M2 + M4
    matrix C11 = create_matrix(halfSize);
    matrix C12 = create_matrix(halfSize);
    matrix C21 = create_matrix(halfSize);
    matrix C22 = create_matrix(halfSize);
    
    sequential_matrix_addition(M3, M5, C12, halfSize);
    sequential_matrix_addition(M2, M4, C21, halfSize);

    // C11 = M1 + M4 - M5 + M7
    sequential_matrix_addition(M1, M4, tmp1, halfSize);
    sequential_matrix_subtraction(tmp1, M5, tmp2, halfSize);
    sequential_matrix_addition(tmp2, M7, C11, halfSize);

    // C22 = M1 - M2 + M3 + M6
    sequential_matrix_subtraction(M1, M2, tmp1, halfSize);
    sequential_matrix_addition(tmp1, M3, tmp2, halfSize);
    sequential_matrix_addition(tmp2, M6, C22, halfSize);

    // cout << "Where is bug ?? " << endl;
    for (int i = 0; i < halfSize; ++i) {
        for (int j = 0; j < halfSize; ++j) {
            res[i][j] = C11[i][j];

            res[i][j + halfSize] = C12[i][j];
    
            res[i + halfSize][j] = C21[i][j];

            res[i + halfSize][j + halfSize] = C22[i][j];
        }
    }
}