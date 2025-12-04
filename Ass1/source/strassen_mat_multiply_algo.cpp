#include "strassen_mat_multiply_algo.h"
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"

// Note: Strassen's algorithm is designed for square matrices of size n x n where n is a power of 2.
// So when dealing with matrices with its size not a power of 2, we need to pad the matrices to the next power of 2. 
// Sorry so much because I did not remember this knowledge when implementing the algorithm before :((( 
// But don't worry, when you read these comments, I have already handled this case :))) 
// So below is the perfect implementation of sequential Strassen's algorithm :))))

// Note: The way to pad the matrix to the next power of 2 is so inefficient because it takes O(n^2) space complexity
// So for more efficient method, I decide to pad the original matrices to the next nearest even number only if n is odd 

bool check_power_of_two(unsigned int n) {
    return (n & (n - 1)) == 0;
}

unsigned int next_power_of_two(unsigned int n) {
    // 5 -> 00000101
    // 5-- -> 4 = 00000100
    // 4 | 4 >> 1 -> 00000100 | 00000010 = 00000110 = 6
    // 6 | 6 >> 2 -> 00000110 | 00000001 = 00000111 = 7
    // 7 | 7 >> 4 -> 00000111 | 00000000 = 00000111 = 7
    // 7 | 7 >> 8 -> 00000111 | 00000000 = 00000111 = 7
    // 7 | 7 >> 16 -> 00000111 | 00000000 = 00000111 = 7
    // 7++ = 8 -> 00001000

    // Note: This function works only for 32-bit unsigned integers, with 64-bit integers, we need to extend the shifts to 32.
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void sequential_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    if (n <= 64) {
        sequential_transpose_matrix_multiplication_naive(A, B, res, n);
        return;
    }

    if ((n & 1)) {
        unsigned int padded_n = n + 1;
        matrix padded_A = create_matrix(padded_n);
        matrix padded_B = create_matrix(padded_n);
        matrix padded_res = create_matrix(padded_n);

        // Copy original matrices into padded matrices
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                padded_A[i][j] = A[i][j];
                padded_B[i][j] = B[i][j];
            }
        }

        // Perform Strassen's algorithm on padded matrices
        sequential_matrix_multiplication_strassen(padded_A, padded_B, padded_res, padded_n);

        // Copy result back to original result matrix
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                res[i][j] = padded_res[i][j];
            }
        }
        
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

    // #pragma omp parallel for collapse(2) schedule(static) num_threads(8)
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

    // #pragma omp parallel for collapse(2) schedule(static) num_threads(8)
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

void openmp_parallel_matrix_multiplication_strassen_operation(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    if (n <= ((n_global/4) + 1)) {
        // openMP_parallel_matrix_multiplication_naive(A, B, res, n);
        openMP_transpose_parallel_matrix_multiplication_naive(A, B, res, n);
        return;
    }

    if ((n & 1)) {
        unsigned int padded_n = n + 1;
        matrix padded_A = create_matrix(padded_n);
        matrix padded_B = create_matrix(padded_n);
        matrix padded_res = create_matrix(padded_n);

        // Copy original matrices into padded matrices
        // #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_THREADS)
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                padded_A[i][j] = A[i][j];
                padded_B[i][j] = B[i][j];
            }
        }

        // Perform Strassen's algorithm on padded matrices
        openmp_parallel_matrix_multiplication_strassen_operation(padded_A, padded_B, padded_res, padded_n);

        // Copy result back to original result matrix
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                res[i][j] = padded_res[i][j];
            }
        }
        
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

    // #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_THREADS)
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

    // matrix tmp1 = create_matrix(halfSize);
    // matrix tmp2 = create_matrix(halfSize);
    // C12 = M3 + M5
    // C21 = M2 + M4

    // --- Parallel region for M1..M7 ---
    if (n >= ((n_global/2) + 1)) { 
        #pragma omp task shared(M1)
        {
            matrix tmp1 = create_matrix(halfSize);
            matrix tmp2 = create_matrix(halfSize);
            parallel_matrix_addition(A11, A22, tmp1, halfSize);
            parallel_matrix_addition(B11, B22, tmp2, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M1, halfSize);
        }
        #pragma omp task shared(M2)
        {
            matrix tmp1 = create_matrix(halfSize);
            parallel_matrix_addition(A21, A22, tmp1, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(tmp1, B11, M2, halfSize);
        }
        #pragma omp task shared(M3)
        {
            matrix tmp1 = create_matrix(halfSize);
            parallel_matrix_subtraction(B12, B22, tmp1, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(A11, tmp1, M3, halfSize);
        }
        #pragma omp task shared(M4)
        {
            matrix tmp1 = create_matrix(halfSize);
            parallel_matrix_subtraction(B21, B11, tmp1, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(A22, tmp1, M4, halfSize);
        }
        #pragma omp task shared(M5)
        {
            matrix tmp1 = create_matrix(halfSize);
            parallel_matrix_addition(A11, A12, tmp1, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(tmp1, B22, M5, halfSize);
        }
        #pragma omp task shared(M6)
        {
            matrix tmp1 = create_matrix(halfSize);
            matrix tmp2 = create_matrix(halfSize);
            parallel_matrix_subtraction(A21, A11, tmp1, halfSize);
            parallel_matrix_addition(B11, B12, tmp2, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M6, halfSize);
        }
        #pragma omp task shared(M7)
        {
            matrix tmp1 = create_matrix(halfSize);
            matrix tmp2 = create_matrix(halfSize);
            parallel_matrix_subtraction(A12, A22, tmp1, halfSize);
            parallel_matrix_addition(B21, B22, tmp2, halfSize);
            openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M7, halfSize);
        }
        #pragma omp taskwait
    } else {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        parallel_matrix_addition(A11, A22, tmp1, halfSize);
        parallel_matrix_addition(B11, B22, tmp2, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M1, halfSize);

        parallel_matrix_addition(A21, A22, tmp1, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(tmp1, B11, M2, halfSize);

        parallel_matrix_subtraction(B12, B22, tmp1, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(A11, tmp1, M3, halfSize);

        parallel_matrix_subtraction(B21, B11, tmp1, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(A22, tmp1, M4, halfSize);

        parallel_matrix_addition(A11, A12, tmp1, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(tmp1, B22, M5, halfSize);

        parallel_matrix_subtraction(A21, A11, tmp1, halfSize);
        parallel_matrix_addition(B11, B12, tmp2, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M6, halfSize);

        parallel_matrix_subtraction(A12, A22, tmp1, halfSize);
        parallel_matrix_addition(B21, B22, tmp2, halfSize);
        openmp_parallel_matrix_multiplication_strassen_operation(tmp1, tmp2, M7, halfSize);
    }
                
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < halfSize; ++i) {
            for (int j = 0; j < halfSize; ++j) {
                res[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];          // C11
                res[i][j + halfSize] = M3[i][j] + M5[i][j];                     // C12
                res[i + halfSize][j] = M2[i][j] + M4[i][j];                     // C21
                res[i + halfSize][j + halfSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]; // C22
            }
        }
    
    
}

void openmp_parallel_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n){
    // omp_set_num_threads(8);
    omp_set_nested(1);
    #pragma omp parallel
    {
    #pragma omp single
     {
        openmp_parallel_matrix_multiplication_strassen_operation(A, B, res, n);
     }
    }
}