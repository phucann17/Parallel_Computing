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
    if (n <= 16) {
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
        #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(5)
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

void mpi_parallel_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n, int size, int rank) {
    if (n <= (n_global / 1.5)) {
        openMP_transpose_parallel_matrix_multiplication_naive(A, B, res, n, 5);
        return;
    }
    int halfSize = n / 2;

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
        mpi_parallel_matrix_multiplication_strassen(padded_A, padded_B, padded_res, padded_n, size, rank);

        // Copy result back to original result matrix
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                res[i][j] = padded_res[i][j];
            }
        }
        
        return;
    }
    // Divide matrices for Strassen's algorithm apply for all processes
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

    // Create matrices M contain results of 7 multiplications
    matrix M1 = create_matrix(halfSize);
    matrix M2 = create_matrix(halfSize);
    matrix M3 = create_matrix(halfSize);
    matrix M4 = create_matrix(halfSize);
    matrix M5 = create_matrix(halfSize);
    matrix M6 = create_matrix(halfSize);
    matrix M7 = create_matrix(halfSize);

    // Each process earn some multiplications to compute
    // calculate M1 = (A11 + A22)(B11 + B22)
    if (0 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        sequential_matrix_addition(A11, A22, tmp1, halfSize);
        sequential_matrix_addition(B11, B22, tmp2, halfSize);
        mpi_parallel_matrix_multiplication_strassen(tmp1, tmp2, M1, halfSize, size, rank);
    }

    // calculate M2 = (A21 + A22)B11
    if (1 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_addition(A21, A22, tmp1, halfSize);
        mpi_parallel_matrix_multiplication_strassen(tmp1, B11, M2, halfSize, size, rank);
    }

    // calculate M3 = A11(B12 - B22)
    if (2 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_subtraction(B12, B22, tmp1, halfSize);
        mpi_parallel_matrix_multiplication_strassen(A11, tmp1, M3, halfSize, size, rank);
    }

    // calculate M4 = A22(B21 - B11)
    if (3 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_subtraction(B21, B11, tmp1, halfSize);
        mpi_parallel_matrix_multiplication_strassen(A22, tmp1, M4, halfSize, size, rank);
    }

    // calculate M5 = (A11 + A12)B22
    if (4 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_addition(A11, A12, tmp1, halfSize);
        mpi_parallel_matrix_multiplication_strassen(tmp1, B22, M5, halfSize, size, rank);
    }

    // calculate M6 = (A21 - A11)(B11 + B12)
    if (5 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        sequential_matrix_subtraction(A21, A11, tmp1, halfSize);
        sequential_matrix_addition(B11, B12, tmp2, halfSize);
        mpi_parallel_matrix_multiplication_strassen(tmp1, tmp2, M6, halfSize, size, rank);
    }

    // calculate M7 = (A12 - A22)(B21 + B22)
    if (6 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        sequential_matrix_subtraction(A12, A22, tmp1, halfSize);
        sequential_matrix_addition(B21, B22, tmp2, halfSize);
        mpi_parallel_matrix_multiplication_strassen(tmp1, tmp2, M7, halfSize, size, rank);
    }

    // Aggregate results from all processes
    // Gather all M matrices to root process
    std::vector<int> flat_buffer(halfSize * halfSize);

    // Broadcast M1 to all processes
    if (rank == 0 % size) {
        for (int i = 0; i < halfSize; ++i) {
            for (int j = 0; j < halfSize; ++j) {
                flat_buffer[i * halfSize + j] = M1[i][j];
            }
        }
    }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 0 % size, MPI_COMM_WORLD);

    if (rank != 0 % size) {
        for (int i = 0; i < halfSize; ++i) {
            for (int j = 0; j < halfSize; ++j) {
                M1[i][j] = flat_buffer[i * halfSize + j];
            }
        }
    }

    // Similar broadcasting should be done for M2 to M7
    if (rank == 1 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M2[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 1 % size, MPI_COMM_WORLD);
    if (rank != 1 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M2[i][j] = flat_buffer[i*halfSize + j]; }
    
    // Tương tự cho M3 đến M7
    if (rank == 2 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M3[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 2 % size, MPI_COMM_WORLD);
    if (rank != 2 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M3[i][j] = flat_buffer[i*halfSize + j]; }

    if (rank == 3 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M4[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 3 % size, MPI_COMM_WORLD);
    if (rank != 3 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M4[i][j] = flat_buffer[i*halfSize + j]; }

    if (rank == 4 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M5[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 4 % size, MPI_COMM_WORLD);
    if (rank != 4 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M5[i][j] = flat_buffer[i*halfSize + j]; }

    if (rank == 5 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M6[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 5 % size, MPI_COMM_WORLD);
    if (rank != 5 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M6[i][j] = flat_buffer[i*halfSize + j]; }

    if (rank == 6 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) flat_buffer[i*halfSize + j] = M7[i][j]; }
    MPI_Bcast(flat_buffer.data(), halfSize * halfSize, MPI_INT, 6 % size, MPI_COMM_WORLD);
    if (rank != 6 % size) { for(int i=0; i<halfSize; ++i) for(int j=0; j<halfSize; ++j) M7[i][j] = flat_buffer[i*halfSize + j]; }

    // assemble final result for all processes
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(5)
    for (int i = 0; i < halfSize; ++i) {
            for (int j = 0; j < halfSize; ++j) {
                res[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];          // C11
                res[i][j + halfSize] = M3[i][j] + M5[i][j];                     // C12
                res[i + halfSize][j] = M2[i][j] + M4[i][j];                     // C21
                res[i + halfSize][j + halfSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j]; // C22
           }
     }
}