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

void mpi_parallel_matrix_multiplication_strassen(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int halfSize = n / 2;

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
        sequential_matrix_multiplication_strassen(tmp1, tmp2, M1, halfSize);
    }

    // calculate M2 = (A21 + A22)B11
    if (1 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_addition(A21, A22, tmp1, halfSize);
        sequential_matrix_multiplication_strassen(tmp1, B11, M2, halfSize);
    }

    // calculate M3 = A11(B12 - B22)
    if (2 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_subtraction(B12, B22, tmp1, halfSize);
        sequential_matrix_multiplication_strassen(A11, tmp1, M3, halfSize);
    }

    // calculate M4 = A22(B21 - B11)
    if (3 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_subtraction(B21, B11, tmp1, halfSize);
        sequential_matrix_multiplication_strassen(A22, tmp1, M4, halfSize);
    }

    // calculate M5 = (A11 + A12)B22
    if (4 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        sequential_matrix_addition(A11, A12, tmp1, halfSize);
        sequential_matrix_multiplication_strassen(tmp1, B22, M5, halfSize);
    }

    // calculate M6 = (A21 - A11)(B11 + B12)
    if (5 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        sequential_matrix_subtraction(A21, A11, tmp1, halfSize);
        sequential_matrix_addition(B11, B12, tmp2, halfSize);
        sequential_matrix_multiplication_strassen(tmp1, tmp2, M6, halfSize);
    }

    // calculate M7 = (A12 - A22)(B21 + B22)
    if (6 % size == rank) {
        matrix tmp1 = create_matrix(halfSize);
        matrix tmp2 = create_matrix(halfSize);
        sequential_matrix_subtraction(A12, A22, tmp1, halfSize);
        sequential_matrix_addition(B21, B22, tmp2, halfSize);
        sequential_matrix_multiplication_strassen(tmp1, tmp2, M7, halfSize);
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
    matrix C11 = create_matrix(halfSize);
    matrix C12 = create_matrix(halfSize);
    matrix C21 = create_matrix(halfSize);
    matrix C22 = create_matrix(halfSize);

    matrix tmp1 = create_matrix(halfSize);
    matrix tmp2 = create_matrix(halfSize);
    // C12 = M3 + M5
    sequential_matrix_addition(M3, M5, C12, halfSize);
    // C21 = M2 + M4
    sequential_matrix_addition(M2, M4, C21, halfSize);
    // C11 = M1 + M4 - M5 + M7
    sequential_matrix_addition(M1, M4, tmp1, halfSize);
    sequential_matrix_subtraction(tmp1, M5, tmp2, halfSize);
    sequential_matrix_addition(tmp2, M7, C11, halfSize);
    // C22 = M1 - M2 + M3 + M6
    sequential_matrix_subtraction(M1, M2, tmp1, halfSize);
    sequential_matrix_addition(tmp1, M3, tmp2, halfSize);
    sequential_matrix_addition(tmp2, M6, C22, halfSize);

    for (int i = 0; i < halfSize; ++i) {
        for (int j = 0; j < halfSize; ++j) {
            res[i][j] = C11[i][j];
            res[i][j + halfSize] = C12[i][j];
            res[i + halfSize][j] = C21[i][j];
            res[i + halfSize][j + halfSize] = C22[i][j];
        }
    }
}