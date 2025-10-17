#include "naive_mat_multiply_algo.h"
#include "matrix_operations.h"

void sequential_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            res[i][j] = tmp;
        } 
    }
}

void parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    #pragma omp parallel for num_threads(8)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[i][k] * B[k][j];
            }
            res[i][j] = tmp;
        } 
    }
}

// void mpi_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int rows_per_process = n / size;
//     int start_row = rank * rows_per_process;
//     int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

//     std::vector<int> Matrix_flat_B(n * n);
//     if (rank == 0) {
//         for (unsigned int i = 0; i < n; ++i) {
//             for (unsigned int j = 0; j < n; ++j) {
//                 Matrix_flat_B[i * n + j] = B[i][j];
//             }
//         }
//     }

//     // Broadcast matrix B to all processes
//     MPI_Bcast(Matrix_flat_B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

//     matrix local_B = create_matrix(n);
//     for (unsigned int i = 0; i < n; ++i) {
//         for (unsigned int j = 0; j < n; ++j) {
//             local_B[i][j] = Matrix_flat_B[i * n + j];
//         }
//     }

//     matrix local_res = create_matrix(end_row - start_row);
//     for (int i = start_row; i < end_row; ++i) {
//         for (unsigned int j = 0; j < n; ++j) {
//             int tmp = 0;
//             for (unsigned int k = 0; k < n; ++k) {
//                 tmp += A[i][k] * local_B[k][j];
//                 // tmp += A[i][k] * B[k][j];
//             }
//             local_res[i - start_row][j] = tmp;
//         } 
//     }

//     if (rank == 0) {
//         // Copy root process results
//         for (int i = start_row; i < end_row; ++i) {
//             res[i] = local_res[i - start_row];
//         }
//         // Collect results from all processes
//         for (int p = 1; p < size; ++p) {
//             int p_start_row = p * rows_per_process;
//             int p_end_row = (p == size - 1) ? n - p_start_row : rows_per_process;
//             for (int i = 0; i < p_end_row; ++i) {
//                 MPI_Recv(res[p_start_row + i].data(), n, MPI_INT, p, p_start_row + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             }
//         }
//     } else {
//         // Send local results to root process
//         for (int i = 0; i < end_row - start_row; ++i) {
//             MPI_Send(local_res[i].data(), n, MPI_INT, 0, start_row + i, MPI_COMM_WORLD);
//         }
//     }
// }

void mpi_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    std::vector<int> Matrix_flat_B(n * n);
    if (rank == 0) {
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                Matrix_flat_B[i * n + j] = B[i][j];
            }
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(Matrix_flat_B.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    matrix local_B = create_matrix(n);
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            local_B[i][j] = Matrix_flat_B[i * n + j];
        }
    }

    int num_rows_local = end_row - start_row;
    matrix local_res(num_rows_local, std::vector<int>(n, 0));

    for (int i = 0; i < num_rows_local; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[start_row + i][k] * local_B[k][j];
            }
            local_res[i][j] = tmp;
        }
    }

    if (rank == 0) {
        // Copy root process results
        for (int i = 0; i < num_rows_local; ++i) {
            res[i] = local_res[i];
        }
        // Collect results from all other processes
        for (int p = 1; p < size; ++p) {
            int p_start_row = p * rows_per_process;
            int p_rows = (p == size - 1) ? n - p_start_row : rows_per_process;
            for (int i = 0; i < p_rows; ++i) {
                MPI_Recv(res[p_start_row + i].data(), n, MPI_INT, p, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Send local results to root process
        for (int i = 0; i < num_rows_local; ++i) {
            MPI_Send(local_res[i].data(), n, MPI_INT, 0, i, MPI_COMM_WORLD);
        }
    }
}