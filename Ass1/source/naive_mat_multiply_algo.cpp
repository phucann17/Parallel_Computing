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


//#################################################################################################################
// Transpose
/*
  The reason for transposing a matrix in multiplication:
  When a matrix is stored in RAM in C/C++, it saves via row, 
  meaning consecutive elements of a row are stored contiguously in memory.
 
  In naive matrix multiplication:
       C[i][j] = sum_k( A[i][k] * B[k][j] )
 
  According to Computer Architecture:
    - Accessing a row of matrix A (A[i][k]) is efficient because elements are stored
    contiguously in memory. When the CPU accesses an element that
    is not in cache (a cache miss), it loads an entire cache line containing
    consecutive elements of the row from RAM. Subsequent accesses to elements
    in the same row are likely to hit the cache, which is much faster than accessing RAM.

    - Accessing a column of matrix B (B[k][j]) is inefficient because the elements
    of the column are not contiguous in memory. Each access jumps to a different
    row in memory, likely triggering a cache miss each time. This increases
    the number of memory accesses to RAM and significantly slows down computation.
 
  Now, using both A[i][k] and B^T[j][k] will improve cache locality → fewer cache misses → significantly faster computation.
 
 Visual example:
 
 Original B:
   B[0][0] B[0][1] B[0][2] B[0][3]
   B[1][0] B[1][1] B[1][2] B[1][3]
   B[2][0] B[2][1] B[2][2] B[2][3]
   B[3][0] B[3][1] B[3][2] B[3][3]
 
 Accessing column B[0][*] jumps in memory → cache misses.
 
 Transposed B^T:
  B[0][0] B[1][0] B[2][0] B[3][0]
  B[0][1] B[1][1] B[2][1] B[3][1]
  B[0][2] B[1][2] B[2][2] B[3][2]
  B[0][3] B[1][3] B[2][3] B[3][3]
 
 Accessing row B^T[j][*] is contiguous → cache-friendly → faster computation.
 */
matrix transpose(const matrix& B) {
    unsigned int n = B.size();
    matrix BT = create_matrix(n);
    // #pragma omp parallel for collapse(2) schedule(static) num_threads(8)
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            BT[j][i] = B[i][j];

    return BT;
}

void sequential_transpose_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    matrix BT = create_matrix(n);
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j){
               BT[j][i] = B[i][j];
    }
    for (unsigned int i = 0; i < n; ++i) {
           for (unsigned int j = 0; j < n; ++j) {
               int tmp = 0;
               for (unsigned int k = 0; k < n; ++k)
                   tmp += A[i][k] * BT[j][k];
               res[i][j] = tmp;
           }
   }
}

void openMP_transpose_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n, int num_thread){
    matrix BT = create_matrix(n);

    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j){
               BT[j][i] = B[i][j];
    }
                
    for (unsigned int i = 0; i < n; ++i) {
           for (unsigned int j = 0; j < n; ++j) {
               int tmp = 0;
               for (unsigned int k = 0; k < n; ++k)
                   tmp += A[i][k] * BT[j][k];
               res[i][j] = tmp;
           }
   }
}

void openMP_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    // #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_THREADS)
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

void mpi_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n, int size, int rank) {

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
            local_B[j][i] = Matrix_flat_B[i * n + j];
        }
    }

    int num_rows_local = end_row - start_row;
    matrix local_res(num_rows_local, std::vector<int>(n, 0));

    for (int i = 0; i < num_rows_local; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k) {
                tmp += A[start_row + i][k] * local_B[j][k];
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