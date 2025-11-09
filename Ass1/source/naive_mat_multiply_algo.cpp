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

matrix transpose_parallel(const matrix& B) {
    unsigned int n = B.size();
    matrix BT = create_matrix(n);
    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            BT[j][i] = B[i][j];

    return BT;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// void sequential_transpose_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n){
//     matrix BT = transpose_parallel(B);

//     //#pragma omp parallel for collapse(2) schedule(static) num_threads(10)
//     for (unsigned int i = 0; i < n; ++i) {
//         for (unsigned int j = 0; j < n; ++j) {
//             int tmp = 0;
//             for (unsigned int k = 0; k < n; ++k)
//                 tmp += A[i][k] * BT[j][k];
//             res[i][j] = tmp;
//         }
//     }
// }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void openMP_transpose_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n){
    matrix BT = transpose_parallel(B);

    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k)
                tmp += A[i][k] * BT[j][k];
            res[i][j] = tmp;
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void openMP_parallel_matrix_multiplication_naive(const matrix& A, const matrix& B, matrix& res, unsigned int n) {
    #pragma omp parallel for collapse(2) schedule(static) num_threads(8)
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> flatten_matrix(const matrix& M) {
    unsigned int n = M.size();
    std::vector<int> flat(n*n);
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            flat[i*n + j] = M[i][j];
    return flat;
}

void openMP_gpu_matrix_multiply(const matrix& A, const matrix& B, matrix& res) {
    unsigned int n = A.size();
    matrix BT = transpose(B);

    int* A_flat = new int[n*n];
    int* BT_flat = new int[n*n];
    int* res_flat = new int[n*n];

    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
            A_flat[i*n+j] = A[i][j];
            BT_flat[i*n+j] = BT[j][i]; // transpose
            res_flat[i*n+j] = 0;
        }

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A_flat[0:n*n], BT_flat[0:n*n]) \
        map(from: res_flat[0:n*n])
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
            int tmp = 0;
            for (unsigned int k = 0; k < n; ++k)
                tmp += A_flat[i*n+k] * BT_flat[j*n+k];
            res_flat[i*n+j] = tmp;
        }


    #pragma omp parallel for collapse(2) schedule(static) num_threads(10)
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
            res[i][j] = res_flat[i*n+j];

    delete[] A_flat;
    delete[] BT_flat;
    delete[] res_flat;
}