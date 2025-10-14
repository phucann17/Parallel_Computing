#include <iostream>
#include <omp.h>
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"
#include "strassen_mat_multiply_algo.h"

using namespace std;

// void free_matrix(matrix m, unsigned int n){
//     for (int i = 0; i < n; ++i){
//         free(m[i]);
//     }
//     free(m);
// }

int main(){
    int n[1] = {2000};
    for (int s = 0; s < 1; ++s) {
        cout << "Implementation with matrix size: " << n[s] << endl;
        matrix A = create_matrix(n[s]);
        matrix B = create_matrix(n[s]);
        matrix res_naive_sequential = create_matrix(n[s]);
        matrix res_naive_parallel = create_matrix(n[s]);
        matrix res_strassen_sequential = create_matrix(n[s]);
        
        for (int i = 0; i < n[s]; ++i){
            for (int j = 0; j < n[s]; ++j){
                A[i][j] = (int)rand() / (16384*30);
                B[i][j] = (int)rand() / (16384*30);
            }
        }

        double start = omp_get_wtime();
        sequential_matrix_multiplication_naive(A, B, res_naive_sequential, n[s]);
        double end = omp_get_wtime();

        printf("Time of sequential naive matrix multiplication algorithm = %.6f s\n", end - start);
        // // write_log("A.txt", A, n);
        // // write_log("B.txt", B, n);
        // // write_log("res.txt", res, n);  

        double start_1 = omp_get_wtime();
        parallel_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        double end_1 = omp_get_wtime();

        printf("Time of parallel naive matrix multiplication algorithm = %.6f s\n", end_1- start_1);

        double start_3 = omp_get_wtime();
        cout << "hello !!!!" << endl;
        sequential_matrix_multiplication_strassen(A, B, res_strassen_sequential, n[s]);
        double end_3 = omp_get_wtime();

        printf("Time of sequential Strassen matrix multiplication algorithm = %.6f s\n", end_3 - start_3);

        // free_matrix(A, n[s]);
        // free_matrix(B, n[s]);
        // free_matrix(res_naive_sequential, n[s]);
        // free_matrix(res_naive_parallel, n[s]);
        // free_matrix(res_strassen_sequential, n[s]);
    }

    return 0;
}