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
bool verify_matrix_multiplication(const matrix& A, const matrix& B, int n) {
  for (int i = 0; i < n; ++i){
    if (A[i] != B[i]){
        return false;
    }
  }
  return true;
}
int main(){
    int n[1] = {10000};
    printf("Typing size of Matrix: ");
    scanf("%d", &n[0]);
    n_global = n[0];
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

        printf("Start running sequential transpose naive matrix multiplication!\n");
        double start = omp_get_wtime();
        sequential_transpose_matrix_multiplication_naive(A, B, res_naive_sequential, n[s]);
        double end = omp_get_wtime();

        printf("Time of sequential transpose naive matrix multiplication algorithm = %.6f s\n", end - start);
        printf("===========================================================================================\n");
        printf("Start running sequential strassen matrix multiplication!\n");
        start = omp_get_wtime();
        // cout << "hello !!!!" << endl;
        sequential_matrix_multiplication_strassen(A, B, res_strassen_sequential, n[s]);
        end = omp_get_wtime();

        printf("Time of sequential Strassen matrix multiplication algorithm = %.6f s\n", end - start);
        printf("Check valid result of sequential naive and sequential strassen: ");
        start = omp_get_wtime();
        if (verify_matrix_multiplication(res_naive_sequential, res_strassen_sequential, n[0]) == false){
            printf("Not equal!\n");
        } else {
            printf("OK!\n");
        }
        end  = omp_get_wtime();
        printf("Time to check = %.6f s\n", end - start);
        printf("===========================================================================================\n");
        // // write_log("B.txt", B, n);
        // // write_log("res.txt", res, n);  
        //###############################################################################################
        // double start_1 = omp_get_wtime();
        // openMP_parallel_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        // // openMP_transpose_parallel_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        // double end_1 = omp_get_wtime();

        // printf("Time of parallel naive matrix multiplication algorithm = %.6f s\n", end_1- start_1);
        printf("Start running parallel openMP transpose naive matrix multiplication!\n");
        double start_1 = omp_get_wtime();
        openMP_transpose_parallel_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        // openmp_parallel_matrix_multiplication_strassen(A, B, res_naive_parallel, n[s]);
        double end_1 = omp_get_wtime();
        printf("Time of parallel openMP transpose parallel naive matrix multiplication algorithm = %.6f s\n", end_1 - start_1);
        printf("Check valid result of sequential naive and parallel openMP naive: ");
        start_1 = omp_get_wtime();
        if (verify_matrix_multiplication(res_naive_sequential, res_naive_parallel, n[0]) == false){
            printf("Not equal!\n");
        } else {
            printf("OK!\n");
        }
        end_1  = omp_get_wtime();
        printf("Time to check = %.6f s\n", end_1 - start_1);
        printf("===========================================================================================\n");
        //###############################################################################################
        printf("Start running parallel openMP strassen matrix multiplication!\n");
        start_1 = omp_get_wtime();
        openmp_parallel_matrix_multiplication_strassen(A, B, res_naive_parallel, n[s]);
        end_1 = omp_get_wtime();
        printf("Time of parallel openMP strassen matrix multiplication algorithm = %.6f s\n", end_1 - start_1);
        printf("Check valid result of sequential strassen and parallel strassen openMP: ");
        start_1 = omp_get_wtime();
        if (verify_matrix_multiplication(res_naive_parallel, res_strassen_sequential, n[0]) == false){
            printf("Not equal!\n");
        } else {
            printf("OK!\n");
        }
        end_1  = omp_get_wtime();
        printf("Time to check = %.6f s\n", end_1 - start_1);

        // sequential_matrix_subtraction(res_strassen_sequential, res_naive_parallel, res_naive_sequential, n[s]);
        // write_log("A.txt", res_naive_sequential, n[s]);
        
        //###############################################################################################
    }

    return 0;
}
