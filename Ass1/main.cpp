#include <iostream>
#include <omp.h>
#include <mpi.h>
#include "matrix_operations.h"
#include "naive_mat_multiply_algo.h"
#include "strassen_mat_multiply_algo.h"

using namespace std;

bool verify_matrix_multiplication(const matrix& A, const matrix& B, int n) {
    cout << "----------------------------------------" << endl;
    cout << "Bat dau kiem tra tinh dung dan..." << endl;

    // So sánh từng phần tử
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i][j] != B[i][j]) {
                // cerr << "[LOI] Phat hien ket qua sai!" << endl;
                // cerr << "Tai vi tri [" << i << "][" << j << "]:" << endl;
                // cerr << "  - Ket qua test: " << A[i][j] << endl;
                // cerr << "  - Ket qua dung : " << B[i][j] << endl;
                // cout << "----------------------------------------" << endl;
                return false;
            }
        }
    }

    cout << "[OK] Ket qua hoan toan chinh xac!" << endl;
    cout << "----------------------------------------" << endl;
    return true; 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n[1] = {10000};
    for (int s = 0; s < 1; ++s) {
        if (rank == 0) {
            cout << "Implementation with matrix size: " << n[s] << endl;
        }
        matrix A = create_matrix(n[s]);
        matrix B = create_matrix(n[s]);
        matrix res_naive_parallel = create_matrix(n[s]);
        matrix res_strassen_parallel = create_matrix(n[s]);
        if (rank == 0) {
            for (int i = 0; i < n[s]; ++i){
                for (int j = 0; j < n[s]; ++j){
                    A[i][j] = (int)rand() / (16384*30);
                    B[i][j] = (int)rand() / (16384*30);
                }
            }
        }

        // Broadcast matrix A and B to all processes
        for (int i = 0; i < n[s]; ++i) {
            MPI_Bcast(A[i].data(), n[s], MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(B[i].data(), n[s], MPI_INT, 0, MPI_COMM_WORLD);
        }

        
        // if (rank == 0){
        //     double s0 = MPI_Wtime();
        //     sequential_transpose_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        //     // sequential_matrix_multiplication_naive(A, B, res_naive_parallel, n[s]);
        //     double e0 = MPI_Wtime();
        //     printf("Time of sequential naive matrix multiplication algorithm = %.6f s\n", e0 - s0);

        //     // s0 = MPI_Wtime();
        //     // sequential_transpose_matrix_multiplication_naive(A, B, res_strassen_parallel, n[s]);
        //     // e0 = MPI_Wtime();
        //     // printf("Time of transpose sequential naive matrix multiplication algorithm = %.6f s\n", e0 - s0);
        //     // if(!verify_matrix_multiplication(res_naive_parallel, res_strassen_parallel, n[s])){
        //     //     printf("Sequential naive != Transpose sequential naive\n");
        //     // }
                
        //     s0 = MPI_Wtime();
        //     sequential_matrix_multiplication_strassen(A, B, res_strassen_parallel, n[s]);
        //     e0 = MPI_Wtime();
        //     printf("Time of sequential strassen matrix multiplication algorithm = %.6f s\n", e0 - s0);
        //     if(!verify_matrix_multiplication(res_naive_parallel, res_strassen_parallel, n[s])){
        //         printf("Sequential naive != Sequential strassen\n");
        //     }
        // }
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize before start timing
        double start_4 = MPI_Wtime();
        mpi_parallel_matrix_multiplication_strassen(A, B, res_strassen_parallel, n[s], size, rank);
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished
        double end_4 = MPI_Wtime();
        if (rank == 0) {
            printf("Time of mpi parallel strassen matrix multiplication algorithm = %.6f s\n", end_4 - start_4);
            // if(!verify_matrix_multiplication(res_naive_parallel, res_strassen_parallel, n[s])){
            //     printf("Sequential naive != Parallel strassen\n");
            // }
        }
    }

    MPI_Finalize();
    return 0;
}
