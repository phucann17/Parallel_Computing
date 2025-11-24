#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cmath>

using namespace std;


void random_matrix(vector<int> &M, int n) {
    #pragma omp parallel for schedule(static) num_threads(20)
    for (int i = 0; i < n * n; i++)
        M[i] = rand() % 10;
}

// CPU OpenMP matrix multiply
void matmul_cpu(const vector<int> &A, const vector<int> &B, vector<int> &C, int n) {
    #pragma omp parallel for collapse(2) schedule(static) num_threads(20)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int tmp = 0;
            for (int k = 0; k < n; k++)
                tmp += A[i * n + k] * B[k * n + j];
            C[i * n + j] = tmp;
        }
}

void matmul_gpu(const vector<int> &A, const vector<int> &B, vector<int> &C, int n) {
    int *A_ptr = const_cast<int*>(A.data());
    int *B_ptr = const_cast<int*>(B.data());
    int *C_ptr = C.data();

    int num_teams = 120;        
    int thread_limit = 256;     

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A_ptr[0:n*n], B_ptr[0:n*n]) map(from: C_ptr[0:n*n]) \
        num_teams(num_teams) thread_limit(thread_limit)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int tmp = 0;
            for (int k = 0; k < n; k++)
                tmp += A_ptr[i * n + k] * B_ptr[k * n + j];
            C_ptr[i * n + j] = tmp;
        }
}


bool verify_matrix(const vector<int> &C1, const vector<int> &C2, int n) {
    int errors = 0;
    #pragma omp parallel for reduction(+:errors)
    for (int i = 0; i < n * n; i++) {
        if (C1[i] != C2[i])
            errors++;
    }
    if (errors == 0)
        return true;
    else {
        cout << "Found " << errors << " mismatched elements!\n";
        return false;
    }
}

int main() {
    int n = 2000;  
    vector<int> A(n * n), B(n * n), C_cpu(n * n), C_gpu(n * n);

    random_matrix(A, n);
    random_matrix(B, n);

    // CPU
    double t1 = omp_get_wtime();
    matmul_cpu(A, B, C_cpu, n);
    double t2 = omp_get_wtime();
    cout << "CPU time: " << t2 - t1 << " s\n";

    // GPU
    double t3 = omp_get_wtime();
    matmul_gpu(A, B, C_gpu, n);
    double t4 = omp_get_wtime();
    cout << "GPU time: " << t4 - t3 << " s\n";

    // Verify
    cout << "Checking results... ";
    if (verify_matrix(C_cpu, C_gpu, n))
        cout << "Correct! GPU = CPU.\n";
    else
        cout << " Mismatch detected!\n";

    return 0;
}
