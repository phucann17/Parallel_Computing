#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cmath>

using namespace std;


void random_matrix(vector<int> &M, int n) {
    #pragma omp parallel for schedule(static) num_threads(18)
    for (int i = 0; i < n * n; i++)
        M[i] = rand() % 10;
}

// CPU OpenMP matrix multiply
void matmul_cpu(const vector<int> &A, const vector<int> &BT, vector<int> &C, int n) {
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(18)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int tmp = 0;
            for (int k = 0; k < n; k++)
                tmp += A[i*n + k] * BT[j*n + k];   // LIÊN TỤC – cực nhanh
            C[i*n + j] = tmp;
        }
}

// void matmul_gpu(const vector<int> &A, const vector<int> &BT, vector<int> &C, int n)
// {
//     int *A_ptr = const_cast<int*>(A.data());
//     int *BT_ptr = const_cast<int*>(BT.data());
//     int *C_ptr = C.data();

//     #pragma omp target data \
//         map(to: A_ptr[0:n*n], BT_ptr[0:n*n]) \
//         map(from: C_ptr[0:n*n])
//     {
//         #pragma omp target teams distribute parallel for collapse(2) \
//             num_teams(480) thread_limit(256)
//         for (int i = 0; i < n; i++)
//             for (int j = 0; j < n; j++)
//             {
//                 int tmp = 0;
//                 for (int k = 0; k < n; k++)
//                     tmp += A_ptr[i*n + k] * BT_ptr[j*n + k];

//                 C_ptr[i*n + j] = tmp;
//             }
//     }
// }

void matmul_gpu(const vector<int> &A, const vector<int> &BT, vector<int> &C, int n)
{
    int *A_ptr = const_cast<int*>(A.data());
    int *BT_ptr = const_cast<int*>(BT.data());
    int *C_ptr = C.data();

    const int BK = 32;

    #pragma omp target enter data map(to: A_ptr[0:n*n], BT_ptr[0:n*n])
    #pragma omp target enter data map(alloc: C_ptr[0:n*n])

    #pragma omp target teams distribute parallel for collapse(2) \
            num_teams((n*n + 255)/256) thread_limit(256)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            int tmp = 0;

            for (int kb = 0; kb < n; kb += BK)
            {
                int limit = (kb + BK > n ? n - kb : BK);

                // ---- SIMD hóa vòng lặp k ----
                #pragma omp simd reduction(+:tmp)
                for (int k = 0; k < limit; k++)
                {
                    tmp += A_ptr[i*n + (kb+k)] * BT_ptr[j*n + (kb+k)];
                }
            }

            C_ptr[i*n + j] = tmp;
        }

    #pragma omp target update from(C_ptr[0:n*n])

    #pragma omp target exit data map(release: A_ptr[0:n*n], BT_ptr[0:n*n], C_ptr[0:n*n])
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
    int n = 10000;  
    vector<int> A(n * n), B(n * n), 
                C_cpu(n * n), 
                C_gpu_normal(n * n), 
                C_gpu_shared(n * n);

    random_matrix(A, n);
    random_matrix(B, n);

    // CPU
    double t1 = omp_get_wtime();
    matmul_cpu(A, B, C_cpu, n);
    double t2 = omp_get_wtime();
    cout << "CPU time: " << t2 - t1 << " s\n";

    // GPU normal
    double t3 = omp_get_wtime();
    matmul_gpu(A, B, C_gpu_normal, n);
    double t4 = omp_get_wtime();
    cout << "GPU time: " << t4 - t3 << " s\n";

    // GPU shared memory
    // double t5 = omp_get_wtime();
    // matmul_gpu_shared(A, B, C_gpu_shared, n);
    // double t6 = omp_get_wtime();
    // cout << "GPU SHARED time: " << t6 - t5 << " s\n";

    // Verify 1: CPU vs GPU normal
    cout << "Checking GPU normal... ";
    if (verify_matrix(C_cpu, C_gpu_normal, n))
        cout << "Correct! GPU normal = CPU.\n";
    else
        cout << "Mismatch detected (normal)!\n";

    // Verify 2: CPU vs GPU shared
    // cout << "Checking GPU shared... ";
    // if (verify_matrix(C_cpu, C_gpu_shared, n))
    //     cout << "Correct! GPU shared = CPU.\n";
    // else
    //     cout << "Mismatch detected (shared)!\n";

    // Optional: verify GPU normal vs GPU shared
    // cout << "Cross-checking GPU normal vs GPU shared... ";
    // if (verify_matrix(C_gpu_normal, C_gpu_shared, n))
    //     cout << "Both GPU versions match.\n";
    // else
    //     cout << "Mismatch between GPU versions!\n";

    return 0;
}
