// #include <iostream>
// #include <cstdlib>
// #include <cuda_runtime.h>
// #include <omp.h>
// using namespace std;

// __global__ void matmul_kernel(int *A, int *B_T, int *C, int n) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < n && col < n) {
//         int tmp = 0;
//         for (int k = 0; k < n; k++)
//             tmp += A[row * n + k] * B_T[col * n + k];  
//         C[row * n + col] = tmp;
//     }
// }

// int main() {
//     int nDevices;
//     cudaGetDeviceCount(&nDevices);

//     for (int i = 0; i < nDevices; i++) {
//         cudaDeviceProp prop;
//         cudaGetDeviceProperties(&prop, i);
//         cout << "GPU " << i << ": " << prop.name << "\n";
//         cout << "  CUDA cores (approx): " << prop.multiProcessorCount * 128 << "\n";
//         cout << "  SMs: " << prop.multiProcessorCount << "\n";
//     }

//     int n = 3000;
//     size_t size = n * n * sizeof(int);

//     int *A, *B, *C, *B_T;
//     cudaMallocManaged(&A, size);
//     cudaMallocManaged(&B, size);
//     cudaMallocManaged(&C, size);
//     cudaMallocManaged(&B_T, size);

//     for (int i = 0; i < n*n; i++) {
//         A[i] = rand() % 10;
//         B[i] = rand() % 10;
//     }

//     for (int i = 0; i < n; i++)
//         for (int j = 0; j < n; j++)
//             B_T[j*n + i] = B[i*n + j];

//     dim3 threads(32, 32);  // 1024 threads/block
//     dim3 blocks((n + 31) / 32, (n + 31) / 32);

//     matmul_kernel<<<1,1>>>(A, B_T, C, 1);  // warm-up
//     cudaDeviceSynchronize();

//     double t3 = omp_get_wtime();          
//     matmul_kernel<<<blocks, threads>>>(A, B_T, C, n);
//     cudaDeviceSynchronize();              
//     double t4 = omp_get_wtime();          

//     cout << "GPU time: " << t4 - t3 << " s\n";
//     cout << "GPU matrix multiplication done!\n";

//     cudaFree(A);
//     cudaFree(B);
//     cudaFree(B_T);
//     cudaFree(C);

//     return 0;
// }

#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
using namespace std;

// ======================
// Kernel gốc: không shared memory
// ======================
__global__ void matmul_kernel(int *A, int *B_T, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        #pragma unroll
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B_T[col * n + k];
        }
        C[row * n + col] = sum;
    }
}

// ======================
// Kernel mới: Shared Memory (tối ưu)
// ======================
#define TILE 32
__global__ void matmul_shared_kernel(int *A, int *B_T, int *C, int n) {
    __shared__ int sA[TILE][TILE];
    __shared__ int sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int sum = 0;

    for (int k = 0; k < n; k += TILE) {
        if (row < n && (k + threadIdx.x) < n)
            sA[threadIdx.y][threadIdx.x] = A[row * n + (k + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < n && (k + threadIdx.y) < n)
            sB[threadIdx.y][threadIdx.x] = B_T[col * n + (k + threadIdx.y)];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; t++)
            sum += sA[threadIdx.y][t] * sB[t][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int n = 10000; // test
    size_t bytes = n * n * sizeof(int);

    int *A, *B, *B_T, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&B_T, bytes);
    cudaMallocManaged(&C, bytes);

    // Khởi tạo
    #pragma omp parallel for
    for (int i = 0; i < n*n; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }

    // Transpose B → B_T
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B_T[j * n + i] = B[i * n + j];

    // Warm-up
    matmul_kernel<<<dim3(1,1), dim3(1,1)>>>(A, B_T, C, n);
    cudaDeviceSynchronize();

    // ------------------------
    // Chạy kernel cũ
    // ------------------------
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (n + 31) / 32);

    auto t1 = omp_get_wtime();
    matmul_kernel<<<grid, block>>>(A, B_T, C, n);
    cudaDeviceSynchronize();
    auto t2 = omp_get_wtime();

    cout << "GPU time (naive): " << (t2 - t1) << " s\n";

    // ------------------------
    // Chạy kernel Shared Memory
    // ------------------------
    auto t3 = omp_get_wtime();
    matmul_shared_kernel<<<grid, block>>>(A, B_T, C, n);
    cudaDeviceSynchronize();
    auto t4 = omp_get_wtime();

    cout << "GPU time (shared): " << (t4 - t3) << " s\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(B_T);
    cudaFree(C);

    cout << "Done!\n";
    return 0;
}
