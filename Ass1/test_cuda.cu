#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>
using namespace std;

// CUDA kernel: matrix multiplication
__global__ void matmul_kernel(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int tmp = 0;
        for (int k = 0; k < n; k++)
            tmp += A[row * n + k] * B[k * n + col];
        C[row * n + col] = tmp;
    }
}

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "GPU " << i << ": " << prop.name << "\n";
        cout << "  CUDA cores (approx): " << prop.multiProcessorCount * 128 << "\n";
        cout << "  SMs: " << prop.multiProcessorCount << "\n";
    }

    int n = 10000;
    size_t size = n * n * sizeof(int);

    int *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < n * n; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    double t3 = omp_get_wtime();          
    matmul_kernel<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();              
    double t4 = omp_get_wtime();          

    cout << "GPU time: " << t4 - t3 << " s\n";

    cout << "GPU matrix multiplication done!\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
