#include <cuda_runtime.h>
#include <cuda/cmath>   // cuda::ceil_div (libcu++, requires CUDA 11.x+)
#include <stdio.h>

#define BLOCK_SIZE 256

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t result = expr;                                      \
    if (result != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s:%d = %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(result));    \
    }                                                               \
} while(0)

// Each thread adds one element: C[i] = A[i] + B[i]
// Global memory access: A, B, C all reside in device global memory (HBM)
// Access pattern: coalesced — consecutive threads access consecutive addresses
__global__ void vecAdd(float *A, float *B, float *C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

int main(int argc, char **argv)
{
    const int N = 1024;

    // Host arrays
    float h_A[N], h_B[N], h_C[N] {};

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device arrays (global memory)
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaGetLastError());
    // Grid size: ceil(N / BLOCK_SIZE)
    int numBlocks = cuda::ceil_div(N, BLOCK_SIZE);  // int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: A[i] + B[i] = i + (N - i) = N = 1024 for all i
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != (float)N) {
            printf("Mismatch at [%d]: %.1f\n", i, h_C[i]);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "PASSED" : "FAILED");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
