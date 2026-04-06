#include <cuda_runtime.h>
#include <stdio.h>

#define N          1024
#define BLOCK_SIZE 256

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t result = expr;                                      \
    if (result != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s:%d = %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(result));    \
    }                                                               \
} while(0)

// __device__ 静态分配 HBM global memory
// 大小编译期确定，程序全程存在，无需 cudaFree
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

// kernel 直接用变量名访问，无需作为参数传入
__global__ void vecAdd(int vectorLength)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vectorLength)
    {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main(int argc, char **argv)
{
    float h_A[N], h_B[N], h_C[N] {};

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // __device__ 变量必须用 cudaMemcpyToSymbol 传数据，不能用 cudaMemcpy
    CUDA_CHECK(cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float)));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAdd<<<numBlocks, BLOCK_SIZE>>>(N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // __device__ 变量必须用 cudaMemcpyFromSymbol 取回数据
    CUDA_CHECK(cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float)));

    // Verify: A[i] + B[i] = i + (N - i) = N = 1024
    printf("First 8 results:\n");
    for (int i = 0; i < 8; i++)
        printf("  h_C[%d] = %.1f\n", i, h_C[i]);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != (float)N) {
            printf("Mismatch at [%d]: %.1f\n", i, h_C[i]);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "PASSED" : "FAILED");

    // 无需 cudaFree，__device__ 变量随程序结束自动释放

    return 0;
}
