#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t result = expr;                                      \
    if (result != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s:%d = %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(result));    \
    }                                                               \
} while(0)

// kernel 与普通版本相同，指针传入即可
__global__ void vecAdd(float *A, float *B, float *C, int vectorLength)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vectorLength)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    const int N = 1024;

    // Unified Memory: CPU 和 GPU 共享同一块内存，无需 cudaMemcpy
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    CUDA_CHECK(cudaMallocManaged((void **)&A, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void **)&B, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void **)&C, N * sizeof(float)));

    // CPU 直接写入，无需 cudaMemcpy
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
        C[i] = 0.0f;
    }

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAdd<<<numBlocks, BLOCK_SIZE>>>(A, B, C, N);
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize 必须调用：等待 GPU 完成后 CPU 才能安全访问 C
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU 直接读取结果，无需 cudaMemcpy
    printf("First 8 results:\n");
    for (int i = 0; i < 8; i++)
        printf("  C[%d] = %.1f\n", i, C[i]);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (C[i] != (float)N) {
            printf("Mismatch at [%d]: %.1f\n", i, C[i]);
            correct = false;
            break;
        }
    }
    printf("%s\n", correct ? "PASSED" : "FAILED");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

