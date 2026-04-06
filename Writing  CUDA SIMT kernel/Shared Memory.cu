#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 128

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t result = expr;                                      \
    if (result != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s:%d = %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(result));    \
    }                                                               \
} while(0)

// assuming blockDim.x is 128
// Each block sums its 128 elements using shared memory
__global__ void example_syncthreads(int *input_data, int *output_data)
{
    __shared__ int shared_data[BLOCK_SIZE];  // 每一个block设置自己的shared_data

    // Every thread writes to a distinct element of 'shared_data':
    shared_data[threadIdx.x] = input_data[blockIdx.x * blockDim.x + threadIdx.x];  // 每一个block的thread负责自己的部分的input

    // 并行维度：
    // 每个warp的32个线程是并行的；
    // 多个warp被同时并行调度；每个 SM 有 4 个 Scheduler，每周期各发射 1 条指令，即每周期最多 4 个 warp 真正并行。几十个 warp 同时驻留（Ampere 最多 64），但大多数在等待内存/依赖，靠切换来隐藏延迟
    // 多个sm同时调度block。
    // 一个sm同时调度多个block是透明的。哪个 block 跑在哪个 SM 上、一个 SM 同时驻留几个 block，完全由 CUDA runtime 和硬件调度器决定，程序员看不到也控制不了。

    // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered
    // before any thread is unblocked from '__syncthreads()':
    __syncthreads();

    // A single thread safely reads 'shared_data':
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}

int main(int argc, char **argv)
{
    const int num_blocks = 4;
    const int N = num_blocks * BLOCK_SIZE;  // 512 elements total

    // Host arrays
    int h_input[N] {};
    int h_output[num_blocks] {};

    // Initialize input: h_input[i] = i
    for (int i = 0; i < N; i++) h_input[i] = i;

    // Device arrays
    int *d_input  {};
    int *d_output = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_input,  N          * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, num_blocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    example_syncthreads<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    // Each block sums BLOCK_SIZE consecutive elements
    // block 0: sum(0..127), block 1: sum(128..255), ...
    for (int b = 0; b < num_blocks; b++) {
        printf("block[%d] sum = %d\n", b, h_output[b]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
