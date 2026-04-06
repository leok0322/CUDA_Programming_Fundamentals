/*
 * atomatic.cu
 *
 * 原子操作示例：以"数组求和"为例，演示三种实现方式的性能差异
 *
 *   1. naive_atomic       : 每个线程直接 atomicAdd → 严重争抢
 *   2. smem_reduction     : block 内 smem 树形规约 + 每 block 一次 atomicAdd
 *   3. betterReduction    : 同上，用 cuda::atomic_ref（C++17 原子引用风格）
 *
 * 原子操作基础：
 *   atomicAdd(addr, val) 等价于：
 *     lock(addr)
 *     old = *addr
 *     *addr = old + val
 *     unlock(addr)
 *     return old
 *   硬件保证 read-modify-write 的原子性，但多线程争抢同一地址会串行化。
 *
 * 优化思路：
 *   争抢者越少越好。N 个线程全部 atomicAdd → B 个 block 各做一次 atomicAdd，
 *   争抢次数从 N 降到 B（B = N/blockDim）。
 *
 * 编译：nvcc -O2 -arch=sm_75 --std=c++17 atomatic.cu -o atomatic
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda/atomic>          // cuda::atomic_ref（CUDA 11+，需要 C++17）
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define N       (1 << 22)   // 4M 个元素
#define BLOCK   256
#define GRID    ((N + BLOCK - 1) / BLOCK)
#define NWARM   5
#define NREP    50

// ===============================
// Kernel 1: Naive atomic — 每个线程直接 atomicAdd 到全局 result
//
//   所有 N 个线程争抢同一个地址 → 完全串行化
//   吞吐量 ≈ 1 次原子操作/周期，N=4M 需要 ~4M 周期
//   这是原子操作最差的使用方式，仅作对比基准。
// ===============================
__global__ void naive_atomic(int n, const float* __restrict__ array,
                              float* result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        atomicAdd(result, array[tid]);   // N 个线程全部争抢同一地址
}

// ==============================
// Kernel 2: Smem reduction + atomic — block 内树形规约，每 block 仅一次 atomic
//
//   Step 1: 每个线程把全局内存数据装入 smem（合并读）
//   Step 2: block 内做 log2(BLOCK)=8 步树形规约，每步减半参与线程
//   Step 3: thread 0 用 atomicAdd 把 block 的局部和累加到全局 result
//
//   争抢次数：N → GRID = N/BLOCK（减少 BLOCK 倍）
//
//   树形规约步骤（BLOCK=8 示意）：
//     初始: [a0, a1, a2, a3, a4, a5, a6, a7]
//     s=4:  [a0+a4, a1+a5, a2+a6, a3+a7, -, -, -, -]
//     s=2:  [a0+a4+a2+a6, a1+a5+a3+a7, -, -, -, -, -, -]
//     s=1:  [sum, -, -, -, -, -, -, -]
// ================================
__global__ void smem_reduction(int n, const float* __restrict__ array,
                                float* result)

{
    __shared__ float smem[BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: 全局内存 → smem（越界补 0）
    smem[threadIdx.x] = (tid < n) ? array[tid] : 0.f;
    __syncthreads();

    // Step 2: block 内树形规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    // ● s >>= 1 是右移赋值，等价于 s = s / 2。整个 for 循环是一个二分折半的过程：

    //   for (int s = blockDim.x / 2; s > 0; s >>= 1)
    //   // s: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 → 0（退出）

    //   ---
    //   配合树形规约理解（BLOCK=8 简化示意）：

    //   初始 smem: [a0, a1, a2, a3, a4, a5, a6, a7]
    //               T0   T1   T2   T3   T4   T5   T6   T7

    //   s=4，threadIdx.x < 4 的线程执行：
    //     T0: smem[0] += smem[4]   →  a0+a4
    //     T1: smem[1] += smem[5]   →  a1+a5
    //     T2: smem[2] += smem[6]   →  a2+a6
    //     T3: smem[3] += smem[7]   →  a3+a7
    //     smem: [a0+a4, a1+a5, a2+a6, a3+a7, -, -, -, -]

    //   s=2，threadIdx.x < 2 的线程执行：
    //     T0: smem[0] += smem[2]   →  (a0+a4)+(a2+a6)
    //     T1: smem[1] += smem[3]   →  (a1+a5)+(a3+a7)
    //     smem: [a0+a2+a4+a6, a1+a3+a5+a7, -, -, -, -, -, -]

    //   s=1，threadIdx.x < 1 的线程执行：
    //     T0: smem[0] += smem[1]   →  全部8个元素的和
    //     smem: [sum, -, -, -, -, -, -, -]

    //   s=0，退出循环

    //   ---
    //   为什么用 >>=1 而不是 s/2：

    //   s >>= 1   // 位运算，编译器直接生成 SHR 指令，略快
    //   s = s / 2 // 除法，编译器通常也会优化成移位，实际无差异

    // Step 3: 每 block 只有 thread 0 做一次 atomicAdd
    if (threadIdx.x == 0)
        atomicAdd(result, smem[0]);
}

// ========================================
// Kernel 3: betterReduction — 与 smem_reduction 逻辑相同，
//           但用 cuda::atomic_ref 替代 atomicAdd，演示 C++ 原子引用风格
//
//   cuda::atomic_ref<T, Scope> 是 C++20 atomic_ref 的 CUDA 扩展：
//     - T     : 操作的数据类型
//     - Scope : 原子操作的可见范围
//         cuda::thread_scope_block   : block 内可见
//         cuda::thread_scope_device  : 整个 GPU 可见（跨 block）
//         cuda::thread_scope_system  : GPU + CPU 可见（统一内存）
//
//   fetch_add(val) 等价于 atomicAdd，返回旧值。
//   优势：与 C++ 标准库风格一致，可读性更好，支持更多操作（fetch_min/max 等）。
// =======================================
__global__ void betterReduction(int n, const float* __restrict__ array,
                                 float* result)
{
    __shared__ float smem[BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: 全局内存 → smem
    smem[threadIdx.x] = (tid < n) ? array[tid] : 0.f;
    __syncthreads();

    // Step 2: block 内 smem 树形规约（log2(BLOCK)=8 步）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    // Step 3: 只有 thread 0 做一次 atomic（争抢者从 N 降到 GRID）
    if (threadIdx.x == 0) {
        // atomic_ref 包装已分配好的全局内存地址，不额外分配内存
        cuda::atomic_ref<float, cuda::thread_scope_device> ref(*result);
        ref.fetch_add(smem[0]);   // atomic 只在这里出现一次
    }
}


// ● 功能完全相同，区别在于风格和能力：

//   ---                                         
//   1. 语法风格                             

//   // C 风格：一行搞定                                                                                                                                       
//   atomicAdd(result, smem[0]);
                                                                                                                                                            
//   // C++ 风格：先创建引用对象，再操作                                                                                                                     
//   cuda::atomic_ref<float, cuda::thread_scope_device> ref(*result);
//   ref.fetch_add(smem[0]);

//   atomic_ref 只是对已有内存地址的包装，不额外分配内存，ref(*result) 中的 *result 就是解引用取得 float 本身。

//   ---
//   2. 最关键的区别：Scope（可见范围）

//   atomicAdd 的 scope 是隐式的，由编译器根据指针类型推断。atomic_ref 的 scope 是显式指定的：

//   // scope_block：只保证 block 内原子性，用于 smem
//   cuda::atomic_ref<float, cuda::thread_scope_block>  ref(smem[0]);

//   // scope_device：保证整个 GPU 原子性，用于跨 block 的全局内存
//   cuda::atomic_ref<float, cuda::thread_scope_device> ref(*result);

//   // scope_system：保证 GPU + CPU 原子性，用于统一内存
//   cuda::atomic_ref<float, cuda::thread_scope_system> ref(*result);

//   scope 越小，硬件开销越低：

//   ┌─────────────────────┬───────────────────────┬──────┐
//   │        scope        │       使用场景        │ 开销 │
//   ├─────────────────────┼───────────────────────┼──────┤
//   │ thread_scope_block  │ smem 内 block 内规约  │ 最小 │
//   ├─────────────────────┼───────────────────────┼──────┤
//   │ thread_scope_device │ 全局内存跨 block 累加 │ 中等 │
//   ├─────────────────────┼───────────────────────┼──────┤
//   │ thread_scope_system │ CPU/GPU 共享内存同步  │ 最大 │
//   └─────────────────────┴───────────────────────┴──────┘

//   ---
//   3. 支持的操作

//   atomicAdd 只做加法。atomic_ref 支持完整的原子操作集：

//   ref.fetch_add(val)   // 原子加
//   ref.fetch_sub(val)   // 原子减
//   ref.fetch_min(val)   // 原子取最小
//   ref.fetch_max(val)   // 原子取最大
//   ref.fetch_and(val)   // 原子与
//   ref.fetch_or(val)    // 原子或
//   ref.store(val)       // 原子写
//   ref.load()           // 原子读
//   ref.exchange(val)    // 原子交换
//   ref.compare_exchange_strong(expected, val)  // CAS

//   对应的 C 风格函数需要分别调用 atomicMin、atomicMax、atomicCAS 等。

// =================
// 计时辅助
// ===============
typedef void (*KernelFn)(int, const float*, float*);

static float bench(KernelFn kernel, int n,
                   const float* d_array, float* d_result)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < NWARM; i++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        kernel<<<GRID, BLOCK>>>(n, d_array, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NREP; i++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        kernel<<<GRID, BLOCK>>>(n, d_array, d_result);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / NREP;
}

// ==========================
// 正确性验证：与 CPU 结果对比
// ===========================
static void verify(float gpu_result, float cpu_result, const char* name)
{
    // 浮点求和顺序不同，允许相对误差 0.1%
    float rel_err = fabsf(gpu_result - cpu_result) / fabsf(cpu_result);
    if (rel_err > 1e-3f) {
        fprintf(stderr, "[%s] MISMATCH: gpu=%.4f cpu=%.4f rel_err=%.6f\n",
                name, gpu_result, cpu_result, rel_err);
        exit(1);
    }
    printf("[%s] OK  gpu=%.2f  cpu=%.2f  rel_err=%.2e\n",
           name, gpu_result, cpu_result, rel_err);
}

// ==============
// main
// ==============
int main()
{
    // ── host 内存 ────────
    float* h_array = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_array[i] = 1.0f;  // 每个元素 = 1，sum = N
    float cpu_sum = (float)N;

    // ── device 内存 ────────
    float *d_array, *d_result;
    CUDA_CHECK(cudaMalloc(&d_array,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_array, h_array, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── 正确性验证 ───────
    float h_out = 0.f;

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    naive_atomic<<<GRID, BLOCK>>>(N, d_array, d_result);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_out, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_out, cpu_sum, "naive_atomic  ");

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    smem_reduction<<<GRID, BLOCK>>>(N, d_array, d_result);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_out, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_out, cpu_sum, "smem_reduction");

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
    betterReduction<<<GRID, BLOCK>>>(N, d_array, d_result);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_out, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    verify(h_out, cpu_sum, "betterReduction");

    printf("\n");

    // ── 性能测试 ─────
    float t_naive  = bench(naive_atomic,    N, d_array, d_result);
    float t_smem   = bench(smem_reduction,  N, d_array, d_result);
    float t_better = bench(betterReduction, N, d_array, d_result);

    printf("===== Atomic Reduction (N=%dM, BLOCK=%d) =====\n\n",
           N >> 20, BLOCK);
    printf("%-20s  time=%7.4f ms   atomic次数=%d\n",
           "naive_atomic", t_naive, N);
    printf("%-20s  time=%7.4f ms   atomic次数=%d   speedup=%.1fx\n",
           "smem_reduction", t_smem, GRID, t_naive / t_smem);
    printf("%-20s  time=%7.4f ms   atomic次数=%d   speedup=%.1fx\n",
           "betterReduction", t_better, GRID, t_naive / t_better);

    printf("\n原子操作争抢分析：\n");
    printf("  naive_atomic   : %d 线程全部争抢同一地址 → 完全串行\n", N);
    printf("  smem_reduction : block 内 smem 规约后，仅 %d 个 block 各做一次 atomic\n", GRID);
    printf("  betterReduction: 与 smem_reduction 等价，使用 cuda::atomic_ref 风格\n");
    printf("\natomicAdd vs cuda::atomic_ref::fetch_add：\n");
    printf("  atomicAdd(ptr, val)                          → C 风格，简洁\n");
    printf("  atomic_ref<T,Scope>(ref).fetch_add(val)      → C++ 风格，Scope 可控\n");
    printf("  thread_scope_device : 跨 block 可见（本例需要）\n");
    printf("  thread_scope_block  : 仅 block 内（smem atomic 用，性能更好）\n");
    printf("  thread_scope_system : GPU+CPU 统一内存原子（最慢）\n");

    // ── 清理 ──────
    CUDA_CHECK(cudaFree(d_array));
    CUDA_CHECK(cudaFree(d_result));
    free(h_array);
    return 0;
}


// ● 编译命令需要加上 -arch=sm_75 和 --std=c++17：
                                                                                          
//   nvcc -arch=sm_75 --std=c++17 atomatic.cu -o at
                                              
//   cuda::atomic_ref 的 thread scope 功能要求 PTX target sm_60+，不指定 -arch 时默认生成 sm_52 PTX，所以报错。
