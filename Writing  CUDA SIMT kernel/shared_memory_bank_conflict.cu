/*
 * shared_memory_bank_conflict.cu
 *
 * 演示三种 Shared Memory 访问模式对 Bank Conflict 的影响：
 *   1. No Conflict     — stride-1 访问        → 0 conflict, 1 cycle
 *   2. 2-way Conflict  — stride-16 访问       → 2-way, 2 cycles
 *   3. 32-way Conflict — 全 warp 同一 bank    → 32-way, 32 cycles
 *   4. Broadcast       — 全 warp 同一地址     → 0 conflict（特殊优化）
 *   5. Padding Fix     — stride-32 + padding  → conflict 消除
 *
 * 编译：nvcc -O2 -arch=sm_80 shared_memory_bank_conflict.cu -o bank_conflict
 * 分析：ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
 *             ./bank_conflict
 *
 * Bank 的结构（32-bit mode，sm_80 默认）：
 *
 *   Bank ID = (byte_address / 4) % 32
 *
 *   Shared Memory 布局：
 *   ┌──────┬──────┬──────┬── ... ──┬──────┐
 *   │Bank 0│Bank 1│Bank 2│         │Bank31│
 *   │ 4B   │ 4B   │ 4B   │         │ 4B   │
 *   ├──────┼──────┼──────┼── ... ──┼──────┤  ← 下一行（同一 bank，不同行）
 *   │Bank 0│Bank 1│Bank 2│         │Bank31│
 *   └──────┴──────┴──────┴── ... ──┴──────┘
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define N       (1 << 14)    // 16K 元素
#define NWARM   5
#define NREP    1000

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: No Conflict — stride-1（完美分散到不同 bank）
//
//   T0 → smem[0]  → bank 0
//   T1 → smem[1]  → bank 1
//   T2 → smem[2]  → bank 2
//   ...
//   T31 → smem[31] → bank 31
//   32 个线程打到 32 个不同 bank → 完全并行 → 1 cycle
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_no_conflict(const float* __restrict__ g_in,
                                    float* __restrict__ g_out, int n)
{
    __shared__ float smem[256];

    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x + tid;

    // 每个线程 load 自己的元素（coalesced global access）
    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // stride-1 shared memory 读：T_i → smem[i]，无冲突
    if (gid < n)
        g_out[gid] = smem[tid] * 2.0f;   // bank_id = tid % 32，全部不同
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: 2-way Bank Conflict — stride-16
//
//   T0 → smem[0]  → bank 0
//   T1 → smem[16] → bank 16
//   T2 → smem[32] → bank 0  ← 撞上 T0！
//   T3 → smem[48] → bank 16 ← 撞上 T1！
//   ...
//   每个 bank 被 2 个线程访问 → 2-way conflict → 2 cycles（串行化两次）
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_2way_conflict(const float* __restrict__ g_in,
                                      float* __restrict__ g_out, int n)
{
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // stride-16：T_i 访问 smem[i * 16 % 256]
    // bank_id = (i * 16) % 32 → 每2个线程共享一个 bank
    int idx = (tid * 16) % 256;
    if (gid < n)
        g_out[gid] = smem[idx] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: 32-way Bank Conflict — stride-32（最差情况）
//
//   T0 → smem[0]   → bank 0
//   T1 → smem[32]  → bank 0 ← 撞！
//   T2 → smem[64]  → bank 0 ← 撞！
//   ...
//   T31 → smem[992] → bank 0 ← 撞！
//   所有线程都打到 bank 0 → 32-way conflict → 32 cycles（完全串行）
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_32way_conflict(const float* __restrict__ g_in,
                                       float* __restrict__ g_out, int n)
{
    // 32 个 bank-0 地址：lane*32 = 0, 32, 64, ..., 992，最大索引 992 < 1024
    __shared__ float smem[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // stride-32：用 lane = tid % 32（warp 内偏移），而非 tid
    // 原因：tid * 32 在 tid >= 32 时越界（smem 只有 1024 个元素，tid=32 → idx=1024 OOB）
    // 每个 warp 复用同一组索引 0, 32, 64, ..., 992，均在 smem[1024] 合法范围内
    // bank_id = (lane * 32) % 32 = 0 → 32 个线程全部打到 bank 0 → 32-way conflict ✓
    int lane = tid % 32;
    int idx  = lane * 32;
    if (gid < n)
        g_out[gid] = smem[idx] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: Broadcast — 全 warp 访问同一地址（特殊优化，无冲突）
//
//   T0~T31 全部访问 smem[0] → 同一地址（不是同一 bank 的不同地址）
//   硬件识别为 broadcast → 只读一次，广播给所有线程 → 1 cycle
//   注意：这和 32-way conflict 的区别：
//     Conflict: 同一 bank，不同地址 → 串行
//     Broadcast: 同一地址          → 广播优化
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_broadcast(const float* __restrict__ g_in,
                                  float* __restrict__ g_out, int n)
{
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // 所有线程读同一个地址 smem[0]
    // 不是冲突，而是 broadcast，1 cycle 完成
    if (gid < n)
        g_out[gid] = smem[0] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: Padding Fix — 用 padding 消除 stride-32 的冲突
//
//   关键思路：在每行末尾加 1 个 padding 元素（1 float = 4 bytes）
//   使相邻行的同列元素不再落在同一 bank：
//
//   无 padding（冲突）：
//   行0: [bank0][bank1]...[bank31]
//   行1: [bank0][bank1]...[bank31]  ← 同列 → 同 bank → 冲突
//
//   有 padding（+1）：
//   行0: [bank0][bank1]...[bank31][bank0_pad]
//   行1: [bank1][bank2]...[bank0]              ← 整体移位 → 不同 bank
//
//   这是矩阵转置等 kernel 的标准优化技巧。
// ─────────────────────────────────────────────────────────────────────────────
#define TILE_DIM 32
#define PADDING  1    // 每行多 1 个 float，打破 bank 对齐

__global__ void kernel_padded(const float* __restrict__ g_in,
                               float* __restrict__ g_out, int n)
{
    // 每行 33 个 float（32 有效 + 1 padding）
    __shared__ float smem[TILE_DIM][TILE_DIM + PADDING];  // smem是每一个block私有的，所以只要满足单个block就行了，和HBM不一样

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 以二维方式填充 smem
    // block 有 256 个线程，smem 是 32×33 的二维数组
    // 用 tid 计算出二维下标：
    //   row = tid / 32  → 第几行（0~7，因为 256/32=8 行）
    //   col = tid % 32  → 第几列（0~31）
    // 例：tid=0  → row=0, col=0  → smem[0][0]
    //     tid=31 → row=0, col=31 → smem[0][31]
    //     tid=32 → row=1, col=0  → smem[1][0]
    int row = tid / TILE_DIM;
    int col = tid % TILE_DIM;
    if (gid < n) smem[row][col] = g_in[gid];
    __syncthreads();

    // 转置访问：smem[col][row]，把行列互换
    // 如果没有 padding，smem 每行 32 个 float，相邻行同列地址差 = 32×4 = 128 bytes
    // bank_id = (地址/4) % 32，128/4=32，32%32=0 → 每行同列都落在 bank 0 → 32-way conflict！
    //
    // 加了 PADDING=1 后，smem 每行 33 个 float，相邻行同列地址差 = 33×4 = 132 bytes
    // bank_id = (row * 33 + col) % 32
    //   row=0, col=0 → bank (0*33+0)%32 = 0
    //   row=1, col=0 → bank (1*33+0)%32 = 1   ← 不同行同列，bank 错开 1
    //   row=2, col=0 → bank (2*33+0)%32 = 2
    //   ...
    //   row=31,col=0 → bank (31*33+0)%32 = 31
    // 不同行的同 col 落在不同 bank → 完全无冲突 ✓
    if (gid < n)
        g_out[gid] = smem[col][row] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 计时辅助
// ─────────────────────────────────────────────────────────────────────────────
typedef void (*KernelFn)(const float*, float*, int);

static float bench(KernelFn kernel, const float* d_in, float* d_out,
                   int n, int block, int grid)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < NWARM; i++) {
        kernel<<<grid, block>>>(d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NREP;  i++) kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / NREP;
}

int main()
{
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    float t_no    = bench(kernel_no_conflict,   d_in, d_out, N, BLOCK, GRID);
    float t_2way  = bench(kernel_2way_conflict, d_in, d_out, N, BLOCK, GRID);
    float t_32way = bench(kernel_32way_conflict,d_in, d_out, N, BLOCK, GRID);
    float t_bc    = bench(kernel_broadcast,     d_in, d_out, N, BLOCK, GRID);
    float t_pad   = bench(kernel_padded,        d_in, d_out, N, BLOCK, GRID);

    // printf 格式占位符说明：
    //
    // %-28s ：字符串，左对齐（-），最小宽度 28 个字符，不足时右侧补空格
    //          使不同长度的名称对齐到同一列，输出整齐
    //          无 - 时默认右对齐；有 - 时左对齐
    //
    // %7.4f ：浮点数，最小总宽度 7 个字符，小数点后保留 4 位
    //          如 0.1234、1.2345，宽度不足时左侧补空格
    //          格式：%[最小总宽度].[小数位数]f
    //
    // %.2f  ：浮点数，小数点后保留 2 位，无最小宽度限制
    //          用于 slowdown 比值，如 1.23、2.50
    //
    // x     ：格式串中的普通字符，直接输出字母 x，与 %.2f 拼接得到 "1.23x"
    //          不是格式符，只是普通字符
    //
    // 对齐效果示例：
    //   No Conflict (stride-1)       time=0.1234 ms
    //   2-way Conflict (stride-16)   time=0.2345 ms  slowdown=1.90x
    //   32-way Conflict (stride-32)  time=0.5678 ms  slowdown=4.60x
    //   ↑←────── 28字符，左对齐 ──────→↑  ↑← %7.4f →↑
    printf("\n===== Shared Memory Bank Conflict 对比 =====\n\n");
    printf("%-28s  time=%7.4f ms\n",    "No Conflict (stride-1)",    t_no);
    printf("%-28s  time=%7.4f ms  slowdown=%.2fx\n",
           "2-way Conflict (stride-16)",  t_2way,  t_2way  / t_no);
    printf("%-28s  time=%7.4f ms  slowdown=%.2fx\n",
           "32-way Conflict (stride-32)", t_32way, t_32way / t_no);
    printf("%-28s  time=%7.4f ms  slowdown=%.2fx  (broadcast 特殊优化)\n",
           "Broadcast (same addr)",       t_bc,    t_bc    / t_no);
    printf("%-28s  time=%7.4f ms  slowdown=%.2fx  (padding 修复)\n",
           "Padded (stride-32 fixed)",    t_pad,   t_pad   / t_no);

    printf("\n===== Bank ID 计算公式 =====\n");
    printf("bank_id = (byte_address / 4) %% 32\n\n");

    printf("访问模式分析（以 warp 为例）：\n");
    printf("  stride-1 : T_i → smem[i]     bank_id = i%%32           → 全部不同 ✓\n");
    printf("  stride-16: T_i → smem[i*16]  bank_id = (i*16)%%32      → 每2个撞  ✗\n");
    printf("  stride-32: T_i → smem[i*32]  bank_id = (i*32)%%32 = 0  → 全部撞   ✗\n");
    printf("  broadcast: T_i → smem[0]     同一地址 → broadcast优化   → 无冲突  ✓\n");

    printf("\n===== 用 Nsight Compute 验证 =====\n");
    printf("ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\\n");
    printf("             ./bank_conflict\n\n");
    printf("预期 bank conflict 计数：\n");
    printf("  No Conflict  → 0\n");
    printf("  2-way        → N/2  (每个 warp 1 次冲突，共 N/32 个 warp)\n");
    printf("  32-way       → 31N/32 (每个 warp 31 次 replay)\n");
    printf("  Broadcast    → 0\n");
    printf("  Padded       → 0\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
