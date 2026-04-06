/*
 * matrix_transpose_with_sm.cu
 *
 * 用 Shared Memory 实现高效矩阵转置，解决纯 HBM 方案读写不能同时合并的问题。
 *
 * 核心思路（tile 缓冲）：
 *   1. 合并读：按行从 HBM 读取一个 TILE×TILE 的数据块到 smem（读合并）
 *   2. 合并写：从 smem 转置后按行写回 HBM（写合并）
 *   smem 充当"转置缓冲区"，把两次 HBM 访问都变成合并访问。
 *
 * Bank conflict 消除：
 *   smem 每行加 1 个 padding（+1），使转置读取时相邻线程访问不同 bank。
 *   无 padding：smem[col][row]，相邻线程 col 相同、row 不同 → 步长 32 → 32-way conflict
 *   有 padding：每行 33 个元素，行间步长 33×4=132 bytes，bank 错开 → 无冲突
 *
 * 性能对比（与 HBM-only naive/coalesced 对比）：
 *   HBM naive     : 读合并 + 写非合并
 *   HBM coalesced : 读非合并 + 写合并
 *   SM transpose  : 读合并 + 写合并（两者都合并）→ 接近理论带宽峰值
 *
 * 编译：nvcc -O2 -arch=sm_75 matrix_transpose_with_sm.cu -o sm_transpose
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define INDX(row, col, ld) ((row) * (ld) + (col))

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define PADDING             1     // 每行多 1 个 float，消除 bank conflict

#define M      1024
#define NWARM  5
#define NREP   100

// ===========================
// Kernel 1: smem transpose（无 padding，存在 bank conflict）
//
//   流程：
//     ① 合并读  a[tileRow+ty][tileCol+tx] → smem[tx][ty]
//        同 warp 内 tx 连续 → 读地址连续 → 合并读 ✓
//
//     ② 合并写  smem[ty][tx] → c[tileCol+ty][tileRow+tx]
//        同 warp 内 tx 连续 → 写地址连续 → 合并写 ✓
//        但读 smem[ty][tx]：ty 相同、tx 变化，步长 32 → 32-way bank conflict ✗
//
//   smem 布局（无 padding，每行 32 个 float = 128 bytes）：
//     smem[0][0..31] → bank 0,1,...,31
//     smem[1][0..31] → bank 0,1,...,31  ← 同列不同行 → 同 bank → conflict
// ========================
__global__ void smem_transpose_no_padding(const float* __restrict__ a,
                                           float* __restrict__ c, int m)
{
    // 无 padding：每行 TILE 个 float，转置读时 32-way bank conflict
    __shared__ float smem[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    // 当前 block 负责的 tile 左上角全局坐标
    const int tileCol = blockDim.x * blockIdx.x;   // 列方向起始
    const int tileRow = blockDim.y * blockIdx.y;   // 行方向起始

    int globalRow = tileRow + threadIdx.y;
    int globalCol = tileCol + threadIdx.x;

    // ① 合并读：同 warp threadIdx.x 连续 → a 的同一行连续元素 → 合并读
    if (globalRow < m && globalCol < m)
        smem[threadIdx.x][threadIdx.y] = a[INDX(globalRow, globalCol, m)];

    __syncthreads();

    // ② 合并写：写到转置后位置，同 warp threadIdx.x 连续 → 写合并
    //    读 smem[threadIdx.y][threadIdx.x]：threadIdx.x 连续，步长 1 → bank 0,1,...,31
    //    无 padding 时 smem 每行 32 个 float，行间地址差 128 bytes，
    //    threadIdx.y 不同行同列 → 同 bank → 32-way conflict
    int transRow = tileCol + threadIdx.y;
    int transCol = tileRow + threadIdx.x;
    if (transRow < m && transCol < m)
        c[INDX(transRow, transCol, m)] = smem[threadIdx.y][threadIdx.x];
}

// ============================
// Kernel 2: smem transpose（有 padding，消除 bank conflict）★ 推荐版本
//
//   与无 padding 版本的唯一区别：smem 每行多 1 个 float（padding=1）
//   使相邻行同列元素的 bank 错开，消除转置读时的 32-way conflict。
//
//   smem 布局（有 padding，每行 33 个 float = 132 bytes）：
//     smem[0][0..31] → bank 0,1,...,31
//     smem[1][0..31] → bank 1,2,...,0  ← 行间步长 33×4=132 bytes，bank 错位 1
//     smem[k][col]   → bank (k + col) % 32
//
//   转置读 smem[ty][tx]（32 个线程 ty=0..31, tx 相同）：
//     smem[0][tx] → bank (0+tx)%32
//     smem[1][tx] → bank (1+tx)%32
//     ...
//     smem[31][tx]→ bank (31+tx)%32
//   → 32 个线程访问 32 个不同 bank → 零冲突 ✓
// =========================
__global__ void smem_transpose(const float* __restrict__ a,
                                float* __restrict__ c, int m)
{
    // 每行 TILE+PADDING 个 float，padding 列不存储有效数据，仅用于错开 bank
    __shared__ float smem[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y + PADDING];

    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;

    int globalRow = tileRow + threadIdx.y;
    int globalCol = tileCol + threadIdx.x;

    // ② 合并写 smem → c（写合并 ✓，且 smem 读无 bank conflict ✓）
    if (globalRow < m && globalCol < m)
        smem[threadIdx.x][threadIdx.y] = a[INDX(globalRow, globalCol, m)];

    __syncthreads();

    // ① 合并读 a → smem（读合并 ✓）
    int transRow = tileCol + threadIdx.y;
    int transCol = tileRow + threadIdx.x;
    if (transRow < m && transCol < m)
        c[INDX(transRow, transCol, m)] = smem[threadIdx.y][threadIdx.x];
}

// =======================
// HBM-only 基准：naive（读合并+写非合并）
// =============================
__global__ void naive_transpose(const float* __restrict__ a,
                                 float* __restrict__ c, int m)
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;
    if (myRow < m && myCol < m)
        c[INDX(myCol, myRow, m)] = a[INDX(myRow, myCol, m)];
}

// =========================
// 计时辅助
// ==========================
typedef void (*KernelFn)(const float*, float*, int);

static float bench(KernelFn kernel, const float* d_a, float* d_c,
                   int m, dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < NWARM; i++) {
        kernel<<<grid, block>>>(d_a, d_c, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NREP; i++) kernel<<<grid, block>>>(d_a, d_c, m);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / NREP;
}

// ===============================
// 正确性验证
// ==============================
static void verify(const float* h_ref, const float* h_gpu, int m,
                   const char* name)
{
    for (int i = 0; i < m * m; i++) {
        if (h_ref[i] != h_gpu[i]) {
            fprintf(stderr, "[%s] MISMATCH at %d: ref=%.4f gpu=%.4f\n",
                    name, i, h_ref[i], h_gpu[i]);
            exit(1);
        }
    }
    printf("[%s] correctness OK\n", name);
}

// ================================
// main
// ==========================
int main()
{
    const int m    = M;
    const int size = m * m * sizeof(float);

    // ── host 内存 ──────────────────────────
    float *h_a   = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < m * m; i++) h_a[i] = (float)i;
    for (int r = 0; r < m; r++)
        for (int c = 0; c < m; c++)
            h_ref[INDX(c, r, m)] = h_a[INDX(r, c, m)];

    // ── device 内存 ──────────────────────
    float *d_a, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // ── launch 配置 ────────────────────
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // ── 正确性验证 ───────────────────────
    CUDA_CHECK(cudaMemset(d_c, 0, size));
    smem_transpose_no_padding<<<grid, block>>>(d_a, d_c, m);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, m, "smem_no_padding");

    CUDA_CHECK(cudaMemset(d_c, 0, size));
    smem_transpose<<<grid, block>>>(d_a, d_c, m);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, m, "smem_padded   ");

    printf("\n");

    // ── 性能测试 ──────────────────
    float t_naive    = bench(naive_transpose,            d_a, d_c, m, grid, block);
    float t_no_pad   = bench(smem_transpose_no_padding,  d_a, d_c, m, grid, block);
    float t_padded   = bench(smem_transpose,             d_a, d_c, m, grid, block);

    double bytes   = 2.0 * m * m * sizeof(float);
    double bw_ref  = bytes / (t_naive  * 1e-3) / 1e9;
    double bw_nop  = bytes / (t_no_pad * 1e-3) / 1e9;
    double bw_pad  = bytes / (t_padded * 1e-3) / 1e9;

    printf("===== Matrix Transpose %dx%d =====\n\n", m, m);
    printf("%-36s  time=%7.4f ms  BW=%6.2f GB/s\n",
           "HBM naive (read✓ write✗)",         t_naive,  bw_ref);
    printf("%-36s  time=%7.4f ms  BW=%6.2f GB/s  speedup=%.2fx\n",
           "SM no-padding (read✓ write✓ conf✗)", t_no_pad, bw_nop,
           t_naive / t_no_pad);
    printf("%-36s  time=%7.4f ms  BW=%6.2f GB/s  speedup=%.2fx\n",
           "SM padded    (read✓ write✓ conf✓)", t_padded, bw_pad,
           t_naive / t_padded);

    printf("\n原理总结：\n");
    printf("  HBM naive   : 读合并，写非合并（列向散乱写）\n");
    printf("  SM no-pad   : 读合并，写合并，但 smem 转置读有 32-way bank conflict\n");
    printf("  SM padded   : 读合并，写合并，padding=1 消除 bank conflict → 最优 ✓\n");
    printf("\npadding 原理：\n");
    printf("  无 padding: smem 行宽 32 floats = 128 bytes，行间 bank 完全重叠\n");
    printf("  有 padding: smem 行宽 33 floats = 132 bytes，相邻行 bank 错位 1\n");
    printf("  smem[k][col] → bank = (k + col) %% 32 → 32 行各不同 bank → 无冲突\n");

    // ── 清理 ──────────────
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_ref); free(h_out);
    return 0;
}
