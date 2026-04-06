/*
 * matrix_transpose_with_GM.cu
 *
 * 用 HBM（Global Memory）实现矩阵转置，演示两种访问模式的性能差异：
 *
 *   1. naive_transpose   : 读合并 + 写非合并（写散乱）
 *   2. coalesced_transpose: 读非合并 + 写合并（读散乱）
 *
 * 理论背景：
 *   HBM 以 sector（32 bytes）为最小传输单位，warp 内连续线程访问连续地址时
 *   可以合并为少数几个 sector transaction（coalesced），否则每个线程单独
 *   发起一次 transaction（uncoalesced）。
 *
 *   对于矩阵转置，读和写不能同时合并（转置本身就是行列互换），只能选其一：
 *   - naive：按行读（合并）→ 按列写（非合并）：写散乱，性能较差
 *   - coalesced：按列读（非合并）→ 按行写（合并）：读散乱，性能略好
 *   实践中两者差异取决于 L2 cache 命中率，现代 GPU 差距已较小。
 *
 * 编译：nvcc -O2 -arch=sm_75 matrix_transpose_with_GM.cu -o gm_transpose
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

/* 用二维索引访问行主序一维数组：row * ld + col */
#define INDX(row, col, ld) ((row) * (ld) + (col))

#define M      1024    // 矩阵边长（M × M）
#define TILE   32      // block tile 边长：每个 block 处理 TILE×TILE 个元素
#define NWARM  5
#define NREP   100

// =============================================================================
// Kernel 1: Naive transpose — 读合并，写非合并
//
//   每个线程：
//     读  a[myRow][myCol] → 同 warp 的线程 myCol 连续 → 读合并（coalesced read）✓
//     写  c[myCol][myRow] → 同 warp 的线程 myRow 连续，但 myCol 不同 → 写非合并 ✗
//
//   内存访问图示（warp 内 T0~T31，同一行，myRow 相同，myCol=0..31）：
//
//     读 a：a[row][0], a[row][1], ..., a[row][31]  → 连续地址 → 1 个 sector transaction
//     写 c：c[0][row], c[1][row], ..., c[31][row]  → 列方向，步长 M → 每个元素不同 sector
//           → 最多 32 个 sector transaction（完全非合并）
// =============================================================================
__global__ void naive_transpose(const float* __restrict__ a,
                                 float* __restrict__ c, int m)
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;  // x 方向 → 列
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;  // y 方向 → 行

    if (myRow < m && myCol < m) {
        // 读：a[myRow][myCol] — 同 warp threadIdx.x 连续 → 合并读
        // 写：c[myCol][myRow] — 同 warp 写到不同列 → 非合并写
        c[INDX(myCol, myRow, m)] = a[INDX(myRow, myCol, m)];
    }
}

// =============================================================================
// Kernel 2: Coalesced transpose — 读非合并，写合并
//
//   交换读写的合并性：
//     读  a[myCol][myRow] → 同 warp myRow 连续，但 myCol 不同 → 读非合并 ✗
//     写  c[myRow][myCol] → 同 warp myCol 连续 → 写合并 ✓
//
//   实践中写合并往往比读合并更重要，因为写操作需要占用写缓冲，
//   非合并写会导致大量独立的 RMW（read-modify-write）事务。
// =============================================================================
__global__ void coalesced_transpose(const float* __restrict__ a,
                                     float* __restrict__ c, int m)
{
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;

    if (myRow < m && myCol < m) {
        // 读：a[myCol][myRow] — 非合并读
        // 写：c[myRow][myCol] — 合并写
        c[INDX(myRow, myCol, m)] = a[INDX(myCol, myRow, m)];
    }
}

// =============================================================================
// 计时辅助
// =============================================================================
typedef void (*KernelFn)(const float*, float*, int);

static float bench(KernelFn kernel, const float* d_a, float* d_c,
                   int m, dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // warmup
    for (int i = 0; i < NWARM; i++) {
        kernel<<<grid, block>>>(d_a, d_c, m);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // timed
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

// =============================================================================
// 正确性验证：CPU 转置结果与 GPU 结果逐元素比对
// =============================================================================

// ● static 在这里的作用是限制函数的链接可见性，仅在本编译单元（.cu 文件）内可见。

//   具体原因：

//   1. 防止符号冲突

//   verify 和 bench 是很通用的名字。如果两个 .cu / .cpp 文件都定义了同名的非 static 函数，链接器会报 multiple definition 错误：

//   # 假设两个文件都有 float bench(...)
//   ld: error: multiple definition of `bench`

//   加了 static 后，每个文件的 bench 都是"私有的"，互不干扰。

//   2. 辅助函数不需要对外暴露

//   verify 和 bench 只是 main 的辅助，没有理由让其他翻译单元调用它们。static 准确表达了这个意图："这是内部实现细节"。

//   3. 允许编译器更激进地优化

//   static 函数调用方固定在本文件内，编译器可以安全地做内联（inline）或其他跨函数优化，而非 static 的函数编译器必须保留其外部可调用的版本。

static void verify(const float* h_ref, const float* h_gpu, int m)
{
    for (int i = 0; i < m * m; i++) {
        if (h_ref[i] != h_gpu[i]) {
            fprintf(stderr, "MISMATCH at index %d: ref=%.4f gpu=%.4f\n",
                    i, h_ref[i], h_gpu[i]);
            exit(1);
        }
    }
}

// =============================================================================
// main
// =============================================================================
int main()
{
    const int m    = M;
    const int size = m * m * sizeof(float);

    // ── host 内存 ────────────────────────────────────────────────────────────
    float *h_a   = (float*)malloc(size);  // 普通内存，不是锁页内存
    float *h_ref = (float*)malloc(size);   // CPU 转置参考结果
    float *h_out = (float*)malloc(size);   // GPU 结果回传缓冲

    // 初始化输入矩阵
    for (int i = 0; i < m * m; i++) h_a[i] = (float)i;

    // CPU 参考转置
    for (int r = 0; r < m; r++)
        for (int c = 0; c < m; c++)
            h_ref[INDX(c, r, m)] = h_a[INDX(r, c, m)];

    // ── device 内存 ──────────────────────────────────────────────────────────
    float *d_a, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // ── launch 配置 ──────────────────────────────────────────────────────────
    // block: TILE×TILE 个线程（每个线程处理一个元素）
    // grid:  ceil(M/TILE) × ceil(M/TILE) 个 block
    dim3 block(TILE, TILE);
    dim3 grid((m + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    // ── 正确性验证 ────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemset(d_c, 0, size));
    naive_transpose<<<grid, block>>>(d_a, d_c, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, m);

    CUDA_CHECK(cudaMemset(d_c, 0, size));
    coalesced_transpose<<<grid, block>>>(d_a, d_c, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_c, size, cudaMemcpyDeviceToHost));
    verify(h_ref, h_out, m);

    printf("correctness check passed\n\n");

    // ── 性能测试 ──────────────────────────────────────────────────────────────
    float t_naive     = bench(naive_transpose,     d_a, d_c, m, grid, block);
    float t_coalesced = bench(coalesced_transpose, d_a, d_c, m, grid, block);

    // 理论带宽：读 M×M floats + 写 M×M floats
    double bytes      = 2.0 * m * m * sizeof(float);
    double bw_naive   = bytes / (t_naive     * 1e-3) / 1e9;  // GB/s
    double bw_coal    = bytes / (t_coalesced * 1e-3) / 1e9;

    printf("===== Matrix Transpose (HBM only, %dx%d) =====\n\n", m, m);
    printf("%-30s  time=%7.4f ms  BW=%6.2f GB/s\n",
           "naive (read-coalesced)",      t_naive,     bw_naive);
    printf("%-30s  time=%7.4f ms  BW=%6.2f GB/s  speedup=%.2fx\n",
           "coalesced (write-coalesced)", t_coalesced, bw_coal,
           t_naive / t_coalesced);

    printf("\n访问模式对比：\n");
    printf("  naive     : 读合并（行方向）→ 写非合并（列方向散乱写）\n");
    printf("  coalesced : 读非合并        → 写合并（行方向）\n");
    printf("  共同局限  : 读写之一必然非合并，彻底解决需借助 Shared Memory\n");

    // ── 清理 ──────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_ref);
    free(h_out);
    return 0;
}
