/*
 * global_memory_coalescing.cu
 *
 * 演示三种 Global Memory 访问模式对 Coalescing 的影响：
 *   1. Coalesced      — stride-1, 连续访问       → 4 transactions / warp
 *   2. Unaligned      — 偏移1个元素, 跨 sector    → 5 transactions / warp
 *   3. Strided        — stride-8, 分散访问        → 32 transactions / warp
 *
 * 编译：nvcc -O2 -arch=sm_80 global_memory_coalescing.cu -o coalescing
 * 分析：ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
 *             ./coalescing
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N        (1 << 20)   // 1M floats = 4MB
// 1 << 20 = 1,048,576 个 float
//         = 1,048,576 × 4 bytes
//         = 4,194,304 bytes
//         ≈ 4 MiB（严格说是 4 MiB，不是 4 MB）
//   ┌──────┬─────────────────────────┬──────────────────────┐                                                                                   
//   │      │     MB（Megabyte）      │   MiB（Mebibyte）    │                                                                                   
//   ├──────┼─────────────────────────┼──────────────────────┤
//   │ 定义 │ 10⁶ = 1,000,000 字节    │ 2²⁰ = 1,048,576 字节 │                                                                                   
//   ├──────┼─────────────────────────┼──────────────────────┤                                                                                   
//   │ 标准 │ SI 国际单位制（十进制） │ IEC 标准（二进制）   │                                                                                   
//   ├──────┼─────────────────────────┼──────────────────────┤                                                                                   
//   │ 差异 │ —                       │ 比 MB 大约 4.9%      │                                                                                   
//   └──────┴─────────────────────────┴──────────────────────┘
#define NWARM    5
#define NREP     100

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: Coalesced Access（完美合并）
//
//   Thread i → array[i]
//   warp 内 32 个线程访问连续 128 bytes → 4 个 32-byte sector
//
//   内存布局（以一个 warp 为例）：
//   ┌──────────┬──────────┬──────────┬──────────┐
//   │ Sector 0 │ Sector 1 │ Sector 2 │ Sector 3 │
//   │  T0~T7   │  T8~T15  │  T16~T23 │  T24~T31 │
//   └──────────┴──────────┴──────────┴──────────┘
// Utilization = Bytes used by threads / Bytes fetched from DRAM
//   transactions per warp = 4, utilization = 100%
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_coalesced(const float* __restrict__ in,
                                  float* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: Unaligned Access（非对齐访问）
//
//   Thread i → array[i + 1]（整体偏移 1 个 float = 4 bytes）
//   虽然线程间还是连续的，但起始地址不在 32-byte 边界上：
//
//   不偏移时（对齐）：
//   Sector 0: [byte 0  ~ 31 ] ← T0~T7 完整落入
//   Sector 1: [byte 32 ~ 63 ] ← T8~T15 完整落入
//   ...
//
//   偏移 4 bytes 后：
//   Sector 0: [byte 0  ~ 31 ] ← T0~T6 落入（7 threads）
//   Sector 1: [byte 32 ~ 63 ] ← T7~T14 落入（跨越了 sector 边界！）
//   ...
//   Sector 4: [byte 128~159] ← T31 落入（多出一个 sector）
// Utilization = Bytes used by threads / Bytes fetched from DRAM
//   transactions per warp = 5（多一次），utilization = 80%
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_unaligned(const float* __restrict__ in,
                                  float* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 偏移 1 使起始地址不再是 32-byte 对齐
    if (tid < n - 1)
        out[tid] = in[tid + 1] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: Strided Access（跨步访问）
//
//   Thread i → array[i * STRIDE]
//   STRIDE = 8 时，相邻线程地址间距 = 32 bytes（恰好一个 sector）
//
//   内存布局：
//   T0 → byte 0     → Sector 0
//   T1 → byte 32    → Sector 1
//   T2 → byte 64    → Sector 2
//   ...
//   T31 → byte 992  → Sector 31
//
//   32 个线程 → 32 个不同 sector → 32 次 transaction
// Utilization = Bytes used by threads / Bytes fetched from DRAM
//   transactions per warp = 32, utilization = 4/32 = 12.5%
//
//   这是 AoS（Array of Structs）布局中常见的性能陷阱。
// ─────────────────────────────────────────────────────────────────────────────
#define STRIDE 8
__global__ void kernel_strided(const float* __restrict__ in,
                                float* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * STRIDE < n)
        out[tid] = in[tid * STRIDE] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 计时辅助
// ─────────────────────────────────────────────────────────────────────────────
static float bench(void (*kernel)(const float*, float*, int),
                   const float* d_in, float* d_out, int n,
                   int block, int grid)
{
    // cudaEvent_t：GPU 时间戳句柄
    //   本质是 GPU 硬件计数器的一个记录点，由驱动管理
    //   与 CPU 的 clock() 不同，event 是插入 GPU 命令流中的，
    //   记录的是 GPU 实际执行到该位置时的时间，不受 CPU-GPU 异步影响
    cudaEvent_t t0, t1;

    // cudaEventCreate：在 GPU 驱动中分配 event 对象
    //   必须先 Create 才能 Record/Synchronize/ElapsedTime
    //   对应 cudaEventDestroy 释放资源
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // warmup：预热，消除以下一次性开销，确保计时结果反映稳定的执行时间
    //
    // 1. GPU 时钟频率稳定（Boost Clock）
    //    GPU 启动时可能处于低功耗状态，首次 kernel 执行期间频率会逐步提升到 Boost Clock
    //    预热让 GPU 提前达到稳定频率，否则第一次计时偏慢
    //
    // 2. L2 Cache / L1 Cache 预热
    //    首次访问数据时 cache 是冷的（cold cache），全部 miss 走 HBM
    //    预热后 cache 进入热态（warm cache），后续计时反映的是实际工作负载的缓存行为
    //    本文件测的是 coalescing，希望排除 cold cache 的干扰
    //
    // 3. CUDA Runtime 初始化
    //    首次 kernel launch 会触发驱动懒初始化（context 建立、显存映射等）
    //    这些开销与 kernel 本身无关，预热后不再触发
    //
    // 4. JIT 编译缓存
    //    若代码路径未被编译器完全展开，首次执行可能触发 PTX → SASS 的 JIT 编译
    //    预热后编译结果已缓存
    //
    // 为什么是 5 次（NWARM=5）而不是 1 次？
    //    GPU 时钟频率从低功耗爬升到 Boost Clock 需要几毫秒，
    //    单次 kernel 执行时间可能不足以让频率完全稳定，
    //    多次预热确保频率、cache 都进入稳态。
    //    实际工程中 NWARM 通常取 3~10，视 kernel 执行时间而定。
    for (int i = 0; i < NWARM; i++)
        kernel<<<grid, block>>>(d_in, d_out, n);

    // cudaEventRecord(t0)：把 t0 插入 GPU 默认 stream 的命令队列
    //   不是立即记录当前时间，而是等 GPU 执行到这条命令时才打上时间戳
    //   CPU 调用后立即返回（异步），不等待 GPU
    cudaEventRecord(t0);
    for (int i = 0; i < NREP; i++)
        kernel<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(t1);

    // cudaEventSynchronize(t1)：CPU 阻塞等待，直到 GPU 执行完 t1 这条 event
    //   确保 t1 的时间戳已经被 GPU 写入，之后才能调用 ElapsedTime
    //   若不同步直接调用 ElapsedTime，返回值未定义
    // GPU 命令流是一个队列，t1 是插入队列中的一个时间戳记录指令。"执行完"就是 GPU 执行到队列中 t1 这条指令，把当前 GPU 时间写入 t1 对象。
    // 在 t1 之前的所有 kernel（100 次循环）都执行完毕后，GPU 才会执行到 t1 这条指令。   
    cudaEventSynchronize(t1);


    

    // cudaEventElapsedTime：计算 t0 到 t1 之间的 GPU 执行时间，单位毫秒
    //   精度约 0.5 微秒，由 GPU 硬件计数器保证
    //   只测量 GPU 上的时间，不包含 CPU 端的开销

    // 不加 cudaEventSynchronize 直接调用 ElapsedTime 会怎样：

    // cudaEventRecord(t1);
    // // GPU 还在跑 kernel，t1 时间戳还没写入
    // cudaEventElapsedTime(&ms, t0, t1);  // 读到的是未定义值，返回 cudaErrorNotReady 
    // cudaEventElapsedTime(&ms, t0, t1)  ← t1 执行完后，t0 必然也执行完了
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);


    // CPU 视角：                    GPU 命令流：                                                                                                  
    // cudaEventRecord(t1) ──插入──→ [kernel 0]
    // 立即返回                       [kernel 1]                                                                                                   
    //                                 ...                                                                                                          
    //                                 [kernel 99]                                                                                                  
    //                                 [t1 时间戳]  ← GPU 还没执行到这里                                                                            
    // cudaEventSynchronize(t1) ─ CPU 阻塞等待                                                                                                     
    //                                 GPU 执行到 t1，写入时间戳
    //                             ─ CPU 解除阻塞
    // cudaEventElapsedTime(...)      ← 现在 t1 时间戳有效，可以读取


    // cudaEventDestroy：释放 event 对象占用的驱动资源
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / NREP;
}

int main()
{
    // ── 分配内存 ──────────────────────────────────────────────────────────
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // 初始化
    cudaMemset(d_in, 0, N * sizeof(float));

    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    // ── 运行三个 kernel ───────────────────────────────────────────────────
    float t_coal = bench(kernel_coalesced, d_in, d_out, N, BLOCK, GRID);
    float t_unal = bench(kernel_unaligned, d_in, d_out, N, BLOCK, GRID);
    float t_strd = bench(kernel_strided,   d_in, d_out, N, BLOCK, GRID / STRIDE);


    // ── 带宽计算 ──────────────────────────────────────────────────────────                                               
    // 带宽（Bandwidth）= 单位时间内线程实际请求的有用数据量                                                                       
    //   有用数据 = 线程真正需要读写的字节数，不包含因 coalescing 不佳而多取的 sector                                       
    //                                                                                                                      
    // 注意：这里计算的是"有效带宽"（Effective Bandwidth），不是"实际总线带宽"                                              
    //                                                                                                                      
    //   有效带宽 = 有用字节数 / 时间                                                                                       
    //   实际总线带宽 = GPU 实际从 HBM 搬运的总字节数 / 时间（含浪费的 sector）                                             
    //                                                                                                                      
    // 例：Strided kernel，32 个线程各取 1 个 float（4 bytes），触发 32 个 sector（32×32=1024 bytes）                       
    //   有用数据   = 32 × 4 = 128 bytes                                                                                    
    //   实际搬运   = 32 × 32 = 1024 bytes                                                                                  
    //   有效带宽   = 128 bytes / t（反映线程实际吞吐）                                                                     
    //   利用率     = 128 / 1024 = 12.5%（浪费了 87.5% 的总线带宽）                                                         
    //                                                                                                                      
    // 有效带宽越接近硬件峰值带宽，说明 coalescing 越好                                                                     
    // bytes = 读字节数 + 写字节数（R + W），kernel 既读 in 又写 out                                                        
    double bytes_coal = 2.0 * N * sizeof(float);              // R + W，N 个 float 读 + N 个 float 写                       
    double bytes_strd = 2.0 * (N / STRIDE) * sizeof(float);  // strided 只用 N/STRIDE 个元素     

    printf("\n===== Global Memory Coalescing 对比 =====\n\n");

    printf("%-20s  time=%6.3f ms  BW=%6.2f GB/s  (理论 transactions/warp=%2d, util=100%%)\n",
           "Coalesced",
           t_coal,
           bytes_coal / t_coal / 1e6,
           4);

    printf("%-20s  time=%6.3f ms  BW=%6.2f GB/s  (理论 transactions/warp=%2d, util= 80%%)\n",
           "Unaligned",
           t_unal,
           bytes_coal / t_unal / 1e6,
           5);

    printf("%-20s  time=%6.3f ms  BW=%6.2f GB/s  (理论 transactions/warp=%2d, util=12.5%%)\n",
           "Strided(x8)",
           t_strd,
           bytes_strd / t_strd / 1e6,
           32);

    printf("\n");
    printf("Slowdown  Unaligned  vs Coalesced : %.2fx\n", t_unal / t_coal);
    printf("Slowdown  Strided    vs Coalesced : %.2fx\n", t_strd / t_coal);

    // ── 用 ncu 查看 sector 数量的提示 ────────────────────────────────────
    printf("\n===== 用 Nsight Compute 验证 =====\n");
    printf("ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \\\n");
    printf("             --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \\\n");
    printf("             ./coalescing\n");
    printf("\n");
    printf("sectors / requests = 平均每个 request 触发的 sector 数\n");
    printf("  Coalesced   → ~4  sectors per warp request\n");
    printf("  Unaligned   → ~5  sectors per warp request\n");
    printf("  Strided(x8) → ~32 sectors per warp request\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
