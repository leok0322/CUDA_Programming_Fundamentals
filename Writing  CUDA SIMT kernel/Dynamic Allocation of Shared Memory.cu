#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 64

#define CUDA_CHECK(expr_to_check) do {                    \
    cudaError_t result  = expr_to_check;                  \
    if(result != cudaSuccess)                             \
    {                                                     \
        fprintf(stderr,                                   \
                "CUDA Runtime Error: %s:%i:%d = %s\n",   \
                __FILE__,                                 \
                __LINE__,                                 \
                result,                                   \
                cudaGetErrorString(result));              \
    }                                                     \
} while(0)


// Dynamic shared memory: single extern __shared__ partitioned into 3 arrays
// equivalent of:
//   short array0[128];
//   float array1[64];
//   int   array2[256];
__global__ void dynamicSharedMemKernel(short *out0, float *out1, int *out2)
{
    // Single dynamic shared memory base pointer
    extern __shared__ float array[];

    // Partition manually via pointer arithmetic
    short *array0 = (short *)array;
    float *array1 = (float *)&array0[128];
    int   *array2 =   (int *)&array1[64];

    int tid = threadIdx.x;

    // Initialize each shared memory partition
    if (tid < 128) array0[tid] = (short)(tid * 2);
    if (tid < 64)  array1[tid] = (float)(tid) * 1.5f;
    if (tid < 256) array2[tid] = tid * tid;

    // array0、array1、array2 是三个独立的指针，各自有自己的索引空间：                                                                                               
   
    // // 用户修改版 —— 错误                                                                                                                                         
    // if (tid < 128+64 && tid >= 128)  array1[tid] = ...;
    // // tid 范围是 128~191，即访问 array1[128] ~ array1[191]
    // // 但 array1 只有 64 个元素！越界了

    // // 原版 —— 正确
    // if (tid < 64) array1[tid] = (float)(tid) * 1.5f;
    // // tid 范围是 0~63，即访问 array1[0] ~ array1[63]，恰好 64 个元素

    // 指针算术已经负责了内存的物理分区：

    // shared memory 物理布局：
    // [array0: 0~255 bytes][array1: 256~511 bytes][array2: 512~1535 bytes]
    //     ↑ array0[0]          ↑ array1[0]             ↑ array2[0]

    // Warp 内：所有线程 lockstep 执行同一条指令，if 语句是顺序走下来的，不满足条件的线程被 predicate off（空转）。
    // 256 线程 = 8 个 warp（每个 32 线程）
    // 每个 warp 内部是 SIMT lockstep 执行，三条 if 语句确实是按顺序一条条执行：
    // 每个 warp 的执行序列：
    // 指令1: if (tid < 128) → 写 array0
    // 指令2: if (tid < 64)  → 写 array1  
    // 指令3: if (tid < 256) → 写 array2
    // predicated off 的线程不是"不执行这条指令"，而是执行了但结果被丢弃，仍然消耗一个指令周期。

    // Warp 间：由 Scheduler 并发调度，不同 warp 可以处于不同的执行进度，某个 warp 在等 shared memory 延迟时，其他 warp 的指令可以插进来执行。

    // 并行维度
    // 维度1：Warp 内 32 线程
    // 真正的 SIMT 硬件并行，同一个周期执行同一条指令，这是"物理上同时"。
    // 维度2：多个 Warp 之间
    // 要分两层：

    // 真正同时 issue：每周期 4 个 Scheduler 各发射 1 条指令，是 4 个 warp 的物理并行
    // 并发驻留：几十个 warp 同时在 SM 上，但大多数在等待，靠 Scheduler 快速切换来隐藏延迟，这是"并发"而非严格"并行"

    // 还有维度3：多个 SM 之间
    // 多个 SM 完全独立并行执行各自的 block，这一层经常被忽略。
    // 完整的并行层次：
    // 多 SM          → block 级并行
    //     ↓
    // SM 内多 Warp   → 并发调度（4个真并行 + latency hiding）
    //     ↓
    // Warp 内 32线程 → SIMT 物理并行
    // 程序员看不到也控制不了具体哪个 block 跑在哪个 SM 上，CUDA 不提供这个信息。唯一能做的是用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询理论上每 SM 能驻留几个 block，用来评估 occupancy。



    __syncthreads();  // block内的线程同步

    // Write results back to global memory
    if (tid < 128) out0[tid] = array0[tid];
    if (tid < 64)  out1[tid] = array1[tid];
    if (tid < 256) out2[tid] = array2[tid];
}

int main(int argc, char **argv)
{
    // ── 1. 查询设备物理限制 ──────────────────────────────────────────────────
    int device = 0;
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("=== Device: %s ===\n", prop.name);
    // sharedMemPerBlock:         默认每 block 可用共享内存（通常 48 KB）
    // sharedMemPerMultiprocessor: 每个 SM 的共享内存物理总量
    // sharedMemPerBlockOptin:    opt-in 后单 block 最大可申请量（可超过 48 KB）
    printf("sharedMemPerBlock        : %zu bytes (%.0f KB)\n",
           prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
    printf("sharedMemPerMultiprocessor: %zu bytes (%.0f KB)\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
    printf("sharedMemPerBlockOptin   : %zu bytes (%.0f KB)\n",
           prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0);

    // ── 2. cudaFuncSetCacheConfig：L1 / Shared Mem 分配偏好 ─────────────────
    // 对于共享内存密集型 kernel，倾向把更多片上 SRAM 给 shared mem
    // cudaFuncCachePreferShared : 更多给 shared mem（L1 变小）
    // cudaFuncCachePreferL1     : 更多给 L1 cache
    // cudaFuncCachePreferEqual  : 各半
    // cudaFuncCachePreferNone   : 驱动自行决定
    // 注意：Volta+ 架构 L1 与 shared mem 已统一为一块 SRAM，
    //       此 API 仍有效但实际比例由硬件架构决定。
    CUDA_CHECK(cudaFuncSetCacheConfig(
        (const void *)dynamicSharedMemKernel,
        cudaFuncCachePreferShared));
    printf("\ncudaFuncSetCacheConfig -> cudaFuncCachePreferShared\n");

    // ── 3. cudaFuncSetAttribute：显式放开动态共享内存上限 ───────────────────
    // 默认每 block 动态共享内存上限 = 48 KB。
    // 若需要更大（如 96 KB），必须用此 API opt-in，
    // 否则 launch 时 sharedMemBytes > 48 KB 会报错。
    // 这里演示：将上限设为设备支持的最大值（sharedMemPerBlockOptin）。

    // 为什么要有这道软件上限？
    // 安全保护机制。大共享内存会降低 SM 上同时驻留的 block 数（occupancy 下降），进而影响延迟隐藏能力。CUDA 不想让开发者无意中用了大共享内存，所以要求显式
    // opt-in，迫使开发者意识到这个 trade-off。
    // 大共享内存会降低 SM 上同时驻留的 block 数（occupancy 下降）的原因是kernel中设置的是单个block的shared mem，如果过大，单个sm驻留的block就十分有限，


    // 因为 sharedMemBytes 只有：                                                                                                                                    
    // 128 * sizeof(short) + 64 * sizeof(float) + 256 * sizeof(int)
    // = 256 + 256 + 1024 = 1536 bytes ≈ 1.5 KB                                                                                                                      
                    
    // 远低于默认 48 KB 上限，所以不调用 cudaFuncSetAttribute 也不会报错。

    // ---
    // cudaFuncSetAttribute 真正有意义的场景：

    // // 需要大共享内存，比如 64 KB
    // size_t sharedMemBytes = 64 * 1024;

    // // 不设置 → launch 报错，即使硬件支持
    // dynamicSharedMemKernel<<<1, 256, sharedMemBytes>>>(...);  // ❌

    // // 设置后 → launch 成功
    // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024);
    // dynamicSharedMemKernel<<<1, 256, sharedMemBytes>>>(...);  // ✓
    // ── 3a. Carveout：先确保 SM 物理空间足够 ────────────────────────────────
    // 设置 SM SRAM 中划给 shared mem 的比例，剩余给 L1 cache
    // 必须在 MaxDynamicSharedMemorySize 之前设置，先保证物理空间，再放开软件上限
    // cudaSharedmemCarveoutMaxShared : 最大化 shared mem（对应 cudaFuncCachePreferShared）
    // cudaSharedmemCarveoutMaxL1     : 最大化 L1
    // cudaSharedmemCarveoutDefault   : 驱动自行决定
    // 也可以直接传 0~100 的整数表示百分比，如 75 表示 75% 给 shared mem
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void *)dynamicSharedMemKernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    printf("cudaFuncSetAttribute -> PreferredSharedMemoryCarveout = MaxShared\n");

    // ── 3b. MaxDynamicSharedMemorySize：再放开软件上限 ──────────────────────
    // 单block默认软件上限是 48 KB。如果你 launch 时传 sharedMemBytes = 64 KB，即使硬件支持，也会报错。
    // 两道关卡：Carveout 管 SM 级别的物理切割，MaxDynamicSharedMemorySize 管 block 级别的申请上限
    size_t maxDynShared = prop.sharedMemPerBlockOptin;
    CUDA_CHECK(cudaFuncSetAttribute(
        (const void *)dynamicSharedMemKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,// 默认软件上限是 48 KB。如果你 launch 时传 sharedMemBytes = 64 KB，即使硬件支持，也会报错：                         
        maxDynShared)); // 可以去掉(int)，隐式转换   
    printf("cudaFuncSetAttribute -> MaxDynamicSharedMemorySize = %zu bytes (%.0f KB)\n\n",
           maxDynShared, maxDynShared / 1024.0);

    // Host output arrays
    short h_out0[128] {0};
    float h_out1[64] {0};
    int   h_out2[256] {0};

    // Device output arrays
    short *d_out0 {};
    float *d_out1 = nullptr;
    int   *d_out2 = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_out0, 128 * sizeof(short)));  // 分配HBW内存
    CUDA_CHECK(cudaMalloc((void **)&d_out1,  64 * sizeof(float))); // 分配HBW内存
    CUDA_CHECK(cudaMalloc((void **)&d_out2, 256 * sizeof(int))); // 分配HBW内存

    // Calculate shared memory size:
    // short[128] + float[64] + int[256]
    size_t sharedMemBytes = 128 * sizeof(short)
                          +  64 * sizeof(float)
                          + 256 * sizeof(int);

    // Launch with 256 threads; 3rd chevron arg = dynamic shared memory size
    // 问题在于 kernel launch 的 <<<>>> 语法不能直接包裹在宏里，因为它是特殊的编译器语法，无法被宏当作表达式处理。      
    // Kernel Launch：不返回 cudaError_t！ 
    // CUDA Runtime 为每个 CPU 线程维护一个独立的错误状态：

    cudaGetLastError();   // 第1次：丢弃 pre-existing error，清空槽位
    dynamicSharedMemKernel<<<1, 256, sharedMemBytes>>>(d_out0, d_out1, d_out2);
    // dynamicSharedMemKernel<<<gridDim, blockDim, sharedMemBytes>>>(...)                                                                                                                    
    // //                                            ↑
    // //                               每个 block 分配 sharedMemBytes 字节                                                                                                                  
                    
    // 关键点：

    // - 每个 block 独立拥有自己的共享内存，互不可见
    // - 总共享内存使用量 = sharedMemBytes × gridDim（block 数量）
    // - 但每个 SM 上的 block 共享该 SM 的共享内存池，受硬件限制



    // 第一步：检查 Launch 本身的错误或者launch之前的错误
    CUDA_CHECK(cudaGetLastError());      // 检查 launch 参数错误（同步前），同时清除错误状态
    // 第二步：等待执行完成，检查执行错误
    // 所以 CUDA_CHECK(cudaDeviceSynchronize()) 执行后：
    // - 返回值被检查了 ✓
    // - 但 CUDA 内部的错误状态槽位没有被清除 ✗
    CUDA_CHECK(cudaDeviceSynchronize()); // 等待 kernel 完成，并捕获运行时错误。这个残留的错误就成了下一次 kernel launch 前的 "pre-existing error"，被第一个裸 cudaGetLastError() 拿走并丢弃。


    // CUDA 的错误状态是单个槽位，不是队列，只存一个错误。所以：                                                                                                     
                                                                                                                                                                
    // pre-existing error → 状态 = Error A                                                                                                                           
    // kernel launch 失败 → 状态 = Error B（覆盖 A，或 A 保持，取决于实现）
    // cudaGetLastError() → 返回并清除当前状态（只有一个）

    // 一次 cudaGetLastError() 只能清除当前槽位里的那一个错误，另一个要么已被覆盖丢失，要么根本没机会被记录。

    // 所以正确模式是 launch 前先主动清除：

    // cudaGetLastError();   // 第1次：丢弃 pre-existing error，清空槽位

    // dynamicSharedMemKernel<<<1, 256, sharedMemBytes>>>(d_out0, d_out1, d_out2);

    // CUDA_CHECK(cudaGetLastError());       // 第2次：槽位干净，拿到的一定是 launch 的错误
    // CUDA_CHECK(cudaDeviceSynchronize());  // 拿到 kernel 运行时错误

    // 两次 cudaGetLastError() 各清除各自时刻的那一个错误，不存在"一次清除两个"的情况。

    CUDA_CHECK(cudaMemcpy(h_out0, d_out0, 128 * sizeof(short), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out1, d_out1,  64 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out2, d_out2, 256 * sizeof(int),   cudaMemcpyDeviceToHost));

    // Print first 8 elements of each array
    printf("array0 (short, tid*2):\n");
    for (int i = 0; i < 8; i++) printf("  [%d] = %d\n", i, h_out0[i]);

    printf("array1 (float, tid*1.5):\n");
    for (int i = 0; i < 8; i++) printf("  [%d] = %.1f\n", i, h_out1[i]);

    printf("array2 (int, tid*tid):\n");
    for (int i = 0; i < 8; i++) printf("  [%d] = %d\n", i, h_out2[i]);

    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}
