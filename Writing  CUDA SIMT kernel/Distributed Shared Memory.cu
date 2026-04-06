#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// ════════════════════════════════════════════════════════════════════════════
// Distributed Shared Memory — 直方图（Histogram）
// ════════════════════════════════════════════════════════════════════════════
//
// 问题：统计输入数组中每个值落在哪个 bin，传统做法需要对 global mem 大量 atomicAdd
//
// 分布式 smem 方案：
//   cluster 内每个 block 负责一段 bin（bins_per_block 个）
//   线程计算出目标 bin → 找到对应 block 的 smem → atomicAdd 到 smem
//   smem 上的 atomicAdd 远快于 global mem（延迟低 ~10x）
//   cluster 内汇总完毕后，每个 block 把自己的 smem 写回 global mem（只需少量 atomicAdd）
//
// bin 到 block 的映射：
//   dst_block_rank = binid / bins_per_block   → 哪个 block 负责这个 bin
//   dst_offset     = binid % bins_per_block   → 在该 block smem 的第几个槽位
//
// 内存层次：
//   输入数据（global mem）
//       ↓ 每个线程读取
//   分布式 smem（cluster 内各 block 的 smem，通过 map_shared_rank 跨 block 写入）
//       ↓ cluster.sync() 后
//   global mem bins（每个 block 把自己负责的 smem 段写回）
//
// 需要：CUDA 12+，SM 9.0+（Hopper）
// 编译：nvcc -arch=sm_90a "Distributed Shared Memory.cu" -o dsm

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// ── Kernel：分布式 smem 直方图 ───────────────────────────────────────────────
// bins           : global mem 中的直方图结果数组，长度 nbins
// nbins          : 总 bin 数，必须等于 cluster_size * bins_per_block
// bins_per_block : 每个 block 负责的 bin 数 = nbins / cluster_size
// input          : 输入数据数组
// array_size     : 输入数据长度


// ● __restrict__ 是告诉编译器：这个指针是访问该内存的唯一途径，不会与其他指针发生别名（alias）。                                                                   
                                                                                                                                                                 
//   ---                                                                                                                                                            
//   指针别名问题：                                                                                                                                                 
                                                                                                                                                                 
//   void foo(int* a, int* b) {                                                                                                                                     
//       *a = 1;
//       *b = 2;                                                                                                                                                    
//       // 编译器不确定 a 和 b 是否指向同一地址
//       // 必须每次都从内存读写，无法优化
//   }

//   如果 a == b（别名），*a = 1 之后 *b 也变了，编译器不敢假设它们独立。

//   ---
//   加 __restrict__ 后：

//   void foo(int* __restrict__ a, int* __restrict__ b) {
//       *a = 1;
//       *b = 2;
//       // 编译器确定 a、b 不重叠
//       // 可以重排指令、放入寄存器缓存、向量化等优化
//   }

//   ---
//   在 CUDA kernel 中的意义：

//   __global__ void kernel(int* output, const int* __restrict__ input)

//   - 告诉 GPU 编译器 input 不会和 output 或其他指针重叠
//   - 编译器可以把 input 的读取放入只读缓存（Read-Only Cache / Texture Cache），带宽更高
//   - 生成更激进的指令调度和向量化代码

__global__ void clusterHist_kernel(int *bins, const int nbins,
                                   const int bins_per_block,
                                   const int *__restrict__ input,
                                   size_t array_size)
{
    // 动态 smem：每个 block 分配 bins_per_block 个 int
    // 物理上每个 block 只有这一段 smem，但通过 map_shared_rank 可访问 cluster 内其他 block 的 smem
    // 分布式 smem 总容量 = cluster_size * bins_per_block * sizeof(int)
    extern __shared__ int smem[];

    namespace cg = cooperative_groups;

    // this_grid().thread_rank()：当前线程在整个 grid 中的全局线程 id
    // 用于 grid-stride 循环，让所有线程均匀处理输入数据
    int tid = cg::this_grid().thread_rank();

    cg::cluster_group cluster     = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();  // 本 block 在 cluster 内的局部编号
    int cluster_size              = cluster.dim_blocks().x; // cluster 内 block 数量

    // 初始化本 block 的 smem 为 0（本 block 负责的那段 bin）
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
        smem[i] = 0;

    // cluster.sync()：等 cluster 内所有 block 的 smem 都初始化为 0 后再开始写入
    // 同时保证所有 block 已开始执行并发驻留（distributed smem 访问的前提）
    cluster.sync();

    // grid-stride 循环：每个线程处理多个输入元素
    // 步长 = blockDim.x * gridDim.x（grid 内总线程数），保证所有元素都被处理
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        // 边界处理：值 < 0 归入 bin 0，值 >= nbins 归入最后一个 bin
        int binid = ldata;
        if (ldata < 0)       binid = 0;
        else if (ldata >= nbins) binid = nbins - 1;

        // 计算目标 block rank 和 smem 槽位
        // 每个 bin 固定归属某一个 block 的 smem
        int dst_block_rank = binid / bins_per_block;  // 哪个 block 负责
        int dst_offset     = binid % bins_per_block;  // 该 block smem 的第几个槽位

        // map_shared_rank：获取目标 block 的 smem 指针
        // 若 dst_block_rank == clusterBlockRank，等价于直接访问本地 smem
        // 若不同，则走 SM-to-SM NVLink Fabric，无需经过 global mem
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        // atomicAdd 到目标 block 的 smem
        // 真实竞争：多个线程可能映射到同一个 binid → 同一个槽位 → 必须原子操作
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // cluster.sync()：确保所有线程的 distributed smem atomicAdd 全部完成
    // 必须在任何 block 退出前同步，否则其他 block 可能还在访问已退出 block 的 smem
    cluster.sync();

    // 把本 block 的 smem 结果写回 global mem
    // 每个 block 只写自己负责的那段 bins（bins_per_block 个）
    // lbins 指向 global mem bins 数组中本 block 负责的起始位置
    int *lbins = bins + clusterBlockRank * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        // 多个 cluster 的同一个 block_rank 可能都写同一段 global bins → atomicAdd
        atomicAdd(&lbins[i], smem[i]);
    }
}

int main()
{
    // ── 1. 检查设备（SM 9.0+ 才支持 distributed shared memory）────────────
    int device = 0;
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  SM %d.%d\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("Distributed shared memory requires SM 9.0+ (Hopper). Skipping.\n");
        return 0;
    }

    // ── 2. 参数设置 ─────────────────
    const int    threads_per_block = 256;
    const size_t array_size        = 1 << 20;  // 1M 个输入元素
    const int    nbins             = 256;       // 总 bin 数


    // 1 << 1  = 2                                                                                                                                                    
    // 1 << 2  = 4                                                                                                                                                    
    // 1 << 3  = 8                                                                                                                                                    
    // 1 << 10 = 1024        (1 KB)                                                                                                                                   
    // 1 << 20 = 1048576     (1 MB，即 1M 个元素)
    // 1 << 30 = 1073741824  (1 GB)                                                                                                                                   
                    
    // 原理： 二进制的 1 向左移 N 位，等价于乘以 2ᴺ。

    // 1        = 0000...0001
    // 1 << 20  = 0001 0000 0000 0000 0000 0000 0000

    // cluster_size 决定 bin 如何在 cluster 内分配
    // cluster_size == 1：退化为普通 shared mem histogram，无分布式 smem
    // cluster_size == 2：每个 block 负责 128 个 bin，分布式 smem 总量 = 2 * 128 * 4 = 1 KB
    const int cluster_size   = 2;
    const int bins_per_block = nbins / cluster_size;  // 每个 block 负责 128 个 bin

    // 这里不需要向上取整，反而需要保证整除。                                                                                                            
    // ---                                                                                                                                                            
    // 原因：bin 到 block 的映射依赖精确整除                                                                                                                     
    // int dst_block_rank = binid / bins_per_block;  // 必须整除才能正确映射                                                                                          
    // int dst_offset     = binid % bins_per_block;
                                                 

    // 第一步：向上取整到 threads_per_block，防止漏掉最后一批数据
    // array_size=1000, threads_per_block=256 → 向下取整只处理 768 个，漏 232 个
    // kernel 内用 grid-stride 循环 + i < array_size 边界判断，多出的线程不会越界
    int grid_size = (int)((array_size + threads_per_block - 1) / threads_per_block);
    // 第二步：向上对齐到 cluster_size 的整数倍（gridDim 必须是 cluster_size 整数倍）
    grid_size = ((grid_size + cluster_size - 1) / cluster_size) * cluster_size;

    printf("array_size     : %zu\n", array_size);
    printf("nbins          : %d\n",  nbins);
    printf("cluster_size   : %d\n",  cluster_size);
    printf("bins_per_block : %d\n",  bins_per_block);
    printf("grid_size      : %d blocks\n\n", grid_size);

    // ── 3. 数据准备 ────────────────────────────────────────────────────────
    int* h_input = new int[array_size];
    int* h_bins  = new int[nbins] {};

    // 生成测试数据：均匀分布在 [0, nbins) 范围内
    for (size_t i = 0; i < array_size; i++)
        h_input[i] = (int)(i % nbins);

    int *d_input, *d_bins;
    CUDA_CHECK(cudaMalloc(&d_input, array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bins,  nbins      * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, array_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bins, 0, nbins * sizeof(int)));

    // ── 4. cluster 启动配置 ────────────────────────────────────────────────
    // 动态 smem 大小：每个 block 分配 bins_per_block 个 int
    // 分布式 smem 总量 = cluster_size * bins_per_block * sizeof(int)
    // 但 dynamicSmemBytes 填的是单个 block 的大小
    size_t dynamicSmemBytes = bins_per_block * sizeof(int);

    // 两道关卡，顺序：先保证物理空间，再放开软件上限
    // 本例 dynamicSmemBytes = 512 bytes，远低于 48 KB，两道关卡天然满足，加不加不影响运行
    // 但作为完整写法，显式设置以应对 dynamicSmemBytes 较大的场景

    // 关卡1：Carveout —— 确保 SM 物理 SRAM 中划给 shared mem 的空间足够
    // carveout 是 SM 级别的比例，dynamicSmemBytes 是单个 block 的大小，量纲不同
    // SM 上同时驻留多个 block，需要按实际最大驻留量计算，而不是单个 block 的大小
    //
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor：
    //   查询该 kernel 在此 SM 上理论最多能同时驻留几个 block
    //   综合考虑寄存器、shared mem、block 数量等硬件限制
    int maxBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM, clusterHist_kernel, threads_per_block, dynamicSmemBytes));
    printf("maxBlocksPerSM : %d\n", maxBlocksPerSM);

    // SM 上所有驻留 block 共享物理 SRAM，carveout 需要覆盖所有 block 的 smem 总需求
    size_t totalSmemNeeded = dynamicSmemBytes * maxBlocksPerSM;

    // carveout 单位是百分比（0~100），先乘 100 再除，避免整数除法精度丢失
    // 向上取整保证划出空间 >= totalSmemNeeded，剩余留给 L1 cache
    // 与 100 取小者，防止 totalSmemNeeded > sharedMemPerMultiprocessor 时超出范围
    int carveout = (int)((totalSmemNeeded * 100 + prop.sharedMemPerMultiprocessor - 1)
                         / prop.sharedMemPerMultiprocessor);
    carveout = carveout < 100 ? carveout : 100;
    printf("carveout       : %d%%\n\n", carveout);
    CUDA_CHECK(cudaFuncSetAttribute(
        (void *)clusterHist_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        carveout));  // 按实际驻留需求划分，剩余给 L1 cache

    // 关卡2：MaxDynamicSharedMemorySize —— 放开单 block 动态 smem 的软件上限（默认 48 KB）
    // 若 dynamicSmemBytes > 48 KB 且不设置此项，launch 时会报错
    CUDA_CHECK(cudaFuncSetAttribute(
        (void *)clusterHist_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamicSmemBytes));

    cudaLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim         = grid_size;
    config.blockDim        = threads_per_block;
    config.dynamicSmemBytes = dynamicSmemBytes;
    config.numAttrs        = 1;
    config.attrs           = attribute;

    // ── 5. 启动 kernel ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaLaunchKernelEx(&config, clusterHist_kernel,
                                  d_bins, nbins, bins_per_block,
                                  d_input, array_size));
    CUDA_CHECK(cudaGetLastError());                              
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 6. 验证结果 ────────────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));

    // 均匀分布：每个 bin 应该有 array_size / nbins 个元素
    int expected = (int)(array_size / nbins);
    printf("=== Histogram result (expected each bin = %d) ===\n", expected);
    bool correct = true;
    for (int i = 0; i < nbins; i++) {
        if (h_bins[i] != expected) {
            printf("  bin[%d] = %d (MISMATCH)\n", i, h_bins[i]);
            correct = false;
        }
    }
    // 打印前 8 个 bin
    for (int i = 0; i < 8; i++)
        printf("  bin[%3d] = %d\n", i, h_bins[i]);
    printf("  ...\n");
    printf("Result: %s\n", correct ? "CORRECT" : "WRONG");

    delete[] h_input;
    delete[] h_bins;
    cudaFree(d_input);
    cudaFree(d_bins);
    return 0;
}
