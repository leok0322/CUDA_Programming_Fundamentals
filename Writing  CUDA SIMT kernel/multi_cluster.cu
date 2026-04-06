#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// 多 cluster 示例
// 需要 SM 9.0+（Hopper），编译：nvcc -arch=sm_90a multi_cluster.cu -o mc
//
// 核心概念：
// - 一个 grid 可以包含多个 cluster
// - cluster.sync() 只同步本 cluster 内的 block，不跨 cluster
// - 每个 cluster 独立并行执行，互不干扰
// - gridDim 必须是 CLUSTER_SIZE 的整数倍

namespace cg = cooperative_groups;

#define BLOCK_SIZE   128
#define CLUSTER_SIZE 4      // 每个 cluster 含 4 个 block
#define NUM_CLUSTERS 3      // 3 个 cluster
#define GRID_SIZE    (CLUSTER_SIZE * NUM_CLUSTERS)  // 12 个 block


// 原则：cluster 内聚合，cluster 间独立                                                                                                                       
                                                                                                                                                            
// 总数据                                                                                                                                                     
// ├── cluster 0 处理分片 0  ──→ 局部结果 0  ┐
// ├── cluster 1 处理分片 1  ──→ 局部结果 1  ├─→ 全局归约（global mem）                                                                                       
// └── cluster 2 处理分片 2  ──→ 局部结果 2  ┘

// - cluster 内：用分布式 smem 直接通信，快
// - cluster 间：必须经过 global mem，慢，但可以并行

// ---
// 具体设计步骤：

// 第一步：确定 cluster 内能覆盖多少数据

// // cluster 内能直接共享的数据量
// size_t cluster_data = CLUSTER_SIZE * BLOCK_SIZE * sizeof(int);
// // 4 * 128 * 4 = 2048 bytes
// // 这是 cluster 内分布式 smem 的总容量，决定单次聚合上限

// 第二步：总数据按 cluster 切片，每个 cluster 独立处理

// // 总数据 N，切成 NUM_CLUSTERS 份
// // 每个 cluster 处理 N / NUM_CLUSTERS 的数据
// int chunk = N / NUM_CLUSTERS;
// // cluster c 处理 input[c * chunk .. (c+1) * chunk - 1]

// 第三步：cluster 内产生局部结果，写回 global mem

// // cluster 内归约完成后，每个 cluster 输出一个局部结果到 global mem
// __shared__ int partial;
// if (threadIdx.x == 0 && cluster.block_rank() == 0)
//     global_partial[cluster_id] = partial;  // 写回 global mem

// 第四步：第二个 kernel 或 atomicAdd 做跨 cluster 的全局归约

// // 方式1：再启动一个小 kernel 归约 NUM_CLUSTERS 个局部结果
// // 方式2：直接 atomicAdd 到全局计数器
// atomicAdd(global_result, partial);

// ---
// 两级归约完整示意：

// 输入：N = 1536 个元素

// Kernel 1（cluster 内归约）：
// cluster 0 → block 0~3 → smem 共享 → partial[0] = 512
// cluster 1 → block 4~7 → smem 共享 → partial[1] = 512
// cluster 2 → block 8~11→ smem 共享 → partial[2] = 512
//                                         ↓ 写回 global mem

// Kernel 2（跨 cluster 归约）：
// partial[0] + partial[1] + partial[2] = 1536  ← 最终结果

// ---
// 如何平衡 CLUSTER_SIZE 和 NUM_CLUSTERS：

// ┌─────────────────────┬─────────────────────────────────────┬───────────────────────────────────────┐
// │        倾向         │              调整方式               │                 代价                  │
// ├─────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
// │ 更多 cluster 内共享 │ 增大 CLUSTER_SIZE（2→4→8）          │ 每 SM 驻留 cluster 数减少，并行度下降 │
// ├─────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
// │ 更高并行度          │ 增大 NUM_CLUSTERS                   │ cluster 间只能走 global mem，通信变慢 │
// ├─────────────────────┼─────────────────────────────────────┼───────────────────────────────────────┤
// │ 最优点              │ CLUSTER_SIZE=4，NUM_CLUSTERS 尽量多 │ 硬件决定，通常 CLUSTER_SIZE=4 是甜点  │
// └─────────────────────┴─────────────────────────────────────┴───────────────────────────────────────┘

// 经验法则：CLUSTER_SIZE 由聚合粒度决定，NUM_CLUSTERS 由数据规模决定，两者独立调整。


#define CUDA_CHECK(expr) do {                                        \
    cudaError_t _e = (expr);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ── Kernel：每个 cluster 内做分布式 shared mem 归约求和 ────
//
// 数据布局：
//   cluster 0 → block 0,1,2,3   处理 input[0   .. 4*128-1]
//   cluster 1 → block 4,5,6,7   处理 input[512 .. 8*128-1]
//   cluster 2 → block 8,9,10,11 处理 input[1024..12*128-1]
//
// 每个 cluster 独立完成：
//   1. 各 block 写入自己的 smem
//   2. cluster.sync() 同步本 cluster 所有 block
//   3. 每个 block 读取 cluster 内所有 block 的 smem，求局部和
//   4. 结果写回 global mem
//
// cluster.sync() 只同步本 cluster，不影响其他 cluster
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
multiClusterReduceKernel(const int* input, int* output)
{
    cg::cluster_group cluster = cg::this_cluster();

    // 每个 block 自己的 shared mem
    __shared__ int local_smem[BLOCK_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    local_smem[threadIdx.x] = input[gid];

    // cluster 内同步：等本 cluster 所有 block 写完各自 smem
    // 注意：只同步本 cluster（4个block），不影响其他 cluster
    cluster.sync();

    // 每个线程对本 cluster 内所有 block 的对应槽位求和
    // cluster 0 的 thread 0 累加 block0[0] + block1[0] + block2[0] + block3[0]
    int sum = 0;
    for (uint32_t rank = 0; rank < cluster.num_blocks(); rank++) {
        int* peer_smem = cluster.map_shared_rank(local_smem, rank);
        sum += peer_smem[threadIdx.x];
    }

    output[gid] = sum;
}

// ── Kernel：展示 cluster_id 与 block_rank 的关系 ──────
// 每个线程把自己所属的 cluster 编号和 block_rank 写出，方便观察多 cluster 布局
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
clusterInfoKernel(int* cluster_id_out, int* block_rank_out)
{
    cg::cluster_group cluster = cg::this_cluster();

    // cluster_id = blockIdx.x / CLUSTER_SIZE（当前 cluster 的编号）
    // block_rank = 本 block 在 cluster 内的局部编号（0~3）
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 只让每个 block 的 thread 0 写，避免重复
    // blockIdx.x 是全局唯一的，block_rank 是 cluster 内局部的，两者关系：
    // blockIdx.x = cluster_id * CLUSTER_SIZE + block_rank
    if (threadIdx.x == 0) {
        cluster_id_out[blockIdx.x]  = blockIdx.x / CLUSTER_SIZE;  //   blockIdx.x / CLUSTER_SIZE  → 第几个 cluster（cluster 编号）
        block_rank_out[blockIdx.x]  = (int)cluster.block_rank();  //   cluster.block_rank()        → 在本 cluster 内是第几个 block（局部编号）                                
    }
}

int main()
{
    // ── 1. 检查 SM 版本 ─────
    int device = 0;
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  SM %d.%d\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("Multi-cluster requires SM 9.0+ (Hopper). Skipping.\n");
        return 0;
    }

    // ── 2. 数据准备 ───────
    const int N = GRID_SIZE * BLOCK_SIZE;  // 12 * 128 = 1536

    int* h_input      = new int[N] {};
    int* h_output     = new int[N] {};
    int* h_cluster_id = new int[GRID_SIZE] {};
    int* h_block_rank = new int[GRID_SIZE] {};

    for (int i = 0; i < N; i++) h_input[i] = 1;  // 全 1，便于验证：sum = CLUSTER_SIZE * 1

    int *d_input, *d_output, *d_cluster_id, *d_block_rank;
    CUDA_CHECK(cudaMalloc(&d_input,      N          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output,     N          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_id, GRID_SIZE  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_rank, GRID_SIZE  * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // ── 3. cluster 启动配置 ───────
    cudaLaunchAttribute attr[1];
    attr[0].id               = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER_SIZE;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.blockDim        = BLOCK_SIZE;
    cfg.dynamicSmemBytes = 0;
    cfg.attrs           = attr;
    cfg.numAttrs        = 1;

    // ── 4. Kernel 1：clusterInfoKernel ────
    printf("=== Kernel 1: cluster layout (%d clusters x %d blocks) ===\n",
           NUM_CLUSTERS, CLUSTER_SIZE);
    cfg.gridDim = GRID_SIZE;
    cudaGetLastError();
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, clusterInfoKernel, d_cluster_id, d_block_rank));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_cluster_id, d_cluster_id, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_rank, d_block_rank, GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    for (int b = 0; b < GRID_SIZE; b++) {
        printf("  blockIdx %2d → cluster_id=%d  block_rank=%d\n",
               b, h_cluster_id[b], h_block_rank[b]);
    }

    // ── 5. Kernel 2：multiClusterReduceKernel ────
    // 每个 cluster 内 4 个 block，每个槽位 sum = 4 * 1 = 4
    printf("\n=== Kernel 2: per-cluster reduce (each slot sum = %d) ===\n", CLUSTER_SIZE);
    cfg.gridDim = GRID_SIZE;
    cudaGetLastError();
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, multiClusterReduceKernel, d_input, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost));

    // 验证每个 cluster 的输出，取每个 block 的前 4 个元素
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        printf("  cluster %d:\n", c);
        for (int b = 0; b < CLUSTER_SIZE; b++) {
            int block_id = c * CLUSTER_SIZE + b;
            printf("    block %2d first 4 outputs: ", block_id);
            for (int i = 0; i < 4; i++)
                printf("%d ", h_output[block_id * BLOCK_SIZE + i]);
            printf("\n");
        }
    }

//   ┌──────┬─────────────────────────────┐
//   │ 格式 │            含义             │
//   ├──────┼─────────────────────────────┤
//   │ %2d  │ 最小宽度2，右对齐，空格补齐 │
//   ├──────┼─────────────────────────────┤
//   │ %-2d │ 最小宽度2，左对齐           │
//   ├──────┼─────────────────────────────┤
//   │ %02d │ 最小宽度2，右对齐，0补齐    │
//   └──────┴─────────────────────────────┘

//   printf("%02d", 1);  // 输出 "01"
//   printf("%02d", 10); // 输出 "10"

    delete[] h_input;
    delete[] h_output;
    delete[] h_cluster_id;
    delete[] h_block_rank;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_cluster_id);
    cudaFree(d_block_rank);
    return 0;
}
