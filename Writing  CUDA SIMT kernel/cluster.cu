#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

// cluster_group / cluster.sync() / map_shared_rank 需要：
// - CUDA 12+
// - SM 9.0+（Hopper，如 H100）
// - 编译：nvcc -arch=sm_90a cluster.cu -o cluster
// cluster.sync() 报错根本原因：编译时未指定 -arch=sm_90a，
// 导致 PTX 不包含 cluster barrier 指令

namespace cg = cooperative_groups;

#define BLOCK_SIZE   128
#define CLUSTER_SIZE 4     // cluster 内的 block 数，必须是 2 的幂且 <= 8


// <= 8 —— 硬件物理限制                                                                                                           
// 一个 cluster 内的所有 block 必须同时驻留在相邻的 SM 上，依靠 SM 间的低延迟互连（NVLink Fabric）直接访问彼此的 shared mem。
// Hopper（H100）每个 GPC（Graphics Processing Cluster）有 8 个 SM，cluster 不能跨 GPC，所以上限是 8。
// GPC（8个SM）
// ├── SM0  ─┐
// ├── SM1   │  cluster 最多覆盖这 8 个 SM
// ├── SM2   │  跨 GPC 没有直接互连
// ├── SM3   │
// ├── SM4   │
// ├── SM5   │
// ├── SM6   │
// └── SM7  ─┘

// ---
// 2 的幂 —— 路由/寻址硬件设计
// SM 间互连的硬件路由使用二进制位掩码寻址，cluster 内 block_rank 的计算、barrier 的 arrive/wait 计数都基于位运算。
// 非 2 的幂会导致路由逻辑出现"空洞"（如 6 个 SM 的 barrier 计数器无法用简单位运算实现），硬件不支持。
// 合法值：1, 2, 4, 8

#define CUDA_CHECK(expr) do {                                        \
    cudaError_t _e = (expr);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ── Kernel 1: cluster.sync() + map_shared_rank 读邻居 smem ──────────────────
// 每个 block 把自己的 smem 写好后，同步整个 cluster，
// 再读取下一个 block（neighborRank）的 smem 数据输出。
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
distributedSmemKernel(int* input, int* output)
{
    cg::cluster_group cluster = cg::this_cluster();

    __shared__ int local_smem[BLOCK_SIZE];  // 每个block有自己的shared mem




    // 每个线程写入本 block 的 smem
    local_smem[threadIdx.x] = input[blockIdx.x * blockDim.x + threadIdx.x];

    // cluster.sync()：等待 cluster 内所有 block 的 smem 写完
    // 等价于 block 级 __syncthreads() 的 cluster 级版本
    cluster.sync();

    // 获取下一个 block 的 rank（环形）
    uint32_t neighborRank = (cluster.block_rank() + 1) % cluster.num_blocks();


    // ┌──────────┬──────────────────────┬────────────────────┐                                                                           
    // │          │       uint32_t       │       size_t       │                                                                           
    // ├──────────┼──────────────────────┼────────────────────┤                                                                           
    // │ 定义     │ 精确 32 位无符号整数 │ 平台相关无符号整数 │                                                                           
    // ├──────────┼──────────────────────┼────────────────────┤                                                                           
    // │ 32位系统 │ 4 字节               │ 4 字节             │
    // ├──────────┼──────────────────────┼────────────────────┤
    // │ 64位系统 │ 4 字节               │ 8 字节             │
    // ├──────────┼──────────────────────┼────────────────────┤
    // │ 头文件   │ <stdint.h>           │ <stddef.h>         │
    // ├──────────┼──────────────────────┼────────────────────┤
    // │ 用途     │ 明确需要 32 位的场景 │ 内存大小、数组索引 │
    // └──────────┴──────────────────────┴────────────────────┘

    // ---
    // 核心区别：uint32_t 跨平台固定大小，size_t 跟随平台指针宽度变化。
    // uint32_t a = 0xFFFFFFFF;  // 永远 4 字节，最大 ~4GB
    // size_t   b = sizeof(arr); // 64位系统 8 字节，可表示 >4GB

    // ---
    // 在 CUDA cluster 代码里的选择原因：

    // // cluster.block_rank() 返回 uint32_t
    // // cluster 内 block 数最多 8，用 32 位完全够，不需要 size_t
    // uint32_t neighborRank = (cluster.block_rank() + 1) % cluster.num_blocks();

    // // 内存大小用 size_t，因为可能超过 4GB
    // size_t sharedMemBytes = 128 * sizeof(short) + 64 * sizeof(float);

    // 原则：涉及内存大小/地址用 size_t，涉及固定范围的计数/ID 用 uint32_t。




    // map_shared_rank：把邻居 block 的 smem 映射到本线程可寻址的指针
    // 底层通过 SM-to-SM 互连（NVLink Fabric）直接访问，无需经过 global mem
    int* neighbor_smem = cluster.map_shared_rank(local_smem, neighborRank);

    // 跨 block 直接读取邻居 smem
    output[blockIdx.x * blockDim.x + threadIdx.x] = neighbor_smem[threadIdx.x];
}

// ── Kernel 2: atomicAdd 跨 Block 写入邻居 smem ──────────────────────────────
// 每个线程将自己的 input 值原子累加到邻居 block 的 smem 对应槽位，
// 演示 distributed shared memory 上的原子写入。
//   __cluster_dims__(4,1,1) —— 编译期提示，写在 kernel 函数上
                                              
//   __global__ void __cluster_dims__(4, 1, 1) myKernel(...) { ... }                                                                                  
   
//   - 告诉编译器这个 kernel 预期以 4x1x1 的 cluster 运行                                                                                             
//   - 编译器据此做优化（寄存器分配、指令调度等）
//   - 不写也能运行，只是少了编译期优化          
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
atomicAddClusterKernel(int* input, int* output)
{
    cg::cluster_group cluster = cg::this_cluster();

    __shared__ int local_smem[BLOCK_SIZE];

    // 初始化本 block 的 smem 为 0（后续作为被写入目标）
    local_smem[threadIdx.x] = 0;

    // 确保所有 block 的 smem 初始化完成，再开始跨 block 写入
    cluster.sync();

    // 每个线程把 input 值 atomicAdd 到邻居 block 的 smem
    // 若多个线程写同一槽位，atomicAdd 保证无竞争
    uint32_t neighborRank = (cluster.block_rank() + 1) % cluster.num_blocks();
    int* neighbor_smem    = cluster.map_shared_rank(local_smem, neighborRank);
    int  val              = input[blockIdx.x * blockDim.x + threadIdx.x];

    // atomicAdd 作用在 distributed shared memory 上
    // 等同于对普通 shared memory 的 atomicAdd，但目标在另一个 block
    atomicAdd(&neighbor_smem[threadIdx.x], val);

    // 原因：每个线程的 threadIdx.x 唯一，neighbor_smem[threadIdx.x] 对应不同槽位，没有多个线程写同一个地址。                                                                                                                       
    // Block 0 的 thread 0  → atomicAdd(&neighbor_smem[0], val)
    // Block 0 的 thread 1  → atomicAdd(&neighbor_smem[1], val)                                                                                         
    // Block 0 的 thread 2  → atomicAdd(&neighbor_smem[2], val)
    // // 各写各的槽位，完全不冲突                                                                                                                                          
    // 所以这里用普通赋值也正确：
    // neighbor_smem[threadIdx.x] = val;  // 无竞争，直接写也对

    // ---
    // atomicAdd 真正有意义的场景是多个线程写同一个地址：
    // // 统计：所有线程累加到同一个计数器
    // __shared__ int count;
    // atomicAdd(&count, 1);  // 256个线程都写 count，必须原子
    // // 直方图：多个线程可能落在同一个 bin
    // atomicAdd(&histogram[bin], 1);  // bin 可能相同，必须原子

    // ---
    // 所以代码里用 atomicAdd 的目的是演示 API 用法，展示它可以作用在 distributed shared memory
    // 上，而不是因为这里有竞争需要解决。实际生产代码中这里用普通赋值更合适。

    // 等待所有写入完成，再读出本 block 的 smem
    cluster.sync();

    output[blockIdx.x * blockDim.x + threadIdx.x] = local_smem[threadIdx.x];
}

int main()
{
    // ── 1. 检查设备是否支持 cluster（SM 9.0+）──────────────────────────────
    int device = 0;
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s  SM %d.%d\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("Cluster requires SM 9.0+ (Hopper). Skipping.\n");
        return 0;
    }

    // ── 2. 数据准备 ────────────────────────────────────────────────────────
    const int N = CLUSTER_SIZE * BLOCK_SIZE;   // 4 * 128 = 512 个元素

    int h_input[N], h_out1[N], h_out2[N];
    for (int i = 0; i < N; i++) h_input[i] = i;

    int *d_input, *d_out1, *d_out2;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out1,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out2,  N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    // ── 3. cluster 启动配置（必须用 cudaLaunchKernelEx）──────────────────
    // <<<>>> 语法无法指定 cluster 维度，必须通过 cudaLaunchKernelEx
    // cudaLaunchAttributeClusterDimension —— 运行期配置，写在 launch 参数里

    // attr[0].id               = cudaLaunchAttributeClusterDimension;
    // attr[0].val.clusterDim.x = CLUSTER_SIZE;

    // - 告诉 Runtime 实际启动时用多大的 cluster
    // - 这个必须写，否则 Runtime 不知道要组 cluster，kernel 会以普通方式启动
    cudaLaunchAttribute attr[1];
    attr[0].id                    = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x      = CLUSTER_SIZE;
    attr[0].val.clusterDim.y      = 1;
    attr[0].val.clusterDim.z      = 1;

    cudaLaunchConfig_t cfg   = {};
    cfg.gridDim              = CLUSTER_SIZE;   // 4 blocks，恰好 1 个 cluster
    cfg.blockDim             = BLOCK_SIZE;
    cfg.dynamicSmemBytes     = 0;
    cfg.attrs                = attr;
    cfg.numAttrs             = 1;

    // ── 4. 启动 Kernel 1 ──────────────────────────────────────────────────
    printf("=== Kernel 1: map_shared_rank read (block reads neighbor's smem) ===\n");
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, distributedSmemKernel, d_input, d_out1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out1, d_out1, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int b = 0; b < CLUSTER_SIZE; b++) {
        int neighbor = (b + 1) % CLUSTER_SIZE;
        printf("Block %d output (reads block %d smem) first 4: ", b, neighbor);
        for (int i = 0; i < 4; i++)
            printf("%4d ", h_out1[b * BLOCK_SIZE + i]);
        printf("\n");
    }

    // ── 5. 启动 Kernel 2 ──────────────────────────────────────────────────
    printf("\n=== Kernel 2: atomicAdd to neighbor's smem ===\n");
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, atomicAddClusterKernel, d_input, d_out2));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out2, d_out2, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int b = 0; b < CLUSTER_SIZE; b++) {
        int src = (b + CLUSTER_SIZE - 1) % CLUSTER_SIZE;  // 谁 atomicAdd 到了本 block  上一个block
        printf("Block %d smem (received atomicAdd from block %d) first 4: ", b, src);
        for (int i = 0; i < 4; i++)
            printf("%4d ", h_out2[b * BLOCK_SIZE + i]);
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_out1);
    cudaFree(d_out2);
    return 0;
}
