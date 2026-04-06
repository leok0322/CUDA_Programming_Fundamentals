/*
 * atomic_scope.cu
 *
 * 演示 cuda::atomic_ref 的五种 thread scope：
 *
 *   cuda::thread_scope_thread  — 只对当前线程自身可见（等同普通变量，无实际同步意义）
 *   cuda::thread_scope_warp    — warp 内 32 线程之间（不跨 warp）
 *   cuda::thread_scope_block   — 同一 block 内所有线程
 *   cuda::thread_scope_device  — 整个 GPU（最常用的跨 block scope）
 *   cuda::thread_scope_system  — CPU + GPU（用于 unified/pinned memory）
 *
 * scope 越大，硬件需要刷新的缓存层级越多，开销越大：
 *   thread < warp < block < device < system
 *
 * 编译：nvcc -O2 -arch=sm_75 --std=c++17 atomic_scope.cu -o atomic_scope
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define N     (1 << 20)   // 1M 个元素
#define BLOCK 256
#define GRID  ((N + BLOCK - 1) / BLOCK)

// =============================================================================
// scope_thread：只对当前线程自身可见
//
//   atomic_ref<T, thread_scope_thread> 不提供跨线程同步，
//   相当于普通变量操作（编译器不会生成任何额外同步指令）。
//   实际用途：对单线程内部的变量做原子风格的 API 操作（几乎没有实际意义）。
//
//   演示：每个线程用 thread scope atomic 对自己的私有寄存器变量累加，
//         最终结果写回 output[tid]（每个线程独立，完全无争抢）。
// =============================================================================
__global__ void scope_thread_demo(float* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // local_val 是寄存器变量，thread scope atomic 对它操作
    // 无跨线程可见性要求，等价于普通 load/store
    float local_val = 0.0f;
    cuda::atomic_ref<float, cuda::thread_scope_thread> ref(local_val);

    // 模拟每个线程对自己的局部变量做 3 次累加
    ref.fetch_add(1.0f, cuda::memory_order_relaxed);
    ref.fetch_add(2.0f, cuda::memory_order_relaxed);
    ref.fetch_add(3.0f, cuda::memory_order_relaxed);

    output[tid] = local_val;  // 期望每个 output[i] == 6.0
}

// =============================================================================
// scope_warp：warp 内 32 个线程之间
//
//   注意：libcudacxx 并未提供 cuda::thread_scope_warp。
//   warp 内 32 线程天然同步执行（SIMT lockstep），不需要 atomic 来保证一致性，
//   而是用 warp primitive：__shfl_down_sync / __shfl_xor_sync 做规约。
//
//   __shfl_down_sync(mask, val, delta)：
//     让每个 lane 从 (laneId + delta) 号 lane 读取 val（寄存器到寄存器，无需 smem）。
//     log2(32) = 5 步完成 warp 内求和，延迟极低（< 10 cycles/步）。
//
//   演示：每个 warp 对 laneId（0..31）求和 → 结果 == 0+1+...+31 == 496
// =============================================================================
__global__ void scope_warp_demo(float* output)
{
    int tid    = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    float val = (float)laneId;  // 每个线程的初始值

    // warp reduce：5 步 __shfl_down_sync，每步将高半部分的值加到低半部分
    // mask 0xffffffff 表示 warp 内所有 32 个 lane 都参与
    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);
    // 5 次迭代后，lane0 的 val == 0+1+2+...+31 == 496
    // 其他 lane 持有中间部分和（通常不使用）

    if (laneId == 0)
        output[blockIdx.x * (BLOCK / 32) + warpId] = val;
    // 期望每个 output[i] == 496
//   下标计算

//   blockIdx.x * (BLOCK / 32) + warpId
//       ↑               ↑          ↑
//   block 的起始偏移   每个block   block内第几个warp
//                     有几个warp   (0..7)
}


// warp 内寄存器到寄存器的直接传输，不经过 smem，不经过 HBM。                                                                                                
                                                                                                                                                            
//   参数含义                                                                                                                                                  
                                                                                                                                                            
//   ┌───────┬───────────────────────────────────────────────────────────────────┐
//   │ 参数  │                               含义                                │                                                                             
//   ├───────┼───────────────────────────────────────────────────────────────────┤                                                                           
//   │ mask  │ 参与的 lane 位掩码，0xffffffff = 32 位全 1 = 全部 32 个 lane 参与 │
//   ├───────┼───────────────────────────────────────────────────────────────────┤
//   │ val   │ 当前 lane 持有的值（寄存器变量）                                  │
//   ├───────┼───────────────────────────────────────────────────────────────────┤
//   │ delta │ 偏移量：我从 laneId + delta 号 lane 读取它的 val                  │
//   └───────┴───────────────────────────────────────────────────────────────────┘

//   单次调用图示（delta = 4，简化为 8 lane）

//   调用前：  lane  0    1    2    3    4    5    6    7
//            val   0    1    2    3    4    5    6    7

//   __shfl_down_sync(0xff, val, 4)
//     → 每个 lane 从 (laneId+4) 号 lane 读 val
//     → lane0 读 lane4，lane1 读 lane5 ...

//   调用后：  lane  0    1    2    3    4    5    6    7
//            val   4    5    6    7    4    5    6    7
//                 ↑                   ↑
//              lane0 得到 lane4 的值   lane4 不变（没有 lane8）

//   lane4~7 没有 laneId+delta 的来源（超出 warp 边界），值未定义（通常保持原值），所以只使用 lane0 的最终结果。

//   ---
//   5 步完成 warp 内求和（delta = 16 → 8 → 4 → 2 → 1）

//   初始：   lane  0  1  2  3  4  5  6  7  8 ... 31
//            val   0  1  2  3  4  5  6  7  8 ... 31

//   delta=16: val[i] += val[i+16]
//            lane0: 0+16=16  lane1: 1+17=18 ... lane15: 15+31=46

//   delta=8:  val[i] += val[i+8]
//            lane0: 16+24=40 ...

//   delta=4:  val[i] += val[i+4]
//   delta=2:  val[i] += val[i+2]
//   delta=1:  val[i] += val[i+1]

//   最终 lane0: 0+1+2+...+31 = 496  ✓

//   每步把"右半边"的部分和加到"左半边"，类似二叉树从叶到根。

//   ---
//   为什么比 smem reduce 快？

//   smem reduce：  寄存器 → smem → __syncthreads() → 寄存器  （需要 L1 往返）
//   shfl reduce：  寄存器 → 寄存器                           （warp 内部连线直接传）

//   __shfl_down_sync 走的是 warp 内的寄存器文件互联，延迟约 4~5 cycles/步，5 步共 ~25 cycles，远低于 smem 的 ~20 cycles/步 × 5 步。

//   ---
//   _sync 后缀的意义

//   CUDA 9+ 引入带 _sync 的版本，第一个参数 mask 指定哪些 lane 参与。没有 _sync 的旧版 __shfl_down 在 Volta+
//   上因独立线程调度可能产生竞态，已被废弃。0xffffffff 表示"32 个 lane 全部参与，必须全部到达这条指令后才执行"。



// =============================================================================
// scope_block：同一 block 内所有线程
//
//   保证 block 内缓存一致，不需要刷新 L2/HBM。
//   典型用途：block 内 reduction，最后一步 atomic 写到 smem 结果。
//
//   演示：block 内所有线程将自己的 threadIdx.x 加到 smem[0]，
//         最终 output[blockIdx.x] == 0+1+...+(BLOCK-1) == BLOCK*(BLOCK-1)/2
// =============================================================================
__global__ void scope_block_demo(float* output)
{
    __shared__ float block_sum;

    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();

    // block scope atomic：保证 block 内所有线程看到一致的 block_sum
    cuda::atomic_ref<float, cuda::thread_scope_block> ref(block_sum);
    ref.fetch_add((float)threadIdx.x, cuda::memory_order_relaxed);

    __syncthreads();

    if (threadIdx.x == 0)
        output[blockIdx.x] = block_sum;
    // 期望每个 output[i] == BLOCK*(BLOCK-1)/2 == 256*255/2 == 32640
}

// =============================================================================
// scope_device：整个 GPU 上所有线程（最常用的跨 block scope）
//
//   保证 device 级别缓存一致（L1/L2 全刷），开销最大（在 device scope 中）。
//   典型用途：跨 block 的全局归约，多个 block 各算一个子结果后合并。
//
//   演示：每个线程将 array[tid] 加到 global_result，
//         最终 global_result == sum(array[0..N-1])
// =============================================================================
__global__ void scope_device_demo(const float* __restrict__ array,
                                   float* global_result, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // device scope atomic：跨 block 可见
    cuda::atomic_ref<float, cuda::thread_scope_device> ref(*global_result);
    ref.fetch_add(array[tid], cuda::memory_order_relaxed);
}

// ===========
// scope_system：CPU + GPU 跨设备（unified memory / pinned memory）
//
//   保证 CPU 和 GPU 之间的原子一致性（PCIe 级别同步）。
//   必须使用 cudaMallocManaged 或 cudaHostAlloc 分配的内存。
//   适用场景：CPU 线程与 GPU 内核共同操作同一变量（生产者-消费者模式）。
//
//   演示：GPU kernel 将 N 个元素加到 managed_result，
//         CPU 随后用同样的 atomic_ref 读取（体现跨设备可见性）。
// ====================
__global__ void scope_system_demo(const float* __restrict__ array,
                                   float* managed_result, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // system scope atomic：保证 GPU 写对 CPU 可见（通过 unified memory 机制）
    cuda::atomic_ref<float, cuda::thread_scope_system> ref(*managed_result);
    ref.fetch_add(array[tid], cuda::memory_order_relaxed);
}

// ========
// main
// ==========
int main()
{
    printf("===== cuda::atomic_ref Thread Scope Demo =====\n\n");

    // ── 公共数据 ────
    float *d_array;
    CUDA_CHECK(cudaMalloc(&d_array, N * sizeof(float)));

    // 初始化 array[i] = 1.0f，方便验证（sum == N）
    // 用 cudaMemset 不行（float 1.0f 不是全 0），用 kernel 填充
    // 这里为简单起见，在 host 端准备后 memcpy
    float* h_array = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_array[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_array, h_array, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── 1. scope_thread ──────
    {
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        scope_thread_demo<<<GRID, BLOCK>>>(d_out, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 验证：抽查前 4 个值，期望都是 6.0
        float h_out[4];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[scope_thread]  output[0..3] = %.1f %.1f %.1f %.1f  "
               "(expected 6.0)\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);

        CUDA_CHECK(cudaFree(d_out));
    }

    // ── 2. scope_warp ─────────
    {
        int num_warps = GRID * (BLOCK / 32);
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, num_warps * sizeof(float)));

        scope_warp_demo<<<GRID, BLOCK>>>(d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 验证：抽查前 4 个 warp，期望都是 496（0+1+...+31）
        float h_out[4];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[scope_warp]    warp_sum[0..3] = %.0f %.0f %.0f %.0f  "
               "(expected 496)\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);

        CUDA_CHECK(cudaFree(d_out));
    }

    // ── 3. scope_block ─────────
    {
        float* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, GRID * sizeof(float)));

        scope_block_demo<<<GRID, BLOCK>>>(d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 验证：抽查前 4 个 block，期望都是 32640（256*255/2）
        float h_out[4];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float expected_block = (float)(BLOCK * (BLOCK - 1) / 2);
        printf("[scope_block]   block_sum[0..3] = %.0f %.0f %.0f %.0f  "
               "(expected %.0f)\n",
               h_out[0], h_out[1], h_out[2], h_out[3], expected_block);

        CUDA_CHECK(cudaFree(d_out));
    }

    // ── 4. scope_device ────
    {
        float* d_result;
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

        scope_device_demo<<<GRID, BLOCK>>>(d_array, d_result, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("[scope_device]  global_sum = %.0f  (expected %d)\n",
               h_result, N);

        CUDA_CHECK(cudaFree(d_result));
    }

    // ── 5. scope_system ───────
    // managed_result 必须用 cudaMallocManaged（CPU 和 GPU 共同访问）
    {
        float* managed_result;
        CUDA_CHECK(cudaMallocManaged(&managed_result, sizeof(float)));
        *managed_result = 0.0f;

        scope_system_demo<<<GRID, BLOCK>>>(d_array, managed_result, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // 确保 GPU 写对 CPU 可见

        // CPU 通过 system scope atomic_ref 读取（体现跨设备原子语义）
        cuda::atomic_ref<float, cuda::thread_scope_system> cpu_ref(*managed_result);
        float cpu_read = cpu_ref.load(cuda::memory_order_acquire);
        printf("[scope_system]  managed_sum (CPU read) = %.0f  (expected %d)\n",
               cpu_read, N);

        CUDA_CHECK(cudaFree(managed_result));
    }


// cpu_ref.load(cuda::memory_order_acquire)                                                                                                                  
                                                                                                                                                            
//   两部分拆开看：                                                                                                                                            
                                                                                                                                                            
//   ---                                                                                                                                                       
//   1. .load() — 原子读取                                                                                                                                     
                                                                                                                                                            
//   cuda::atomic_ref<float, cuda::thread_scope_system> cpu_ref(*managed_result);                                                                              
//   float cpu_read = cpu_ref.load(...);                       

//   等价于读取 *managed_result 的当前值，但保证是原子读（不会读到"写了一半"的中间状态）。

//   普通的 float cpu_read = *managed_result 在 CPU+GPU 共享内存场景下是不安全的，可能读到 stale 缓存值。

//   ---
//   2. memory_order_acquire — 内存顺序

//   memory_order 控制的是指令重排的边界，一共 6 种：

//   ┌──────────────┬──────────────────────────────────────────┐
//   │ memory_order │                   含义                   │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ relaxed      │ 无顺序保证，只保证原子性，可以被随意重排 │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ acquire      │ 此行之后的所有读写，不能被移到此行之前   │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ release      │ 此行之前的所有读写，不能被移到此行之后   │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ acq_rel      │ 同时具备 acquire + release               │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ seq_cst      │ 全局严格顺序，最强，开销最大             │
//   ├──────────────┼──────────────────────────────────────────┤
//   │ consume      │ acquire 的弱化版，实际很少用             │
//   └──────────────┴──────────────────────────────────────────┘

//   ---
//   acquire/release 配对使用

//   经典模式：GPU 用 release 写，CPU 用 acquire 读：

//   GPU 线程：
//     计算数据...
//     result[i] = 42;                          // ← 普通写，先于 release
//     flag.store(1, memory_order_release);     // ← release：上面的写不能被推迟到这行后

//   CPU 线程：
//     while (flag.load(memory_order_acquire) == 0) {}  // ← acquire：等到 flag=1
//     use(result[i]);                          // ← 此行之后的读，不能被提前到 acquire 前
//                                              //   保证能看到 GPU 的 result[i] = 42

//   acquire 的承诺：一旦我读到了 release 写入的值，release 之前的所有写对我也可见。


// ● flag 是一个用于线程间通信的信号变量，本质上就是一个整数，用来表示"某件事已经完成"。                                                                       
                                                                                                                                                            
//   上面那段代码是伪代码，演示 release/acquire 配对的经典用法：                                                                                               
                                                                                                                                                            
//   // 某处声明（managed memory，CPU GPU 都能访问）                                                                                                           
//   cuda::atomic<int, cuda::thread_scope_system> flag(0);  // 初始值 0 = "未完成"                                                                             
   
//   ---                                                                                                                                                       
//   执行流程                                                                                                                                                

//   GPU 线程                              CPU 线程

//   result[i] = 42;       // 写数据
//                                         while (flag.load(acquire) == 0)
//                                             {}  // 自旋等待，flag 还是 0，继续循环

//   flag.store(1, release); // flag写入1，发信号
//                                         // 然后读到 flag == 1，退出循环，否则等待继续循环
//                                         use(result[i]);  // 安全读数据

//   ---
//   本文件中的具体含义

//   // GPU kernel 用 relaxed 写（因为 cudaDeviceSynchronize 已经做了全局同步）
//   ref.fetch_add(array[tid], cuda::memory_order_relaxed);

//   // CPU 等 kernel 结束后读
//   CUDA_CHECK(cudaDeviceSynchronize());  // ← 这里已经保证了同步

//   float cpu_read = cpu_ref.load(cuda::memory_order_acquire);

//   这里 acquire 是保守写法——cudaDeviceSynchronize() 本身已经保证 GPU 所有写对 CPU 可见，所以 relaxed 也能工作。但用 acquire
//   更能表达意图：我要读一个"由别人写好的"值。

    // ── 总结 ──────────────────────────────────────────────────────────────────
    printf("\n===== Scope 对比总结 =====\n");
    printf("  scope_thread  : 无跨线程同步，仅语法统一，开销最小\n");
    printf("  scope_warp    : libcudacxx 不提供此 scope；warp 内用 __shfl_down_sync\n");
    printf("  scope_block   : 保证 block 内可见，刷 L1（smem 本身一致）\n");
    printf("  scope_device  : 保证全 GPU 可见，刷 L1 + L2，开销最大（GPU 内）\n");
    printf("  scope_system  : 保证 CPU+GPU 可见，需 managed/pinned 内存\n");
    printf("\n选择原则：用满足需求的最小 scope，避免不必要的缓存刷新。\n");

    // ── 清理 ──────────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);
    return 0;
}
