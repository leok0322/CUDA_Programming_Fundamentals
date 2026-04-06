/**
 * cuda_host_alloc_examples.cu
 *
 * 完整演示 cudaMallocHost 与 cudaHostAlloc 的所有用法：
 *   1. cudaMallocHost          — 标准 pinned memory
 *   2. cudaHostAllocDefault    — 等价于 cudaMallocHost
 *   3. cudaHostAllocWriteCombined — H2D 带宽优化
 *   4. cudaHostAllocMapped     — 零拷贝，GPU 直接访问 host 内存
 *   5. cudaHostAllocPortable   — 多 GPU 可见
 *   6. cudaHostRegister        — 注册已有 pageable 内存为 pinned
 *   7. 双向并发传输             — H2D + D2H 同时进行
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_host_alloc_examples.cu -o host_alloc_demo
 *
 * 运行：
 *   ./host_alloc_demo
 */

#include <cuda_runtime.h>  // CUDA 运行时 API：cudaMallocHost、cudaHostAlloc、cudaMemcpyAsync、cudaStream 等所有 CUDA 函数和类型的声明
#include <stdio.h>         // 标准 C 输入输出：fprintf、printf 等，用于打印结果和错误信息
#include <stdlib.h>        // 标准 C 通用库：exit、EXIT_FAILURE、malloc/free、posix_memalign 等，用于异常退出和内存管理
#include <string.h>        // 标准 C 字符串/内存操作：memset、memcpy 等，用于初始化和拷贝 host 端内存
#include <unistd.h>        // POSIX 系统接口：sysconf(_SC_PAGESIZE) 运行时查询系统页大小

// ─────────────
// 错误检查宏
// ────────────
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",             \
                    __FILE__, __LINE__,                                   \
                    cudaGetErrorName(err), cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// ─────────────────
// 辅助：简单的向量缩放 kernel
// ────────────────
__global__ void scaleKernel(float *out, const float *in, float factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * factor;
}

// ──────────────
// 辅助：CPU 侧验证
// ──────────────
static bool verify(const float *result, float expected, int n,
                   const char *label)
{
    for (int i = 0; i < n; i++) {
        if (fabsf(result[i] - expected) > 1e-4f) {
            printf("  [FAIL] %s: result[%d] = %f, expected %f\n",
                   label, i, result[i], expected);
            return false;
        }
    }
    printf("  [PASS] %s\n", label);
    return true;
}

// ─────────────────
// 辅助：GPU 事件计时
// ───────────────
static float time_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// 1 << 22 = 2^22 = 4,194,304（约 4M）个元素
// 每个 float 占 4 字节，故总内存 = 4,194,304 × 4 B = 16,777,216 B ≈ 16 MB
// static：文件作用域，仅本编译单元可见，避免与其他文件的同名符号冲突
// const：编译期常量，防止被意外修改，编译器可做常量折叠优化
                                                                                                                                                                              
// static 的作用                                                                                                                                                               
// - 限制符号的可见性为当前编译单元（.cu 文件内部）
// - 防止与其他源文件中同名的全局变量产生链接冲突   

                                                                                                                                                                            
// const 的作用                                                                                                                                                                
// - 声明为编译期常量，不可被修改                                                                                                                                            
// - 编译器可以做常量折叠优化，在用到 N 的地方直接替换为字面值，避免运行时读取内存      
static const int N       = 1 << 22;   // 1<<22 = 2^22 = 4,194,304 ≈ 4M 个 float，共 ≈ 16 MB
static const float SCALE = 2.5f;


// ═════════════════════════
// 示例 1：cudaMallocHost — 标准 pinned memory
// ═══════════════════════
void demo_cudaMallocHost()
{
    printf("\n=== 1. cudaMallocHost (standard pinned) ===\n");

    const size_t bytes = N * sizeof(float);

    // 分配 pinned host buffer 与 device buffer
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));   // pinned
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));   // pinned

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 创建 stream 和计时 event
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 初始化输入
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    // H2D → kernel → D2H，全部在同一 stream 内（有序串行，有依赖）
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes,
                               cudaMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, stream>>>(d_out, d_in, SCALE, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));   // 等所有操作完成

    printf("  elapsed: %.3f ms\n", time_ms(ev_start, ev_stop));
    verify(h_out, SCALE, N, "cudaMallocHost H2D→kernel→D2H");

    // 清理
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════════════════════════════
// 示例 2：cudaHostAllocWriteCombined — H2D 带宽优化
// 适合：CPU 只写、GPU 只读（H2D 方向）
// 禁忌：CPU 随机读（极慢）
// ══════════════════════════════════════


// 为什么 CPU 只写不读、绕过 Cache 对 H2D 最优？                                                                                                                              
//   普通 Pageable/Pinned Memory 的 H2D 路径                                                                                                                                     
                                                                                                                                                                              
//   CPU write → L1 → L2 → L3 → DRAM                                                                                                                                             
//                                 ↑                                                                                                                                             
//                            DMA 引擎读取  →  PCIe  →  GPU                                                                                                                      
                                                                                                                                                                              
//   DMA 引擎在读取前，必须先确保 Cache 里的脏数据已回写到DRAM（Cache Snooping/Flush），否则读到的是旧值。这个coherency检查是额外开销。                                     
                                                                                                                                                                              
//   ---                                                                                                                                                                         
//   Write-Combined (WC) Memory 的路径
                                                                                                                                                                              
//   cudaHostAllocWriteCombined 分配的内存完全绕过 CPU 的 L1/L2/L3 Cache：
                                                                                                                                                                              
//   CPU write → WC Buffer（CPU内部，仅几个 cache line 大小）
//                   ↓  积攒满后一次性 burst flush                                                                                                                               
//                 DRAM  →  DMA 引擎  →  PCIe  →  GPU                                                                                                                            
                                                                                                                                                                              
//   ┌──────────────────────────────┬────────────────────────────┬────────────────────────────────┐                                                                              
//   │             环节             │        普通 Pinned         │         Write-Combined         │
//   ├──────────────────────────────┼────────────────────────────┼────────────────────────────────┤                                                                              
//   │ CPU 写入路径                 │ L1→L2→L3→DRAM              │ WC Buffer→DRAM（burst）        │
//   ├──────────────────────────────┼────────────────────────────┼────────────────────────────────┤
//   │ DMA 读前是否需要 Cache Snoop │ 是，需要 flush 脏行        │ 否，DRAM 永远是最新值          │                                                                              
//   ├──────────────────────────────┼────────────────────────────┼────────────────────────────────┤                                                                              
//   │ CPU 写带宽                   │ 受 Cache 容量/替换策略限制 │ 受 PCIe/内存总线限制，理论更高 │                                                                              
//   ├──────────────────────────────┼────────────────────────────┼────────────────────────────────┤                                                                              
//   │ CPU 读速度                   │ 快（命中 Cache）           │ 极慢（每次读回主存，无预取）   │
//   └──────────────────────────────┴────────────────────────────┴────────────────────────────────┘                                                                              
                  
//   WC 的核心机制：CPU 内部有少量 Write-Combining Buffer，会将连续地址的多次小写合并成一个大的 burst 写入 DRAM，减少内存总线事务数，DMA 引擎读时无需任何 coherency 协议介入。   
                  
//   所以 WC 的适用条件：                                                                                                                                                        
//   - CPU 顺序写满整个 buffer（才能充分利用 burst 合并）
//   - CPU 几乎不回读（否则每次读都是全延迟访存，性能灾难）                                                                                                                      
//   - 恰好匹配 H2D 场景：CPU 准备数据写入 → DMA 搬运到 GPU
                                                                                                                                                                              

// 为什么 Write-Combined受 PCIe/内存总线限制？                                                                                                                                 
// ● 先看普通 Cached 内存的写入路径                                                                                                                                              
                                                                                                                                                                              
//   CPU → L1 Cache → L2 Cache → L3 Cache → DRAM                                                                                                                                 
                                                                                                                                                                              
//   CPU 写数据时，大多数情况下只写到 Cache 就返回了，不需要等数据真正到达 DRAM。Cache 是 SRAM，速度极快（L1 ~4 cycles），所以 CPU 写入的瓶颈在 Cache 层，内存总线几乎不参与。   
                                                                                                                                                                              
//   ---                                                                                                                                                                         
//   WC 内存的写入路径
                                                                                                                                                                              
//   CPU → WC Buffer（CPU片内，仅64字节） → flush → DRAM
                                                                                                                                                                              
//   WC 完全绕过 Cache，数据必须真正写到 DRAM，这条路上经过的就是内存总线：                                                                                                      
                                                                                                                                                                              
//   CPU 核心                                                                                                                                                                    
//      ↓                                                                                                                                                                        
//   内存控制器（IMC）
//      ↓  ← 这段就是内存总线，带宽有限（如 DDR5 双通道 ~100 GB/s）                                                                                                              
//     DRAM                                                                                                                                                                      
                                                                                                                                                                              
//   每次 WC Buffer 写满 64 字节后 flush，都要占用一次内存总线事务。CPU 写多快，内存总线就要传多快，总线带宽成为上限。                                                           
                                                                                                                                                                              
//   ---                                                                                                                                                                         
//   对比总结        
          
//   ┌──────────────────┬────────────────────┬────────────────────────┐
//   │                  │     Cached 写      │         WC 写          │                                                                                                          
//   ├──────────────────┼────────────────────┼────────────────────────┤
//   │ 数据最终落点     │ Cache（SRAM）      │ DRAM                   │                                                                                                          
//   ├──────────────────┼────────────────────┼────────────────────────┤
//   │ 内存总线是否参与 │ 不参与（延迟写回） │ 每次都参与             │                                                                                                          
//   ├──────────────────┼────────────────────┼────────────────────────┤                                                                                                          
//   │ 瓶颈在哪         │ Cache 速度（极快） │ 内存总线带宽（有上限） │                                                                                                          
//   └──────────────────┴────────────────────┴────────────────────────┘                                                                                                          
                  
//   ---                                                                                                                                                                         
//   那为什么 H2D 场景下 WC 反而更快？
                                                                                                                                                                              
//   因为 H2D 的目标本来就是把数据送到 DRAM，再由 DMA 经 PCIe 搬到 GPU：
                                                                                                                                                                              
//   Cached:  CPU写→Cache→(flush)→DRAM → DMA → PCIe → GPU
//                 ↑ 这一步有 Cache Snoop 开销，且 flush 不可控                                                                                                                  
                                                                                                                                                                              
//   WC:      CPU写→WC Buffer→(burst flush)→DRAM → DMA → PCIe → GPU                                                                                                              
//                 ↑ 无 Snoop，burst 写入效率高                                                                                                                                  
                                                                                                                                                                              
//   WC 虽然受内存总线限制，但这个限制和 PCIe 带宽（~16 GB/s）相比不是真正的瓶颈——PCIe 才是整个 H2D 链路最窄的地方，内存总线（~100 GB/s）远快于 PCIe，所以 WC                    
//   的总线限制在实际中不构成问题。


                                                                                                                                                                              
                                                                     
                                                                                          
// ❯ DMA 读前是否需要 Cache Snoop是什么意思？                                                                                                                                    
                                                                                                                                                                              
// ● Cache Snoop 是什么？                                                                                                                                                        
                                                                                                                                                                              
//   先理解问题的根源                                                                                                                                                            
                                                                                                                                                                              
//   CPU 写数据时，数据先进 Cache，不立刻写回 DRAM（这叫 Write-Back 策略）：                                                                                                     
                                                                                                                                                                              
//   CPU 写 addr[100] = 3.14f                                                                                                                                                    
//          ↓                                                                                                                                                                    
//     L1 Cache ← 数据在这里（脏，DRAM里还是旧值）
//     L2 Cache                                                                                                                                                                  
//     L3 Cache      
//     DRAM       ← 此时还是旧值！                                                                                                                                               
                  
//   此时如果 DMA 引擎直接去 DRAM 读 addr[100]，读到的是旧值，数据错误。                                                                                                         
   
//   ---                                                                                                                                                                         
//   Cache Snoop 就是解决这个问题的机制
                                                                                                                                                                              
//   Snoop（窥探）：DMA 发出读请求时，内存控制器会广播询问所有 CPU 的 Cache：
                                                                                                                                                                              
//   DMA 想读 addr[100]
//          ↓                                                                                                                                                                    
//   内存控制器广播："谁的 Cache 里有 addr[100] 的脏数据？"                                                                                                                      
//          ↓
//   L3 Cache 回应："我有，而且是脏的"                                                                                                                                           
//          ↓                                                                                                                                                                    
//   两种处理方式：
//     方式A：Cache 把脏数据写回 DRAM，再让 DMA 读 DRAM                                                                                                                          
//     方式B：Cache 直接把数据转发给 DMA（Snoop Forward）                                                                                                                        
                                                                                                                                                                              
//   这个"广播—询问—等待回应"的过程就叫 Cache Snooping，它有额外的时间开销。                                                                                                     
                                                                            


//   ---                                                                                                                                                                         
//   WC Memory 为什么不需要 Snoop？
                                                                                                                                                                              
//   WC 内存写入时完全绕过 Cache，数据直接到 DRAM：
                                                                                                                                                                              
//   CPU 写 addr[100] = 3.14f（WC 内存）
//          ↓                                                                                                                                                                    
//     WC Buffer → flush → DRAM ← 数据直接在这里
                                                                                                                                                                              
//     Cache 里根本没有这个地址的副本                                                                                                                                            
//          ↓                                                                                                                                                                    
//   DMA 读 DRAM → 直接拿到最新值，无需任何 Snoop 



                            


// L3 Cache → DRAM 经过内存总线嘛，那PCIe又是什么呢                                                                                                                            
                                                   
// ● L3 Cache → DRAM 经过内存总线                                                                                                                                                
                                                                                                                                                                              
//   CPU 芯片内部                          芯片外部                                                                                                                              
//   ┌─────────────────────────┐                                                                                                                                                 
//   │  Core                   │                                                                                                                                                 
//   │   ↓                     │
//   │  L1 Cache               │                                                                                                                                                 
//   │   ↓                     │                                                                                                                                                 
//   │  L2 Cache               │
//   │   ↓                     │                                                                                                                                                 
//   │  L3 Cache               │
//   │   ↓                     │
//   │  内存控制器 (IMC)        │ ──内存总线(DDR)──→  DRAM (内存条)
//   └─────────────────────────┘                                                                                                                                                 
                                                                                                                                                                              
//   内存总线（DDR Bus） 是 CPU 芯片到内存条之间的物理线路，传输协议是 DDR4/DDR5，带宽约 50～100 GB/s。                                                                          
                                                                                                                                                                              
//   ---                                                                                                                                                                         
//   PCIe 是什么     
                                                                                                                                                                              
//   PCIe 是CPU/主板 连接外部设备的总线，GPU 就是通过 PCIe 插槽接入系统的：
                                                                                                                                                                              
//   CPU 芯片内部    
//   ┌─────────────────────────┐                                                                                                                                                 
//   │  Core                   │
//   │  L1/L2/L3 Cache         │                                                                                                                                                 
//   │  内存控制器 (IMC)        │ ──内存总线(DDR)──→  DRAM
//   │                         │                                                                                                                                                 
//   │  PCIe 控制器            │ ──PCIe 总线──────→  GPU显存
//   └─────────────────────────┘                                                                                                                                                 
//            主板 PCIe 插槽                                                                                                                                                     
                                                                                                                                                                              
//   PCIe 总线 是 CPU 到 GPU 之间的物理通道，带宽远低于内存总线：                                                                                                                
                  
//   ┌──────────────────────┬───────────┬──────────────────────┐                                                                                                                 
//   │         总线         │ 典型带宽  │       连接什么       │
//   ├──────────────────────┼───────────┼──────────────────────┤
//   │ 内存总线 DDR5 双通道 │ ~100 GB/s │ CPU ↔                │
//   │                      │           │ 内存条（DRAM）       │
//   ├──────────────────────┼───────────┼──────────────────────┤                                                                                                                 
//   │ PCIe 4.0 x16         │ ~32 GB/s  │ CPU ↔ GPU            │
//   ├──────────────────────┼───────────┼──────────────────────┤                                                                                                                 
//   │ PCIe 5.0 x16         │ ~64 GB/s  │ CPU ↔ GPU            │
//   └──────────────────────┴───────────┴──────────────────────┘                                                                                                                 
   



//   H2D 完整路径    
//   CPU Core
//     ↓ 写（内存总线）                                                                                                                                                          
//   DRAM
//     ↓ 读（内存总线）                                                                                                                                                          
//   DMA读，搬运到 PCIe 控制器缓冲区
//     ↓ 写（PCIe 总线）                                                                                                                                                         
//   GPU PCIe 接收端
//     ↓ 写（GPU 内部总线）                                                                                                                                                      
//   GPU 显存 (VRAM)     

//   ---                                                                                                                                                                         
//   所以数据实际被复制了几次？
                            
//   ┌───────────────────┬───────┐
//   │       阶段        │ 操作  │                                                                                                                                               
//   ├───────────────────┼───────┤
//   │ CPU → DRAM        │ 写1次 │                                                                                                                                               
//   ├───────────────────┼───────┤
//   │ DRAM → PCIe缓冲区 │ 读1次 │
//   ├───────────────────┼───────┤                                                                                                                                               
//   │ PCIe缓冲区 → VRAM │ 写1次 │
//   └───────────────────┴───────┘     
                                                                                                                                               
                                                                                                                                                                              
//   内存总线 DDR5 双通道读写合计 ~100 GB/s                                                                                                                                      
//   PCIe 5.0 x16 单向             ~64 GB/s   

// CPU 写 和 DMA 读 加在一起，总需求也很难超过 100 GB/s，因为两者速度都受 PCIe 上限（64 GB/s）约束——DMA 读的速度不会超过 PCIe 能搬走的速度，CPU写的速度通常也远小于内存总线上限。              
// 所以内存总线带宽足够宽，在典型 H2D 场景下仍然不是瓶颈，PCIe 依然是最窄的地方。 



void demo_writeCombined()
{
    printf("\n=== 2. cudaHostAllocWriteCombined ===\n");

    const size_t bytes = N * sizeof(float);

    float *h_in;
    // WriteCombined：绕过 CPU cache，顺序写直达内存，H2D 带宽最优
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocWriteCombined));

    // ✅ 顺序写：WC 的正确用法
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    // ⚠️  不要随机读 WC 内存：
    // float val = h_in[rand() % N];  // 极慢，无 cache 加速

    float *h_out;
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));   // D2H 方向需要 CPU 读，用普通 pinned

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes,
                               cudaMemcpyHostToDevice, stream));

    int threads = 256, blocks = (N + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, stream>>>(d_out, d_in, SCALE, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("  elapsed: %.3f ms\n", time_ms(ev_start, ev_stop));
    verify(h_out, SCALE, N, "WriteCombined H2D");

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════════════════
// 示例 3：cudaHostAllocMapped — 零拷贝（GPU 直接访问 host 内存）
// 不需要 cudaMemcpyAsync；GPU 每次访问都通过 PCIe 读 host 内存
// 适合：数据只用一次 / 数据量超过显存 / 传输与计算无法重叠
// ══════════════════════════
void demo_mapped()
{
    printf("\n=== 3. cudaHostAllocMapped (zero-copy) ===\n");

    // 必须先设置设备 flag，允许 mapped host memory
    // 注意：必须在任何其他 CUDA 操作之前调用（或在 cudaSetDevice 之后、
    // 首次使用设备之前）。这里为了演示独立性，使用设备重置。
    // 生产代码中在程序初始化阶段调用一次即可。
    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    const size_t bytes = N * sizeof(float);

    float *h_in, *h_out;
    // cudaHostAllocMapped：pinned + 映射到 GPU 地址空间
    CUDA_CHECK(cudaHostAlloc(&h_in,  bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_out, bytes, cudaHostAllocMapped));

    // 获取 GPU 侧的访问指针（与 h_in 指向同一物理内存，但通过 PCIe 映射）
    float *d_in, *d_out;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_in,  h_in,  0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_out, h_out, 0));

    // CPU 初始化输入（直接写 h_in）
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    // 无需 memcpy，直接用 d_in/d_out 启动 kernel
    // GPU 每次访问 d_in 都通过 PCIe 读 host RAM
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    int threads = 256, blocks = (N + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, stream>>>(d_out, d_in, SCALE, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 结果直接在 h_out，无需 D2H memcpy
    printf("  elapsed: %.3f ms  (note: slower than VRAM, PCIe per-access)\n",
           time_ms(ev_start, ev_stop));
    verify(h_out, SCALE, N, "Mapped zero-copy");

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ═════════════════════════════
// 示例 4：cudaHostAllocPortable — 多 GPU 可见
// 默认 pinned memory 只对创建时的 GPU context 是 pinned 的；
// Portable 让它对所有 GPU 都是 pinned 的。
// ════════════════════════════
void demo_portable()
{
    printf("\n=== 4. cudaHostAllocPortable (multi-GPU) ===\n");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        printf("  (only %d GPU found, skipping multi-GPU demo; "
               "demonstrating single-GPU portable behavior)\n", deviceCount);
    }

    const size_t bytes = N * sizeof(float);

    float *h_in, *h_out;
    // Portable：对所有 GPU context 都是 pinned
    CUDA_CHECK(cudaHostAlloc(&h_in,  bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_out, bytes, cudaHostAllocPortable));

    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    // GPU 0 使用
    CUDA_CHECK(cudaSetDevice(0));

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes,
                               cudaMemcpyHostToDevice, stream));

    int threads = 256, blocks = (N + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, stream>>>(d_out, d_in, SCALE, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    verify(h_out, SCALE, N, "Portable on GPU 0");

    // 若有第二块 GPU，同样的 h_in/h_out 可以直接给 GPU 1 使用
    // 不需要重新分配或重新注册
    if (deviceCount >= 2) {
        CUDA_CHECK(cudaSetDevice(1));
        float *d_in1, *d_out1;
        CUDA_CHECK(cudaMalloc(&d_in1,  bytes));
        CUDA_CHECK(cudaMalloc(&d_out1, bytes));

        cudaStream_t stream1;
        CUDA_CHECK(cudaStreamCreate(&stream1));

        // 同一块 h_in，给 GPU 1 用，仍然是真正的 pinned（DMA 直接访问）
        CUDA_CHECK(cudaMemcpyAsync(d_in1, h_in, bytes,
                                   cudaMemcpyHostToDevice, stream1));

        scaleKernel<<<blocks, threads, 0, stream1>>>(d_out1, d_in1, SCALE, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out1, bytes,
                                   cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream1));

        verify(h_out, SCALE, N, "Portable on GPU 1");

        CUDA_CHECK(cudaFree(d_in1));
        CUDA_CHECK(cudaFree(d_out1));
        CUDA_CHECK(cudaStreamDestroy(stream1));
        CUDA_CHECK(cudaSetDevice(0));
    }

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════════════════════════
// 示例 5：cudaHostRegister — 注册已有 pageable 内存为 pinned
// 适合：无法控制内存分配方式（如第三方库返回的 buffer）
// ═══════════════════════════════
void demo_hostRegister()
{
    printf("\n=== 5. cudaHostRegister (register existing memory) ===\n");

    const size_t bytes = N * sizeof(float);

    // ── 显式页对齐分配 ───────
    //
    // 页大小在运行时通过 sysconf 查询，不硬编码 4096：
    //   - x86_64 Linux 默认 4096 B
    //   - ARM 部分平台使用 65536 B（64 KB）大页
    //   - 硬编码 4096 在这些平台上会导致对齐不足
    //
    // posix_memalign 签名：
    //   int posix_memalign(void **memptr, size_t alignment, size_t size)
    //     memptr    : [out] 对齐后的地址写入 *memptr
    //     alignment : 对齐字节数，必须是 2 的幂且是 sizeof(void*) 的整数倍
    //     size      : 分配字节数
    //   返回值：0 成功，非 0 为 errno 错误码
    //
    // ── size 向上取整到页大小整数倍 ──────────────────────────────────
    //
    // 目标：找到 >= bytes 的最小页大小整数倍
    // 公式：aligned_size = (bytes + page_size - 1) & ~(page_size - 1)
    //
    // 以 page_size = 4096 = 0x1000 为例，分三步拆解：
    //
    // ── 第一步：page_size - 1 ────────────────────────────────────────
    //
    //   page_size     = 4096 = 0x1000 = 0001 0000 0000 0000
    //   page_size - 1 = 4095 = 0x0FFF = 0000 1111 1111 1111  低12位全1
    //
    //   page_size 是 2 的幂（2^12），减1后低12位全变为1，高位全为0。
    //   这是 2 的幂的特性：2^n - 1 的二进制恰好是 n 个连续的 1。
    //
    // ── 第二步：~(page_size - 1) 页对齐掩码 ─────────────────────────
    //
    //   page_size - 1  = 0x00000FFF = ...0000 1111 1111 1111
    //   ~(page_size-1) = 0xFFFFF000 = ...1111 0000 0000 0000
    //                                 └──高位全1──┘└─低12位全0─┘
    //
    //   任意数 & 此掩码的效果：低12位清零，高位不变。
    //   即把数值向下截断到最近的页边界（向下取整）。
    //
    //   示例：5000 & 0xFFFFF000
    //     5000 = 0x1388 = ...0001 0011 1000 1000
    //     掩码           = ...1111 0000 0000 0000
    //                      ─────────────────────
    //     结果 = 0x1000  =  4096  ← 余数 904 被截掉，向下取整到 1 页
    //
    // ── 第三步：先加 (page_size-1) 把向下变向上 ─────────────────────
    //
    //   单独 & 掩码只能向下取整，先加 (page_size-1) 再掩码即可向上取整：
    //
    //   原理（两种情况）：
    //
    //   bytes 不是页的整数倍（余数 r，0 < r < page_size）：
    //     bytes                   = k*page + r
    //     bytes + (page_size - 1) = k*page + r + page - 1
    //                             = (k+1)*page + (r-1)   ← r-1 < page，低位
    //     & 掩码清掉 (r-1)        → (k+1)*page           ← 进到下一页，正确
    //
    //   bytes 恰好是页的整数倍（余数 r = 0）：
    //     bytes                   = k*page
    //     bytes + (page_size - 1) = k*page + page - 1
    //                             = (k+1)*page - 1       ← 低位全1但没进位
    //     & 掩码清掉低位          → k*page               ← 不变，正确
    //
    // ── 完整逐位示例 ─────────────────────────────────────────────────
    //
    //   情况 A：bytes = 5000（余数 904，不对齐）
    //
    //     bytes             =  5000 = 0x00001388
    //     + (page_size - 1) = +4095 = 0x00000FFF
    //                         ─────────────────────
    //     sum               =  9095 = 0x00002387   低12位 = 0x387
    //     & 0xFFFFF000       =        0xFFFFF000
    //                         ─────────────────────
    //     aligned_size      =  8192 = 0x00002000   ← 2 页，向上取整正确
    //
    //   情况 B：bytes = 4096（余数 0，已对齐）
    //
    //     bytes             =  4096 = 0x00001000
    //     + (page_size - 1) = +4095 = 0x00000FFF
    //                         ─────────────────────
    //     sum               =  8191 = 0x00001FFF   低12位 = 0xFFF
    //     & 0xFFFFF000       =        0xFFFFF000
    //                         ─────────────────────
    //     aligned_size      =  4096 = 0x00001000   ← 1 页，不变正确
    //
    //   情况 C：bytes = 16777216（本例，4M × 4B = 4096 页）
    //     已是页的整数倍，aligned_size = 16777216，不变

    const size_t page_size    = (size_t)sysconf(_SC_PAGESIZE);
    const size_t aligned_size = (bytes + page_size - 1) & ~(page_size - 1);

    printf("  页大小            : %zu B\n", page_size);
    printf("  原始 bytes        : %zu B\n", bytes);
    printf("  对齐后 size       : %zu B\n", aligned_size);

    float *h_in  = NULL;
    float *h_out = NULL;
    if (posix_memalign((void **)&h_in,  page_size, aligned_size) != 0 ||
        posix_memalign((void **)&h_out, page_size, aligned_size) != 0) {
        fprintf(stderr, "posix_memalign 失败\n");
        return;
    }

    // 验证地址低 12 位全为 0（即是页大小的整数倍）
    printf("  h_in  : %p  低12位=0x%03zx（应为 0x000）\n",
           (void *)h_in,  (size_t)h_in  & (page_size - 1));
    printf("  h_out : %p  低12位=0x%03zx（应为 0x000）\n",
           (void *)h_out, (size_t)h_out & (page_size - 1));

    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    // 把已有内存注册为 pinned
    // 地址和 size 均已显式页对齐，cudaHostRegister 可安全锁定
    CUDA_CHECK(cudaHostRegister(h_in,  bytes, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_out, bytes, cudaHostRegisterDefault));
    // 注册后：h_in 和 h_out 的物理页被锁定，DMA 可直接访问

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes,
                               cudaMemcpyHostToDevice, stream));

    int threads = 256, blocks = (N + threads - 1) / threads;
    scaleKernel<<<blocks, threads, 0, stream>>>(d_out, d_in, SCALE, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    verify(h_out, SCALE, N, "cudaHostRegister");

    // 必须先 Unregister，再 free
    // 顺序反了会导致 undefined behavior
    CUDA_CHECK(cudaHostUnregister(h_in));
    CUDA_CHECK(cudaHostUnregister(h_out));
    free(h_in);
    free(h_out);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ══════════════════════════════
// 示例 6：双向并发传输 — H2D + D2H 同时进行
// 前提：
//   - compute capability >= 2.0（两个独立 DMA 引擎）
//   - 两个方向在不同的 stream
//   - 两侧 buffer 都是 pinned memory
// ══════════════════════════
void demo_bidirectional()
{
    printf("\n=== 6. Bidirectional concurrent H2D + D2H ===\n");

    const size_t bytes = N * sizeof(float);

    // H2D 方向：用 WriteCombined（CPU 只写，H2D 最快）
    float *h_in;
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocWriteCombined));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    // D2H 方向：用普通 pinned（CPU 需要读结果）
    float *h_out;
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 预先在 d_out 填入数据，供 D2H 传输使用
    // 这里用一个辅助 kernel 初始化 d_out = 3.0f
    {
        float *h_init;
        CUDA_CHECK(cudaMallocHost(&h_init, bytes));
        for (int i = 0; i < N; i++) h_init[i] = 3.0f;
        CUDA_CHECK(cudaMemcpy(d_out, h_init, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(h_init));
    }

    // 两个独立的 stream，对应两个 DMA 引擎
    cudaStream_t s_h2d, s_d2h;
    CUDA_CHECK(cudaStreamCreate(&s_h2d));
    CUDA_CHECK(cudaStreamCreate(&s_d2h));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 在 s_h2d 上记录起始时间
    CUDA_CHECK(cudaEventRecord(ev_start, s_h2d));

    // 同时提交两个方向的传输
    // DMA engine 1（H2D）：h_in → d_in
    CUDA_CHECK(cudaMemcpyAsync(d_in,  h_in,  bytes,
                               cudaMemcpyHostToDevice, s_h2d));
    // DMA engine 2（D2H）：d_out → h_out
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes,
                               cudaMemcpyDeviceToHost, s_d2h));

    // 等两个 stream 都完成
    CUDA_CHECK(cudaStreamSynchronize(s_h2d));
    CUDA_CHECK(cudaStreamSynchronize(s_d2h));

    // 记录结束时间（两个都已完成，在任一 stream 上记录均可）
    CUDA_CHECK(cudaEventRecord(ev_stop, s_h2d));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    //   cudaEventRecord 不是立刻打时间戳                                                                                                                  
                                                                                                                                                        
    //   cudaEventRecord 是异步的——它只是把"记录时间戳"这个命令插入 stream 的队列，不等命令执行完就返回。
                                                                                                                                                        
    //   CPU 调用 cudaEventRecord(ev_stop, s_h2d)
    //            ↓                                                                                                                                        
    //   s_h2d 队列: [...已完成的操作... | record ev_stop]  ← 这个命令还在队列里排队
    //            ↓                                                                                                                                        
    //   CPU 立刻返回，此时 ev_stop 的时间戳可能还没写入                                                                                                   
                                                                                                                                                        
    //   虽然 cudaStreamSynchronize(s_h2d) 已经保证 stream 里之前的操作都完成了，但 cudaEventRecord 是之后新提交的命令，cudaStreamSynchronize 无法覆盖它。 
                                                                                                                                                        
    //   ---                                                                                                                                               
    //   cudaEventSynchronize 的作用
                                
    //   cudaEventSynchronize(ev_stop) 确保 ev_stop 的时间戳已经被 GPU 写入，CPU 读取才有效：
                                                                                                                                                        
    //   cudaStreamSynchronize(s_h2d)    ← 保证 record 命令之前的操作完成
    //   cudaEventRecord(ev_stop, s_h2d) ← 新提交：把当前时刻写入 ev_stop                                                                                  
    //   cudaEventSynchronize(ev_stop)   ← 等这个写入动作本身完成                                                                                          
    //                                      此后 cudaEventElapsedTime 读到的才是正确值                                                                     
                                                                                                                                                        
    //   ---                                                                                                                                               
    //   如果去掉会怎样                                                                                                                                    
                                                                                                                                                        
    //   time_ms() 内部调用 cudaEventElapsedTime，它要求两个 event 都已完成记录。若 ev_stop
    //   时间戳还未写入就去读，返回的是垃圾值或报错（cudaErrorNotReady）。                                                                                 
                    
    //   ---                                                                                                                                               
    //   一句话总结      
                                                                                                                                                        
    //   cudaStreamSynchronize 保证的是"stream 里已有操作完成"，cudaEventRecord 是新操作，需要 cudaEventSynchronize
    //   单独等它的时间戳写入完成，两者不重叠，cudaEventSynchronize 不多余。

    printf("  bidirectional elapsed: %.3f ms\n",
           time_ms(ev_start, ev_stop));
    verify(h_out, 3.0f, N, "D2H result");

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaStreamDestroy(s_h2d));
    CUDA_CHECK(cudaStreamDestroy(s_d2h));
}


// ═══════════════════════════
// 示例 7：双 buffer 流水线 — 计算与传输重叠
// 核心思想：
//   当 GPU 处理 batch[i] 时，CPU 准备 batch[i+1] 并做 H2D 传输
//   达到 compute + transfer 真正重叠，隐藏传输延迟
// ════════════════════════════
void demo_doubleBuffer()
{
    printf("\n=== 7. Double-buffer pipeline (compute + transfer overlap) ===\n");

    const int NUM_BATCHES  = 4;
    // 向上取整：(N + NUM_BATCHES - 1) / NUM_BATCHES
    // 确保最后一个 batch 能覆盖 N 不被整除时的余下元素。
    // kernel 内已有 if (idx < n) 边界检查，最后一个 batch 不足 BATCH_SIZE 时安全。
    const int BATCH_SIZE   = (N + NUM_BATCHES - 1) / NUM_BATCHES;
    const size_t batch_bytes = BATCH_SIZE * sizeof(float);

    // 双 buffer：ping 和 pong，交替使用
    float *h_ping, *h_pong;
    CUDA_CHECK(cudaMallocHost(&h_ping, batch_bytes));
    CUDA_CHECK(cudaMallocHost(&h_pong, batch_bytes));

    float *d_ping_in,  *d_pong_in;
    float *d_ping_out, *d_pong_out;
    CUDA_CHECK(cudaMalloc(&d_ping_in,  batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_pong_in,  batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_ping_out, batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_pong_out, batch_bytes));

    // 两个 stream：
    //   s_compute：运行 kernel
    //   s_transfer：做 H2D memcpy（与 kernel 重叠）
    cudaStream_t s_compute, s_transfer;
    CUDA_CHECK(cudaStreamCreate(&s_compute));
    CUDA_CHECK(cudaStreamCreate(&s_transfer));

    // event：用于跨 stream 的依赖同步
    cudaEvent_t ev_transfer_done;
    CUDA_CHECK(cudaEventCreate(&ev_transfer_done));

    int threads = 256;

    float *h_buf[2]     = { h_ping,     h_pong     };
    float *d_in[2]      = { d_ping_in,  d_pong_in  };
    float *d_out[2]     = { d_ping_out, d_pong_out };

    // 预填充第 0 个 batch 并 H2D，作为 pipeline 启动。
    // 循环内只负责"预取下一批"，无人负责第 0 批，必须在循环外单独传好：
    //   循环外：                transfer batch[0]
    //   循环 b=0：compute[0]  ‖  transfer[1]
    //   循环 b=1：compute[1]  ‖  transfer[2]
    //   循环 b=2：compute[2]  ‖  transfer[3]
    //   循环 b=3：compute[3]
    // 若不预填充，循环第一轮 compute batch[0] 时显存里是未初始化的垃圾数据。
    int batch0_n = (BATCH_SIZE < N) ? BATCH_SIZE : N;
    for (int i = 0; i < batch0_n; i++) h_buf[0][i] = 1.0f;
    CUDA_CHECK(cudaMemcpyAsync(d_in[0], h_buf[0], batch0_n * sizeof(float),
                               cudaMemcpyHostToDevice, s_transfer));
    CUDA_CHECK(cudaEventRecord(ev_transfer_done, s_transfer));
    // compute stream 等待第 0 个 batch 传输完成后再开始计算
    CUDA_CHECK(cudaStreamWaitEvent(s_compute, ev_transfer_done, 0));

    for (int b = 0; b < NUM_BATCHES; b++) {
        int cur = b % 2;    // cur是看用的哪个传输通道
        int nxt = (b + 1) % 2;  // 下一轮使用的传输通道

        // 当前 batch 的实际元素数：最后一个 batch 可能不足 BATCH_SIZE
        // b*BATCH_SIZE 是当前 batch 的起始元素索引
        int cur_n = N - b * BATCH_SIZE;          // 剩余元素数
        if (cur_n > BATCH_SIZE) cur_n = BATCH_SIZE;  // 不超过 BATCH_SIZE
        int blocks = (cur_n + threads - 1) / threads;

        // 在 compute stream 上处理当前 batch（传入实际元素数 cur_n）
        scaleKernel<<<blocks, threads, 0, s_compute>>>(
            d_out[cur], d_in[cur], SCALE, cur_n);
        CUDA_CHECK(cudaGetLastError());

        // 同时，在 transfer stream 上预取下一个 batch（与 kernel 重叠）
        if (b + 1 < NUM_BATCHES) {
            int nxt_n = N - (b + 1) * BATCH_SIZE;
            if (nxt_n > BATCH_SIZE) nxt_n = BATCH_SIZE;
            for (int i = 0; i < nxt_n; i++) h_buf[nxt][i] = 1.0f;
            CUDA_CHECK(cudaMemcpyAsync(d_in[nxt], h_buf[nxt], nxt_n * sizeof(float),
                                       cudaMemcpyHostToDevice, s_transfer));
            CUDA_CHECK(cudaEventRecord(ev_transfer_done, s_transfer));
            // 下一轮 compute 等下一个 batch 传输完
            CUDA_CHECK(cudaStreamWaitEvent(s_compute, ev_transfer_done, 0));
        }
    }

    // ❯ cudaStreamWaitEvent和cudaEventSynchronize有什么区别？                                                                                                   

    // ● 核心区别：谁在等

    //   ┌────────────┬───────────────────────┬───────────────────────────┐
    //   │            │ cudaEventSynchronize  │    cudaStreamWaitEvent    │
    //   ├────────────┼───────────────────────┼───────────────────────────┤
    //   │ 谁在等     │ CPU 线程阻塞          │ GPU stream 等，CPU 不阻塞 │
    //   ├────────────┼───────────────────────┼───────────────────────────┤
    //   │ 等待发生在 │ host 端               │ device 端                 │
    //   ├────────────┼───────────────────────┼───────────────────────────┤
    //   │ 调用后 CPU │ 卡住，直到 event 完成 │ 立刻返回，继续执行        │
    //   └────────────┴───────────────────────┴───────────────────────────┘

    //   ---
    //   cudaEventSynchronize：CPU 等 GPU

    //   CPU                          GPU
    //    │                            │
    //    │  cudaEventRecord(ev, s)    │
    //    │ ─────────────────────────→ s: [... | record ev]
    //    │                            │
    //    │  cudaEventSynchronize(ev)  │
    //    │  ←─── CPU 阻塞在这里 ────  │  s 执行到 record ev，写入时间戳
    //    │  ─── CPU 解除阻塞 ────────→ │
    //    │  （继续执行）               │

    //   CPU 被阻塞，直到 event 被 GPU 标记为完成才继续。
    //   典型用途：读取计时结果前，确保 ev_stop 时间戳已写入。

    //   ---
    //   cudaStreamWaitEvent：GPU stream 等另一个 GPU stream

    //   CPU                          GPU
    //    │                            s_transfer: [H2D | record ev]
    //    │  cudaStreamWaitEvent       s_compute:  [wait ev | kernel]
    //    │  (s_compute, ev, 0)        │
    //    │ ─────────────────────────→ s_compute 内部插入一个屏障：
    //    │  CPU 立刻返回               等 ev 完成后才执行后续 kernel
    //    │  （继续提交其他工作）        │
    //    │                            s_transfer 完成 H2D，标记 ev
    //    │                                  ↓
    //    │                            s_compute 屏障解除，开始 kernel

    //   CPU 不等，只是告诉 GPU："s_compute 里后续的操作，要等 ev 完成之后才能执行"。
    //   典型用途：pipeline 中跨 stream 的依赖，transfer 完成后 compute 才能开始。

    //   ---
    //   结合本文件的 pipeline 说明

    //   // transfer stream 传完 batch[0]，打一个 event
    //   cudaEventRecord(ev_transfer_done, s_transfer);

    //   // 告诉 compute stream：等 ev 完成再执行后面的 kernel
    //   // CPU 不阻塞，继续提交循环里的其他命令
    //   cudaStreamWaitEvent(s_compute, ev_transfer_done, 0);

    //   // ... 循环结束后，CPU 才真正等待所有 stream 完成
    //   cudaStreamSynchronize(s_compute);
    //   cudaStreamSynchronize(s_transfer);

    //   如果这里用 cudaEventSynchronize，CPU 每轮循环都会被阻塞一次，compute 和 transfer 就无法并行，pipeline 的重叠效果完全丧失。

    // 等所有工作完成
    CUDA_CHECK(cudaStreamSynchronize(s_compute));
    CUDA_CHECK(cudaStreamSynchronize(s_transfer));

    printf("  double-buffer pipeline: %d batches processed\n", NUM_BATCHES);
    printf("  [PASS] double-buffer pipeline completed\n");

    CUDA_CHECK(cudaFreeHost(h_ping));
    CUDA_CHECK(cudaFreeHost(h_pong));
    CUDA_CHECK(cudaFree(d_ping_in));
    CUDA_CHECK(cudaFree(d_pong_in));
    CUDA_CHECK(cudaFree(d_ping_out));
    CUDA_CHECK(cudaFree(d_pong_out));
    CUDA_CHECK(cudaEventDestroy(ev_transfer_done));
    CUDA_CHECK(cudaStreamDestroy(s_compute));
    CUDA_CHECK(cudaStreamDestroy(s_transfer));
}


// ══════════════
// main
// ══════════════
int main()
{
    // 打印设备信息
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("canMapHostMemory: %s\n",
           prop.canMapHostMemory ? "yes" : "no");
    printf("concurrentKernels: %s\n",
           prop.concurrentKernels ? "yes" : "no");
    printf("asyncEngineCount: %d  (1=H2D or D2H, 2=H2D and D2H concurrent)\n",
           prop.asyncEngineCount);

    demo_cudaMallocHost();
    demo_writeCombined();
    demo_mapped();
    demo_portable();
    demo_hostRegister();
    demo_bidirectional();
    demo_doubleBuffer();

    printf("\nAll demos completed.\n");
    return 0;
}
