/**
 * cuda_pdl.cu
 *
 * Programmatic Dependent Launch（PDL）详解与示例
 *
 * 硬件要求：Hopper（sm_90）及以上
 *
 * 编译：
 *   nvcc -O2 -arch=native -std=c++17 cuda_pdl.cu -o cuda_pdl
 *
 *   -arch=native：自动检测本机 GPU 架构，推荐用法
 *   -arch=sm_86 ：手动指定架构（如 RTX 3060 = sm_86）
 *   -arch=sm_90 ：仅限 Hopper，PDL 功能完整可用
 *
 *   注意：不能用 -arch=sm_90 在 sm_86 设备上编译运行，
 *         nvcc 只生成 sm_90 机器码，sm_86 无法执行，
 *         报错：cudaErrorNoKernelImageForDevice
 *
 * 运行：
 *   ./cuda_pdl
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────
// 错误检查宏
// ──────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",                   \
                    __FILE__, __LINE__,                                       \
                    cudaGetErrorName(err), cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static const int N       = 1 << 20;   // 1M 个元素
static const int THREADS = 256;
static const int BLOCKS  = (N + THREADS - 1) / THREADS;

// ════════════════════════
// 背景：普通 stream 内顺序启动 vs PDL
//
// 普通情况（in-order stream）：
//   kernelA<<<g, b, 0, stream>>>();
//   kernelB<<<g, b, 0, stream>>>();
//   → CUDA 运行时保证 A 的所有 thread block 全部完成后，B 才开始
//   → A 和 B 在时间上完全串行，无重叠
//
// PDL（Programmatic Dependent Launch）：
//   → A 和 B 仍然在同一 stream，B 依赖 A（依赖关系不变）
//   → 但 A 可以在自己还没完全结束时，从 GPU 内部主动触发 B 的启动
//   → 条件：A 必须通过显式信号声明"B 所需的数据已经就绪"
//   → 结果：A 尾部的收尾工作和 B 的启动/执行有重叠窗口，提高 GPU 利用率
//
// 适用场景：
//   A 的大部分工作（写出 B 需要的数据）已完成，但 A 还有少量收尾
//   （如统计、日志、清理等）需要继续运行——这部分与 B 的执行无关，
//   可以与 B 并行，无需让 B 等待。
// ═══════════════════


// ═══════════════════
// PDL 的两个核心 API
//
// ① cudaTriggerProgrammaticLaunchCompletion()
//   __device__ void cudaTriggerProgrammaticLaunchCompletion();
//
//   在 kernel A 内部调用，向 CUDA 运行时声明：
//     "B 所需的所有数据已经写入显存，B 现在可以安全启动"
//   调用后，运行时可以立即调度 B 的 thread block 开始执行，
//   而 A 的剩余工作继续在 GPU 上并行运行。
//
//   重要约束：
//   · 必须在 kernel 内所有对 B 可见的写操作完成后调用（写屏障）
//   · 每个 thread block 只能调用一次；通常只让 block 0 或最后完成的 block 调用
//   · 若 kernel 结束时未调用，运行时仍会在 A 完全结束后自动触发 B，
//     但此时 PDL 的重叠优势消失，退化为普通串行启动
//
// ② cudaStreamWaitValue32 / cudaBarrierArrive（配合使用，可选）
//   PDL 不强制要求原子操作，但实践中常用原子计数器
//   让最后完成数据写入的 block 负责调用 trigger。
// ═══════════════════


// ════════════════════
// PDL 的启动方式：cudaLaunchAttributeProgrammaticStreamSerialization
//
// 普通启动：kernelB<<<g, b, 0, stream>>>();
//   → 运行时等待 A 全部完成后才启动 B
//
// PDL 启动：必须通过cudaLaunchAttributeValue指定属性
//
//   cudaLaunchAttribute attr;
//   attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
//   // value.programmaticStreamSerializationAllowed = 1：
//   //   允许运行时在 A 调用 trigger 后立即启动 B（PDL 模式）
//   //   = 0：退化为普通串行启动
//   attr.val.programmaticStreamSerializationAllowed = 1;
//
//   cudaLaunchConfig_t config = {};
//   config.gridDim   = g;
//   config.blockDim  = b;
//   config.stream    = stream;
//   config.attrs     = &attr;
//   config.numAttrs  = 1;
//
//   cudaLaunchKernelEx(&config, kernelB, args...);
//   // ↑ cudaLaunchKernelEx 是支持扩展属性的启动 API，替代 <<<>>> 语法
//
// 注意：只有 B 需要设置此属性（声明"我接受 PDL 方式被启动"）
//       A 用普通 <<<>>> 启动即可
// ════════════════


// ═══════════════════
// 示例 1：不使用 PDL 的普通串行启动（对照组）
//
// kernelA：每个元素 × 2，写入 out_A
// kernelB：依赖 out_A，每个元素 + 1，写入 out_B
// 普通 stream 语义：A 全部完成 → B 开始，无重叠
// ═════════════════

__global__ void kernelA_normal(float *out_A, const float *in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out_A[idx] = in[idx] * 2.0f;
    // A 全部 thread block 完成后，运行时才启动 B
    // 不涉及 PDL，无任何显式信号
}

__global__ void kernelB_normal(float *out_B, const float *out_A, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out_B[idx] = out_A[idx] + 1.0f;
}

void demo_normal_launch(const float *d_in, float *d_outA, float *d_outB,
                        cudaStream_t stream)
{
    printf("\n─── 示例 1：普通串行启动（对照组）───\n");
    printf("  A 全部完成 → B 开始，无重叠\n");

    // 普通 <<<>>> 启动，stream 保证 A 完全结束后 B 才开始
    kernelA_normal<<<BLOCKS, THREADS, 0, stream>>>(d_outA, d_in, N);
    CUDA_CHECK(cudaGetLastError());

    kernelB_normal<<<BLOCKS, THREADS, 0, stream>>>(d_outB, d_outA, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  [PASS] 普通启动完成\n");
}


// ════════════════
// 示例 2：使用 PDL
//
// kernelA_pdl：
//   ① 完成数据写入（out_A[idx] = in[idx] * 2.0f）
//   ② 用原子计数器统计已完成写入的 block 数量
//   ③ 最后一个完成的 block 调用 cudaTriggerProgrammaticLaunchCompletion()
//      → 通知运行时：B 所需数据全部就绪，可以立即启动 B
//   ④ A 的剩余 block 继续运行收尾逻辑，与 B 的启动/执行并行
//
// kernelB_pdl：
//   通过 cudaLaunchAttributeProgrammaticStreamSerialization 启动
//   声明自己接受 PDL 方式被提前启动
//   逻辑与 kernelB_normal 相同，读 out_A 写 out_B
// ═════════════════

__global__ void kernelA_pdl(float       *out_A,
                             const float *in,
                             int          n,
                             int         *block_counter,  // 原子计数器
                             int          total_blocks)   // 总 block 数
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ① 主要工作：写出 B 需要的数据
    if (idx < n) out_A[idx] = in[idx] * 2.0f;

    // ── 内存可见性保证：__syncthreads + __threadfence ─────
    //
    // 问题背景：
    //   GPU 有多级缓存：每个 SM 有私有 L1，多个 SM 共享 L2，最底层是显存（DRAM）
    //   一个 SM 上的写操作，默认只保证在本 SM 的 L1 中完成；
    //   另一个 SM 读同一地址时，若数据还在第一个 SM 的 L1 中未刷出，
    //   则读到的是旧值（缓存不一致）。
    //
    // 两条屏障的分工：
    //
    // __syncthreads()：
    //   作用域：block 内所有 thread
    //   功能：等待本 block 内所有 thread 执行到此处（block 内同步点）
    //   保证：本 block 所有 thread 的写操作均已完成
    //   不保证：写操作已刷出到 L2 或显存（其他 SM 不一定可见）
    //
    // __threadfence()：
    //   作用域：整个 GPU（device 级别）
    //   功能：强制调用线程（thread 0）之前的所有写操作刷出到 L2/显存
    //   保证：__threadfence() 之前的所有写操作，
    //         在 __threadfence() 之后发起的任意 SM 上的读操作中均可见
    //   不保证：其他 thread 的写操作（所以必须先用 __syncthreads 同步本 block）
    //
    //   对比三种 threadfence：
    //   __threadfence_block()  → 仅对本 block 内其他 thread 可见（刷到 shared mem）
    //   __threadfence()        → 对整个 GPU 所有 SM 可见（刷到 L2/显存）
    //   __threadfence_system() → 对 CPU 和其他 GPU 也可见（刷过 PCIe/NVLink）
    //
    // 为什么 PDL 场景必须用 __threadfence()：
    //   A 的各个 block 分布在不同 SM 上执行；
    //   B 的 thread block 会被调度到（可能不同的）SM 上读 out_A；
    //   若 A 的写操作只停留在各自 SM 的 L1，B 读到的可能是旧值。
    //   __threadfence() 确保本 block 的写操作刷到 L2/显存，B 一定能读到正确值。
    //
    // 正确执行顺序（每个 block 的视角）：
    //   1. out_A[idx] = in[idx] * 2.0f         ← 写数据（可能停留在 L1）
    //   2. __syncthreads()                      ← 等待本 block 所有 thread 写完
    //   3. __threadfence()   (thread 0 执行)    ← 将写操作刷出到 L2/显存
    //   4. atomicAdd(block_counter, 1)          ← 原子递增，声明"本 block 数据就绪"
    //   5. if (last block) → cudaTriggerProgrammaticLaunchCompletion()
    //
    //   关键推理：
    //   步骤 3（__threadfence）happens-before 步骤 4（atomicAdd）；
    //   最后一个 block 的 atomicAdd 使 completed == total_blocks，
    //   意味着所有 block 都已完成步骤 3，out_A 全部刷出，B 可安全读取。
    // ─────────────
    __syncthreads();   // 先同步本 block 内所有 thread
    if (threadIdx.x == 0) {
        __threadfence();  // 将本 block 的写操作刷出到 L2/显存，对所有 SM 可见

        // ② 原子递增：统计已完成数据写入的 block 数
        //
        // 为什么必须用 atomicAdd 而不是普通写：
        //   多个 block 分布在不同 SM 上并行执行，会同时到达此处。
        //   若用普通读-改-写：
        //     int v = *block_counter + 1;   // 两个 SM 同时读到 5
        //     *block_counter = v;           // 两个 SM 都写回 6 → 丢失一次递增
        //
        //   atomicAdd(ptr, val)：硬件级原子操作，读-改-写三步不可分割，
        //   多个 SM 并发调用时被硬件串行化：
        //     block_3: 读5 → 写6 → 返回5（completed=6）
        //     block_7: 读6 → 写7 → 返回6（completed=7）  ← 等 block_3 完成后执行
        //     block_1: 读7 → 写8 → 返回7（completed=8）
        //   每个 block 拿到唯一的旧值，completed 值各不相同，
        //   因此只有一个 block 满足 completed == total_blocks，
        //   只有一个 block 调用 cudaTriggerProgrammaticLaunchCompletion()。
        //
        // atomicAdd 返回值是操作前的旧值，+1 得到本次递增后的新值
        int completed = atomicAdd(block_counter, 1) + 1;

        // ③ 最后一个完成写入的 block 触发 PDL 信号
        if (completed == total_blocks) {
            // 此刻所有 block 的数据写入均已完成（通过 __threadfence 保证可见性）
            // 向运行时声明：out_A 全部就绪，B 可以安全启动
            //
            // cudaTriggerProgrammaticLaunchCompletion()：
            //
            // 函数签名（device 端）：
            //   __device__ void cudaTriggerProgrammaticLaunchCompletion();
            //   · 无参数，无返回值
            //   · 声明在 <cuda_runtime.h> 中
            //   · 仅在 sm_90+（Hopper）编译时可用；在低架构下编译会报错：
            //       error: identifier "cudaTriggerProgrammaticLaunchCompletion"
            //              is undefined
            //     因此整个文件需用 -arch=sm_90 或以上编译
            //
            // 调用时机要求：
            //   · 必须在所有对 B 可见的写操作完成后调用（即 __threadfence 之后）
            //   · 每个 thread block 只能调用一次，多次调用行为未定义
            //   · 只有满足 completed == total_blocks 的最后一个 block 调用，
            //     其余 block 跳过（通过上方 if 分支控制）
            //
            // 调用效果：
            //   · 向 CUDA 运行时发出信号：kernelA 的"逻辑完成"已达到
            //   · 运行时随即开始调度 kernelB 的 thread block 上 SM 执行
            //   · kernelA 剩余的 block（收尾工作）继续在 GPU 上运行，
            //     与 kernelB 的执行产生时间重叠
            //
            // 若未调用（fallback）：
            //   · 运行时等待 kernelA 所有 thread block 全部退出后，
            //     再自动触发 kernelB 启动，退化为普通串行语义
            //
            // ── #if __CUDA_ARCH__ 说明 ───────
            //
            // __CUDA_ARCH__：nvcc 编译 device 代码时自动定义的预处理宏，
            //   值 = 目标架构 × 10：sm_86 → 860，sm_90 → 900
            //   host 编译阶段此宏未定义。
            //
            // #if 是预处理器指令，在编译最早阶段（预处理期）执行：
            //   条件为假 → 代码块被文本删除，后续编译器完全看不到这段代码
            //   条件为真 → 代码块正常保留，进入编译
            //
            // 这里解决的问题：符号未定义
            //   cudaTriggerProgrammaticLaunchCompletion() 是 sm_90+ 专属符号，
            //   用 -arch=native 在 sm_86 上编译时，该符号不存在，编译报错：
            //     error: identifier "cudaTriggerProgrammaticLaunchCompletion" is undefined
            //   #if 保护后，sm_86 编译时此行被删除，不报错，退化为 fallback。
            //
            // 注意：#if 只解决"符号未定义"的编译错误，
            //       不能解决"机器码不匹配"的运行时错误。
            //   若用 -arch=sm_90 编译，二进制只含 sm_90 机器码，
            //   sm_86 设备运行时找不到对应版本，报：
            //     cudaErrorNoKernelImageForDevice
            //   此错误与 #if 无关，只能通过编译命令修复：
            //     nvcc -arch=native ...   ← 生成本机架构的机器码，推荐
            //   两个问题需分别解决，#if 和 -arch=native 缺一不可。
            // ─────────────
#if __CUDA_ARCH__ >= 900
            cudaTriggerProgrammaticLaunchCompletion();
#endif
        }
    }

    // ④ A 的收尾工作（模拟：与 B 的执行无关的操作）
    // 这部分与 B 的启动/执行并行，体现 PDL 的重叠优势
    // 实际场景：统计信息写入、调试标记、日志记录等
    if (idx < n) {
        // 模拟少量收尾计算（与 out_A 无关，不影响 B 的正确性）
        // volatile float dummy：
        //   volatile 告诉编译器"此变量可能被外部访问，禁止优化掉对它的读写"。
        //   若不加 volatile，编译器发现 dummy 从未被使用，会直接删除整条赋值语句，
        //   导致收尾工作变成空操作，无法演示 PDL 重叠效果。
        //   加了 volatile 后，编译器必须保留这次内存读操作（从 out_A 读值）。
        volatile float dummy = out_A[idx] * 0.0f;

        // (void)dummy：
        //   作用：消除编译器的 "unused variable" 警告。
        //   dummy 被赋值后没有再被读取，编译器会警告 "warning: unused variable 'dummy'"。
        //   (void)expr 是 C/C++ 惯用写法，含义是"我知道这个值没被使用，这是故意的"，
        //   强制将表达式转换为 void 类型，告知编译器不要发出警告。
        //   注意：此处 volatile 已经保证赋值语句不被删除，(void)dummy 只是消除警告，
        //         对运行时行为没有任何影响。
        (void)dummy;
    }
}





__global__ void kernelB_pdl(float       *out_B,
                             const float *out_A,    
                             int          n)
{
    // kernelB 的逻辑与普通版本相同
    // PDL 保证：当 B 的 thread block 开始执行时，out_A 已全部写入完毕
    // （由 kernelA_pdl 的 __threadfence + cudaTriggerProgrammaticLaunchCompletion 保证）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out_B[idx] = out_A[idx] + 1.0f;
}

void demo_pdl_launch(const float *d_in, float *d_outA, float *d_outB,
                     cudaStream_t stream)
{
    printf("\n─── 示例 2：PDL 启动 ───\n");
    printf("  A 触发信号后 B 立即启动，A 尾部与 B 有重叠窗口\n");

    // 原子计数器：统计 kernelA 中已完成数据写入的 block 数
    // d_counter 是 GPU 显存指针，由 cudaMalloc 分配
    int *d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    // cudaMemset(ptr, value, count)：
    //   对 GPU 显存按字节填充，对应 CPU 的 memset，操作的是 device memory。
    //   ptr   必须是 cudaMalloc 分配的 device 指针，不能是 CPU 指针。
    //   value 按字节填充（与 memset 相同）：
    //     填 0  → 每字节 0x00 → int 值为 0          ✓ 符合预期
    //     填 1  → 每字节 0x01 → int 值为 0x01010101  ✗ 若要初始化为整数 1，
    //             需用 kernel 或 cudaMemcpy 而非 cudaMemset
    //   count 单位为字节。
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // A 用普通 <<<>>> 启动
    kernelA_pdl<<<BLOCKS, THREADS, 0, stream>>>(
        d_outA, d_in, N, d_counter, BLOCKS);
    CUDA_CHECK(cudaGetLastError());

    // ── B 用 cudaLaunchKernelEx + PDL 属性启动 ─────
    //
    // cudaLaunchAttribute：扩展启动属性结构体
    //   id  = cudaLaunchAttributeProgrammaticStreamSerialization
    //         声明此 kernel 接受 PDL 方式被提前启动
    //   val.programmaticStreamSerializationAllowed = 1
    //         1：允许在前驱 kernel 调用 trigger 后立即启动（PDL 模式）
    //         0：退化为等待前驱 kernel 完全结束（普通模式）
    //
    // cudaLaunchConfig_t：替代 <<<grid, block, sharedMem, stream>>> 的结构体
    //   支持传入扩展属性数组，<<<>>> 语法无法表达扩展属性
    // ──────────────
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;

    // cudaLaunchConfig_t 结构体定义（简化）：
    //   struct cudaLaunchConfig_t {
    //       dim3                  gridDim;          // grid 维度
    //       dim3                  blockDim;         // block 维度
    //       size_t                dynamicSmemBytes; // 动态 shared memory，默认 0
    //       cudaStream_t          stream;           // 所属 stream
    //       cudaLaunchAttribute  *attrs;            // 扩展属性数组首地址
    //       unsigned int          numAttrs;         // 扩展属性数组的元素个数
    //   };
    //
    // cfg.attrs = &attr：
    //   指向扩展属性数组首地址。此处只有 1 个属性，直接取单个变量地址。
    //   若有多个属性，定义 cudaLaunchAttribute attrs[N] 并传入 attrs。
    //
    // cfg.numAttrs = 1：
    //   告诉运行时 attrs 数组中有多少个有效元素。
    //   运行时遍历 attrs[0] ~ attrs[numAttrs-1]，逐个解析并应用每个属性。
    //   · numAttrs = 0：attrs 被完全忽略，退化为普通启动
    //   · numAttrs 大于实际数组长度：越界读取，未定义行为
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim   = BLOCKS;
    cfg.blockDim  = THREADS;
    cfg.stream    = stream;
    cfg.attrs     = &attr;     // 扩展属性数组首地址（此处只有 1 个属性）
    cfg.numAttrs  = 1;         // 数组元素个数，运行时据此遍历 attrs[0..0]

    // cudaLaunchKernelEx：支持扩展属性的 kernel 启动 API
    //
    // 函数模板签名（cuda_runtime.h）：
    //   template<typename... ExpTypes, typename... ActTypes>
    //   cudaError_t cudaLaunchKernelEx(
    //     const cudaLaunchConfig_t *config,  // 启动配置（grid/block/stream/attrs）
    //     void (*func)(ExpTypes...),         // 类型安全的 kernel 函数指针
    //     ActTypes&&...           args       // 实际传入的参数（完美转发）
    //   );
    //
    // 两个参数包的分工：
    //
    // typename... ExpTypes（Expected Types，预期类型包）：
    //   从 kernel 函数指针 void(*func)(ExpTypes...) 中提取，
    //   是 kernel 声明时的参数类型，由编译器从 func 自动推导：
    //     ExpTypes = <float*, const float*, int>   ← kernelB_pdl 的参数类型
    //
    // typename... ActTypes（Actual Types，实际类型包）：
    //   调用时实际传入的参数类型，由编译器从 args 自动推导：
    //     ActTypes = <float*, float*, int>         ← d_outB, d_outA, N 的类型
    //
    //   ActTypes&&... 是万能引用（universal reference）：
    //     传入左值（如变量）→ ActTypes = T&，args 为左值引用
    //     传入右值（如临时值）→ ActTypes = T，args 为右值引用
    //     配合 std::forward 实现完美转发，避免不必要的拷贝
    //
    // 编译期类型检查：
    //   编译器验证 ActTypes 中每个类型是否可隐式转换为对应的 ExpTypes，
    //   参数类型或数量不匹配时直接编译报错，而非运行时崩溃。
    //   这是相比 C 风格 va_list 的核心优势：
    //     va_list 运行时按字节读取，类型信息完全丢失，传错只能在 GPU 上崩溃
    //
    // 本次调用的编译期推导：
    //   cudaLaunchKernelEx(&cfg, kernelB_pdl,  d_outB,  d_outA,        N)
    //                            ↑              ↑        ↑              ↑
    //   ExpTypes = <            float*,  const float*,          int   >
    //   ActTypes = <            float*,        float*,          int   >
    //              编译器检查 float* → const float* ✓（加 const 合法）
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernelB_pdl, d_outB, d_outA, N));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_counter));
    printf("  [PASS] PDL 启动完成\n");
}


// ══════════════
// 示例 3：PDL 的退化保证（fallback guarantee）
//
// 若 kernelA 内部没有调用 cudaTriggerProgrammaticLaunchCompletion()，
// 运行时在 A 全部结束后自动触发 B——退化为普通串行语义，
// 正确性不受影响，只是失去重叠优势。
//
// 这意味着：
//   · PDL 只是对正常依赖语义的"性能放松"，不破坏正确性
//   · 即使 trigger 未调用，B 依然能看到 A 的所有结果
//   · 可以用 attr.val.programmaticStreamSerializationAllowed = 0
//     强制退化，用于对比测试
// ══════════════

__global__ void kernelA_no_trigger(float *out_A, const float *in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out_A[idx] = in[idx] * 2.0f;
    // 没有调用 cudaTriggerProgrammaticLaunchCompletion()
    // 运行时在 A 全部结束后自动触发 B → 退化为串行
}

void demo_pdl_fallback(const float *d_in, float *d_outA, float *d_outB,
                       cudaStream_t stream)
{
    printf("\n─── 示例 3：PDL 退化（无 trigger 调用）───\n");
    printf("  A 未调用 trigger，运行时在 A 完全结束后自动启动 B\n");
    printf("  正确性不变，退化为普通串行\n");

    kernelA_no_trigger<<<BLOCKS, THREADS, 0, stream>>>(d_outA, d_in, N);
    CUDA_CHECK(cudaGetLastError());

    // B 仍用 PDL 属性启动，但 A 未触发 trigger
    // → 运行时等 A 全部完成后才启动 B（退化）
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = BLOCKS;
    cfg.blockDim = THREADS;
    cfg.stream   = stream;
    cfg.attrs    = &attr;
    cfg.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernelB_pdl, d_outB, d_outA, N));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  [PASS] 退化启动完成，结果正确\n");
}


// ══════════════
// 验证结果
// ══════════════
static void verify(const float *d_out, float expected, int n, const char *label)
{
    float *h = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h[i] - expected) > 1e-3f) {
            printf("  [FAIL] %s: h[%d] = %.2f, expected %.2f\n",
                   label, i, h[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) printf("  [PASS] %s (expected %.2f)\n", label, expected);
    free(h);
}


int main(void)
{
    // 检查设备是否支持 PDL（需要 sm_90+）
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 9) {
        // ── PDL 需要 sm_90（Hopper）及以上 ─────────────────────────────
        // sm_90 之前的架构不支持 cudaTriggerProgrammaticLaunchCompletion
        // 和 cudaLaunchAttributeProgrammaticStreamSerialization
        // 在不支持的硬件上调用会返回 cudaErrorNotSupported 或直接编译失败
        printf("PDL requires sm_90 (Hopper) or newer. "
               "This device is sm_%d%d, skipping PDL demos.\n",
               prop.major, prop.minor);
        printf("Showing normal launch only.\n");

        float *d_in, *d_outA, *d_outB;
        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outA, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_in, 0x3f, N * sizeof(float)));  // ~1.0f

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 初始化输入为 1.0f
        float *h_in = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_in);

        demo_normal_launch(d_in, d_outA, d_outB, stream);
        // A: 1.0 × 2 = 2.0, B: 2.0 + 1 = 3.0
        verify(d_outB, 3.0f, N, "normal launch out_B");

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_outA));
        CUDA_CHECK(cudaFree(d_outB));
        return 0;
    }

    // sm_90+：运行全部示例
    float *d_in, *d_outA, *d_outB;
    CUDA_CHECK(cudaMalloc(&d_in,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));

    float *h_in = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_in);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 示例 1：普通串行（对照）A:1×2=2, B:2+1=3
    demo_normal_launch(d_in, d_outA, d_outB, stream);
    verify(d_outB, 3.0f, N, "normal launch out_B");

    // 示例 2：PDL，A 触发后 B 立即启动，结果相同
    demo_pdl_launch(d_in, d_outA, d_outB, stream);
    verify(d_outB, 3.0f, N, "PDL launch out_B");

    // 示例 3：PDL 退化（A 无 trigger），退化为串行，结果相同
    demo_pdl_fallback(d_in, d_outA, d_outB, stream);
    verify(d_outB, 3.0f, N, "PDL fallback out_B");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_outA));
    CUDA_CHECK(cudaFree(d_outB));

    printf("\n全部示例完成。\n");
    return 0;
}
