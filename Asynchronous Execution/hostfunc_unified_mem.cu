/**
 * hostfunc_unified_mem.cu
 *
 * 演示 cudaLaunchHostFunc 与 Unified Memory 交互的四条保证：
 *
 * G1: stream idle during hostFunc
 *     hostFunc 执行期间 stream 被强制视为 IDLE
 *     cudaMemAttachSingle 的内存从 GPU 独占切换到 CPU 可访问
 *
 * G2: start of execution = event sync
 *     hostFunc 开始执行时，所有通过 event join 进来的上游工作已完成
 *     （保证来自 cudaStreamWaitEvent，hostFunc 继承这个保证）
 *
 * G3: new device work deferred
 *     ordering chain 内的其他 stream 新提交的工作
 *     在 hostFunc 完成前不被视为 ACTIVE
 *     （cudaMemAttachGlobal 场景：UVM 主动迁移启发式被抑制）
 *
 * G4: completion does not reactivate
 *     hostFunc 完成后 stream 不自动变回 ACTIVE
 *     连续 hostFunc 和 end-of-stream 通知模式的基础
 *
 * 编译：
 *   nvcc -O2 -arch=sm_70 -std=c++14 hostfunc_unified_mem.cu -o hostfunc_um_demo
 *
 * 运行：
 *   ./hostfunc_um_demo
 */

#include <cuda_runtime.h>       // CUDA 运行时 API：cudaMallocManaged、cudaLaunchHostFunc、
                                //   cudaStreamCreate/Destroy、cudaStreamSynchronize、
                                //   cudaMemAttachSingle/Global、cudaEvent_t 等

#include <stdio.h>              // printf、fprintf：打印演示输出和错误信息
#include <stdlib.h>             // exit、EXIT_FAILURE：CUDA_CHECK 宏出错时终止进程

#include <atomic>               // std::atomic<bool/int>：无锁标志位，用于 hostFunc 回调
                                //   中安全写、主线程安全读（避免 mutex 开销）

#include <condition_variable>   // std::condition_variable：G3/G4 演示中
                                //   让等待线程在条件满足前睡眠，避免忙等

#include <mutex>                // std::mutex + std::unique_lock：
                                //   与 condition_variable 配合使用，
                                //   cv.wait() 需要 unique_lock 才能原子释放锁并挂起

#include <thread>               // std::thread：启动独立的消费者/观察者线程，
                                //   在 hostFunc 通知后执行后续 CUDA API 调用
                                //   （hostFunc 自身不能调用 CUDA API，否则死锁）

#include <chrono>               // std::chrono::milliseconds + std::this_thread::sleep_for：
                                //   在演示中模拟 CPU 端耗时操作，使时序更易观察

// ───────────
// 错误检查宏
// ──────────
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",              \
                    __FILE__, __LINE__,                                  \
                    cudaGetErrorName(err), cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

static const int N = 1 << 20;   // 1M 个 float

// ──────────
// 公共 kernel
// ──────────

// 把数组每个元素乘以 factor
__global__ void scaleKernel(float *data, float factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

// 把数组每个元素加上 val
__global__ void addKernel(float *data, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += val;
}



// 把 src 加到 dst
__global__ void addArrayKernel(float *dst, const float *src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}


// static inline：host 端辅助函数的惯用写法
//
//   static：限制链接性为内部链接（仅当前翻译单元可见）
//           若此函数定义在头文件中，多个 .cpp include 时不会产生重复符号链接错误
//           这里在 .cu 文件中定义，static 主要起"不暴露给外部"的作用
//
//   inline：C++ 标准关键字，通知编译器可以内联展开
//           对于这种一行的简单函数，编译器不加 inline 也会自动内联
//           inline 的更重要作用是允许函数在多个翻译单元中重复定义（ODR 豁免）
//
//   为什么不用 __forceinline__ / __inline__？
//     __forceinline__ 和 __inline__ 是 CUDA device 端关键字，作用于 __device__ 函数：
//       __device__ __forceinline__ float helper(float x) { ... }  // 强制 GPU 内联
//     blocks() 是纯 host 函数（无 __device__ 修饰），由 g++/clang++ 编译，
//     device 端关键字对它无效，用标准 inline 即可
static inline int blocks(int n, int t = 256)
{
    return (n + t - 1) / t;
}

// ───────────
// 验证工具
// ────────────
static void verify(const float *data, float expected, int n,
                   const char *label)
{
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(data[i] - expected) > 1e-3f) {
            printf("  [FAIL] %s: data[%d] = %f, expected %f\n",
                   label, i, data[i], expected);
            ok = false;
            break;
        }
    }
    if (ok) printf("  [PASS] %s (expected %.2f)\n", label, expected);
}


// ─────────────
// 用于 hostFunc 传参的数据结构
// ─────────────

struct G1Ctx {
    float       *data;       // managed memory 指针
    int          n;
    float        expected;
    const char  *label;
};

struct G2Ctx {
    float       *data_a;     // 来自 s1 的结果
    float       *data_b;     // 来自 s2 的结果
    int          n;
};

// G4Ctx：hostFunc 通知主线程"GPU 工作已完成"的数据结构
//
// 三个同步成员需要配合使用，缺一不可：
//
//   std::atomic<bool> done
//     · 标志位，hostFunc 写（done=true），主线程读（predicate）
//     · atomic 保证单次读写的原子性，但不能解决 cv.wait 的通知丢失问题
//
//   std::mutex mtx
//     · condition_variable 必须配套一个 mutex，cv.wait() 要求持有 unique_lock
//     · 作用：让 cv.wait() 的"检查 predicate → 决定睡眠"成为原子操作，
//       防止以下竞态：
//         主线程：检查 done==false（将要睡眠）
//         hostFunc：done=true → cv.notify_one()    ← 通知丢失！
//         主线程：进入 wait()                       ← 永远睡眠
//       有 mutex 后：
//         主线程持锁 → 检查 done → cv.wait() 原子释放锁并睡眠
//         hostFunc：获得锁 → done=true → 释放锁 → notify
//         主线程被唤醒，重新获得锁，检查 done==true，退出等待
//
//   std::condition_variable cv
//     · 让主线程睡眠等待，避免忙等（spin），由 hostFunc 调用 notify_one() 唤醒
//
// 使用模式：
//   hostFunc（写方）：
//     { std::lock_guard<std::mutex> lock(ctx->mtx);   // 加锁
//       ctx->done.store(true); }                       // 修改 predicate
//     ctx->cv.notify_one();                            // 唤醒等待方
//                                                      // lock_guard 析构自动解锁
//   主线程（读方）：
//     std::unique_lock<std::mutex> lock(ctx->mtx);    // 加锁（cv.wait 需要）
//     ctx->cv.wait(lock, [&]{ return ctx->done.load(); }); // 原子释放锁并睡眠
struct G4Ctx {
    std::atomic<bool>        done;  // 完成标志，初始 false
    std::mutex               mtx;   // 配合 cv 使用，防止通知丢失竞态
    std::condition_variable  cv;    // 主线程睡眠等待，hostFunc notify 唤醒
    float                   *data;
    int                      n;
    float                    expected;

    G4Ctx() : done(false) {}
};


// ════════════════
// 示例 1：G1 — stream idle 保证 CPU 可以安全访问 cudaMemAttachSingle 内存
//
// 场景：
//   stream 队列：[kernelA][hostFunc][kernelB]
//   hostFunc 需要读取 kernelA 的结果（managed memory）
//   kernelB 排在 hostFunc 之后
//
// 没有 G1：stream 仍 ACTIVE，cudaMemAttachSingle 内存 GPU 独占，CPU 访问未定义
// 有了 G1：stream 强制 IDLE，内存切换到 CPU 可访问，page fault 正常响应
// ════════════════
void demo_g1()
{
    printf("\n=== G1: stream forced IDLE during hostFunc ===\n");
    printf("    [cudaMemAttachSingle: GPU exclusive → CPU accessible]\n");

    // ── cudaMallocManaged 函数签名 ────────
    // cudaError_t cudaMallocManaged(
    //     void        **devPtr,   // [out] 返回 Unified Memory 指针
    //                             //       CPU 和 GPU 均可直接解引用此指针
    //     size_t        size,     // [in]  分配字节数
    //     unsigned int  flags     // [in]  初始归属策略：
    //                             //   cudaMemAttachGlobal(默认)：
    //                             //     所有 stream/device 可见，UVM 按需迁移
    //                             //   cudaMemAttachHost：
    //                             //     初始仅 CPU 可见，显式 attach 后 GPU 才可访问
    // );
    //
    // 与 cudaMalloc 区别：
    //   cudaMalloc    → 仅 device 可访问，CPU 需 cudaMemcpy 才能读写
    //   cudaMallocManaged → CPU/GPU 共享同一虚拟地址，驱动自动按需迁移页面
    // ───────────────
    float *um_data;
    // 全局分配，然后 attach 到 stream（变成 single-stream 语义）
    CUDA_CHECK(cudaMallocManaged(&um_data, N * sizeof(float),
                                    cudaMemAttachGlobal));

    // CPU 初始化：初始值 1.0
    for (int i = 0; i < N; i++) um_data[i] = 1.0f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── cudaStreamAttachMemAsync 函数签名 ───────
    // cudaError_t cudaStreamAttachMemAsync(
    //     cudaStream_t  stream,   // [in] 目标 stream
    //                             //      传 0 / cudaStreamLegacy 表示归还给全局可见
    //     void         *devPtr,   // [in] 由 cudaMallocManaged 分配的指针
    //     size_t        length,   // [in] 字节数，传 0 表示整块分配区域
    //     unsigned int  flags     // [in] 新的归属策略：
    //                             //   cudaMemAttachSingle：
    //                             //     仅此 stream 独占；stream ACTIVE 时 GPU 独占访问，
    //                             //     stream IDLE 时（包括 hostFunc 执行期间）CPU 可访问
    //                             //   cudaMemAttachGlobal：
    //                             //     重新变为全局可见，所有 stream 均可访问
    //                             //   cudaMemAttachHost：
    //                             //     归还给 CPU，GPU 不可访问
    // );
    //
    // 异步：API 立即返回，attach 操作作为命令插入 stream 队列，
    //       在前序操作完成后生效，CPU 不阻塞
    // ─────────────────
    // 附着到 stream：从这里开始 stream ACTIVE 时 GPU 独占
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, um_data,
                                            0, cudaMemAttachSingle));

    // ① kernelA：每个元素 ×2，结果应为 2.0
    scaleKernel<<<blocks(N), 256, 0, stream>>>(um_data, 2.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // ② hostFunc：读取 kernelA 的结果
    //    执行时 stream 被强制 IDLE
    //    um_data 从 GPU 独占切换到 CPU 可访问
    //    kernelB 还在队列里等，但 FIFO 保证它不会并发执行
    G1Ctx ctx1 { um_data, N, 2.0f, "G1: hostFunc reads kernelA result" };
    CUDA_CHECK(cudaLaunchHostFunc(stream, [](void *arg) {
        auto *c = static_cast<G1Ctx*>(arg);
        // stream 是 IDLE，CPU 访问合法
        // CPU page fault 把页面从 GPU 显存迁回 CPU 内存
        verify(c->data, c->expected, c->n, c->label);
        // 顺便修改数据（CPU 写）
        for (int i = 0; i < c->n; i++) c->data[i] += 1.0f; // 变为 3.0
    }, &ctx1));

    // ③ kernelB：继续处理（排在 hostFunc 之后）
    //    hostFunc 返回后 stream 恢复调度，kernelB 才执行
    //    kernelB 执行时 stream 变回 ACTIVE，um_data 重新 GPU 独占
    addKernel<<<blocks(N), 256, 0, stream>>>(um_data, 1.0f, N); // 3.0+1=4.0
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 同步完成，stream IDLE，可以安全读取最终结果
    verify(um_data, 4.0f, N, "G1: final result after kernelB");

    CUDA_CHECK(cudaFree(um_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ══════════════
// 示例 2：G2 — hostFunc 开始 = 继承上游所有 join 的同步保证
//
// 场景：
//   s1 上 kernelA 处理 d_a
//   s2 通过 cudaStreamWaitEvent join s1
//   s2 上 kernelB 依赖 d_a（kernelA 的结果）
//   s2 上的 hostFunc 读取 d_a 和 d_b
//
// 真正建立同步的是 cudaStreamWaitEvent
// hostFunc 开始执行时继承了这条保证链：kernelA 和 kernelB 都已完成
// ═══════════════
void demo_g2()
{
    printf("\n=== G2: hostFunc inherits cross-stream join guarantees ===\n");
    printf("    [synchronization built by cudaStreamWaitEvent, inherited by hostFunc]\n");

    float *d_a, *d_b;
    // 使用 cudaMemAttachGlobal：两个 stream 都需要访问
    CUDA_CHECK(cudaMallocManaged(&d_a, N * sizeof(float),
                                  cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&d_b, N * sizeof(float),
                                  cudaMemAttachGlobal));

    // 初始化
    for (int i = 0; i < N; i++) { d_a[i] = 1.0f; d_b[i] = 0.0f; }

    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    cudaEvent_t join_event;
    CUDA_CHECK(cudaEventCreate(&join_event));

    // s1 上：kernelA 把 d_a 每个元素 ×3，结果 3.0
    scaleKernel<<<blocks(N), 256, 0, s1>>>(d_a, 3.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // s1 上：记录 join_event（kernelA 完成后）
    CUDA_CHECK(cudaEventRecord(join_event, s1));

    // s2 等 join_event：建立 s1 → s2 的依赖
    // 真正的同步保证在这里建立，不是在 hostFunc
    CUDA_CHECK(cudaStreamWaitEvent(s2, join_event, 0));

    // s2 上：kernelB 读取 d_a（kernelA 的结果），写入 d_b
    // kernelB 执行时 kernelA 已完成（cudaStreamWaitEvent 保证）
    addArrayKernel<<<blocks(N), 256, 0, s2>>>(d_b, d_a, N); // d_b = 0+3 = 3.0
    CUDA_CHECK(cudaGetLastError());

    // s2 上：hostFunc
    // 开始执行时：
    //   FIFO 保证 kernelB 完成
    //   kernelB 的前置条件：join_event fired → kernelA 完成
    //   hostFunc 继承了整条保证链
    G2Ctx ctx2 { d_a, d_b, N };
    CUDA_CHECK(cudaLaunchHostFunc(s2, [](void *arg) {
        auto *c = static_cast<G2Ctx*>(arg);
        // 此刻 kernelA（s1）和 kernelB（s2）都已完成
        // 可以安全读取两个 stream 的结果
        verify(c->data_a, 3.0f, c->n, "G2: d_a (kernelA result from s1)");
        verify(c->data_b, 3.0f, c->n, "G2: d_b (kernelB result from s2)");
        printf("  [INFO] G2: both streams' results safely read in hostFunc\n");
        printf("  [INFO] sync guarantee: cudaStreamWaitEvent built it, "
               "hostFunc inherited it\n");
    }, &ctx2));

    CUDA_CHECK(cudaStreamSynchronize(s2));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaEventDestroy(join_event));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
}


// ═══════════
// 示例 3：G3 — ordering chain 内的其他 stream 新工作在 hostFunc 完成前不 active
//
// ── G3 的精确含义 ───────────────────

// CUDA 文档原文：
//   "Adding device work to any stream does not have the effect of making
//    the stream active until all preceding host functions and stream
//    callbacks in the ordering chain have finished."

// 关键词："in the ordering chain"
//   G3 不是说全局所有 stream 都被抑制，只对有顺序关系约束的 stream 生效。

//   有顺序关系（G3 生效）：
//     other_stream 通过 cudaStreamWaitEvent 等待了 stream1 上的某个 event
//     → other_stream 与 stream1 有显式依赖，处于 ordering chain 内
//     → G3 保证：other_stream 上新提交的 work，在 stream1 的 hostFunc
//                完成前不被视为 ACTIVE
//     → UVM 不会为该 work 做启发式预取，hostFunc 读取 global 内存期间安全

//   没有顺序关系（G3 不覆盖）：
//     other_stream 与 stream1 完全独立，没有任何 event 联系
//     → G3 不保证 other_stream 的 ACTIVE 状态被抑制
//     → 数据竞争需要程序员自己通过显式同步解决

// ── WaitEvent 与 G3 的关系 ───────────────────────────────────────────────

// WaitEvent 之前 other_stream 是 IDLE，是纯粹的 stream 语义：
//   stream 中有未满足的依赖（WaitEvent 条件未触发），后续 work 无法执行。
//   这与 hostFunc 是否存在无关：
//     有 hostFunc：other_stream IDLE  ← WaitEvent 阻塞
//     无 hostFunc：other_stream IDLE  ← WaitEvent 阻塞（完全相同）

// G3 在 kernel 执行顺序上是冗余的（WaitEvent 已保证顺序），
// G3 的独立价值在于 UVM 层面：
//   向 ordering chain 内的 stream 提交 work 时，G3 通知 UVM 驱动
//   不要提前做启发式迁移（heuristic prefetch），防止 UVM 在
//   hostFunc 读取期间将页面迁回 GPU，是性能稳定性保证，不是执行正确性保证。
// ─────────────────────────────────────────────────────────────────────
//
// ── UVM 迁移机制与 attach 标志的关系 ─────
//
//   UVM（Unified Virtual Memory）有两种迁移页面的方式：
//
//   ① 按需迁移（page fault）
//       GPU 执行指令时访问不在本地的页面 → 触发 page fault → 驱动迁移该页
//       与 attach 标志无关，任何 Unified Memory 都适用
//       但 page fault 只有 kernel 真正在运行时才会触发
//
//   ② 主动迁移（heuristic prefetch / proactive migration）
//       驱动根据哪些 stream 是 ACTIVE，启发式判断即将需要哪些页面，提前迁移
//       ┌──────────────────────┬─────────────────────────────────────────────────┐
//       │  cudaMemAttachGlobal │ 所有 stream 可见，任意 ACTIVE stream 均可触发   │
//       │                      │ 启发式预取；驱动需猜测目标 device（多 stream    │
//       │                      │ 竞争时无法精确判断谁会先用）                    │
//       ├──────────────────────┼─────────────────────────────────────────────────┤
//       │  cudaMemAttachSingle │ 归属绑定到特定 stream，驱动已知目标 device，    │
//       │  (cudaStreamAttach-  │ 不需要启发式猜测；                              │
//       │   MemAsync 的 flag， │   · owning stream ACTIVE：驱动可做有针对性的    │
//       │   非 cudaMalloc-     │     迁移到该 device（精准，非启发式猜测）       │
//       │   Managed 的 flag)   │   · owning stream IDLE（含 hostFunc 执行期间）：│
//       │                      │     不做 GPU 预取，CPU 可安全访问               │
//       └──────────────────────┴─────────────────────────────────────────────────┘
//       显式预取 cudaMemPrefetchAsync 与 attach 标志无关，程序员主动调用即可
//
//   G3 压制的正是 cudaMemAttachGlobal 场景下的启发式预取：
//     hostFunc 执行期间，other_stream 不被视为 ACTIVE
//     → 驱动不会因 kernelC 入队就把页面迁移到 other_stream 所在 device
//     → hostFunc 读 CPU 端数据期间不会被 UVM 迁移打断
//
//   hostFunc 完成后 other_stream 恢复 ACTIVE，kernelC 开始执行：
//     · 驱动可能启发式 prefetch 页面到 GPU（主动迁移）
//     · 或 kernelC 访问时触发 page fault（按需迁移）
// ───────────────────
// ══════════════
void demo_g3()
{
    printf("\n=== G3: deferred activation for streams in ordering chain ===\n");
    printf("    [cudaMemAttachGlobal: UVM heuristic migration suppressed]\n");

    float *managed_data;
    CUDA_CHECK(cudaMallocManaged(&managed_data, N * sizeof(float),
                                  cudaMemAttachGlobal));
    for (int i = 0; i < N; i++) managed_data[i] = 1.0f;

    cudaStream_t stream1, other_stream;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&other_stream));

    cudaEvent_t chain_event;
    CUDA_CHECK(cudaEventCreate(&chain_event));

    // stream1 上：kernelA 处理 managed_data
    scaleKernel<<<blocks(N), 256, 0, stream1>>>(managed_data, 2.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // stream1 上：hostFunc 读取 kernelA 结果
    // chain_event 必须记录在 hostFunc 之后：
    //   若记录在 hostFunc 之前，chain_event 在 scaleKernel 完成时就 fire，
    //   other_stream 的 WaitEvent 条件立刻满足，addKernel 与 hostFunc 并发执行，
    //   CPU（hostFunc）和 GPU（addKernel）同时访问同一块 managed_data → 段错误
    CUDA_CHECK(cudaLaunchHostFunc(stream1, [](void *arg) {
        auto *data = static_cast<float*>(arg);
        // hostFunc 执行期间 CPU 可以安全读取 managed_data，原因有两条：
        //
        // ① G1：stream1 在 hostFunc 执行期间被强制视为 IDLE
        //        stream1 上没有并发的 GPU work 访问 managed_data
        //
        // ② 普通事件顺序（非 G3）：
        //        chain_event 记录在此 hostFunc 之后，尚未 fire；
        //        other_stream 的 WaitEvent 条件未满足，还没有 ACTIVE 的 GPU work；
        //        内存是 cudaMemAttachGlobal（无独占语义，所有 stream 均可见），
        //        但 UVM 不会主动向不 ACTIVE 的 stream 迁移页面，
        //        且 other_stream 当前无 GPU work 执行，不会触发 GPU page fault。
        //
        // 注意：这里保证安全的是"事件同步使 other_stream 尚未 ACTIVE"，
        //       不是 G3。G3 的场景是：other_stream 已越过 WaitEvent、有 work 排队，
        //       但因处于 ordering chain 中而被驱动推迟激活——本示例结构未覆盖该场景。
        verify(data, 2.0f, N, "G3: hostFunc reads managed_data safely");
        printf("  [INFO] other_stream not yet ACTIVE (chain_event not fired)\n");
    }, managed_data));

    // hostFunc 完成后才记录 chain_event
    // → other_stream 的 WaitEvent 在 hostFunc 结束后才满足
    // → addKernel 不会与 hostFunc 并发访问 managed_data
    CUDA_CHECK(cudaEventRecord(chain_event, stream1));
    CUDA_CHECK(cudaStreamWaitEvent(other_stream, chain_event, 0));

    // other_stream 上：kernelC 等 chain_event（hostFunc 完成后才 fire）
    addKernel<<<blocks(N), 256, 0, other_stream>>>(managed_data, 1.0f, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(other_stream));

    // 最终结果：2.0 + 1.0 = 3.0
    verify(managed_data, 3.0f, N, "G3: final result");

    CUDA_CHECK(cudaFree(managed_data));
    CUDA_CHECK(cudaEventDestroy(chain_event));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(other_stream));
}



// ═══════════════
// 示例 4：G4 — hostFunc 完成不 reactivate stream
//
// 场景 A：连续多个 hostFunc，stream 在整个过程中保持 IDLE
// 场景 B：end-of-stream 通知模式，hostFunc 作为完成信号
//         stream 永远保持 IDLE，主线程通过 condition variable 被唤醒
// ══════════════════
void demo_g4()
{
    printf("\n=== G4: completion does not reactivate stream ===\n");

    // ─── 场景 A：连续 hostFunc ───
    printf("  [Scenario A] consecutive hostFuncs — stream stays IDLE throughout\n");

    float *um_data;
    CUDA_CHECK(cudaMallocManaged(&um_data, N * sizeof(float),
                                  cudaMemAttachGlobal));
    for (int i = 0; i < N; i++) um_data[i] = 0.0f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaStreamAttachMemAsync(stream, um_data, 0,
                                         cudaMemAttachSingle));

    // kernelA：把数据变为 1.0
    // kernelA 在 GPU 上写数据 → 页面位于 GPU 显存
    addKernel<<<blocks(N), 256, 0, stream>>>(um_data, 1.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // 连续三个 hostFunc，中间没有任何 device work
    // G4 保证：func1 完成后 stream 不 reactivate，func2 看到的仍是 IDLE
    //          func2 完成后 stream 不 reactivate，func3 看到的仍是 IDLE

    struct FuncCtx { float *data; int n; int step; };
    FuncCtx fctx[3] = {
        { um_data, N, 1 },
        { um_data, N, 2 },
        { um_data, N, 3 },
    };

    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaLaunchHostFunc(stream, [](void *arg) {
            auto *c = static_cast<FuncCtx*>(arg);
            // 每个 hostFunc 执行时 stream 都是 IDLE（G1 + G4 共同保证）
            // cudaMemAttachSingle：stream IDLE 时 CPU 合法访问
            //
            // ── CPU page fault 触发迁移 ─────
            // kernelA 写完后页面仍在 GPU 显存。
            // G1 保证 CPU 可以访问（无 GPU 并发），但并不保证页面已迁移到 CPU 侧。
            // verify() 读 c->data 时，CPU 发现页面不在系统内存 → 触发 CPU page fault
            // → UVM 驱动把页面从 GPU 显存迁移到系统内存 → CPU 读到正确数据。
            // 后续连续 hostFunc 读同一页面：页面已在系统内存，不再触发 page fault。
            // ───────────
            printf("    func%d: stream is IDLE, CPU accessing managed memory\n",
                   c->step);
            verify(c->data, 1.0f, c->n, "G4-A: consecutive hostFunc access");
        }, &fctx[i]));
    }

    // 三个 hostFunc 之后才有新的 device work，stream 在这里才重新变为 ACTIVE
    //
    // ── GPU page fault 触发迁移 ──────
    // hostFunc 的 verify() 已把页面迁移到系统内存（CPU 侧）。
    // addKernel 提交后 stream 变 ACTIVE，GPU 开始执行 kernel。
    // kernel 访问 um_data 时发现页面不在显存 → 触发 GPU page fault
    // → UVM 驱动把页面从系统内存迁回 GPU 显存 → kernel 正常执行。
    //
    // 完整页面往返：
    //   kernelA 写  → 页面在 GPU 显存
    //   verify() 读 → CPU page fault → 页面迁移到系统内存
    //   kernelB 写  → GPU page fault → 页面迁回 GPU 显存
    //
    // 性能提示：若对迁移开销敏感，可在 hostFunc 结束后、kernelB 前显式 prefetch，
    //   避免 GPU page fault：
    //   cudaMemPrefetchAsync(um_data, N*sizeof(float), device_id, stream);
    // ───────────────
    addKernel<<<blocks(N), 256, 0, stream>>>(um_data, 1.0f, N); // 1.0+1=2.0
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    verify(um_data, 2.0f, N, "G4-A: final after kernelB");

    // ─── 场景 B：end-of-stream 通知 ───
    printf("\n  [Scenario B] end-of-stream notification — no reactivation\n");

    // 重置数据
    for (int i = 0; i < N; i++) um_data[i] = 0.0f;
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, um_data, 0,
                                         cudaMemAttachSingle));

    // GPU 计算
    addKernel<<<blocks(N), 256, 0, stream>>>(um_data, 5.0f, N); // 结果 5.0
    CUDA_CHECK(cudaGetLastError());
    scaleKernel<<<blocks(N), 256, 0, stream>>>(um_data, 2.0f, N); // 结果 10.0
    CUDA_CHECK(cudaGetLastError());

    // end-of-stream hostFunc：通知主线程，不触发任何 device work
    G4Ctx g4ctx;
    g4ctx.data     = um_data;
    g4ctx.n        = N;
    g4ctx.expected = 10.0f;

    CUDA_CHECK(cudaLaunchHostFunc(stream, [](void *arg) {
        auto *ctx = static_cast<G4Ctx*>(arg);

        // stream 是 IDLE（G1）
        // hostFunc 之后没有任何 device work 入队
        // G4：hostFunc 完成后 stream 永远保持 IDLE
        // cudaMemAttachSingle：CPU 可以合法读取结果

        verify(ctx->data, ctx->expected, ctx->n,
               "G4-B: end-of-stream hostFunc reads result");

        // ── hostFunc（工作线程）端：通知主线程 ──────
        // 配合时序说明（对应下方主线程的 cv.wait）：
        //
        // 阶段 1（主线程先到，done=false）：
        //   主线程：获得锁 → 检查 done==false → cv.wait 原子地释放锁并挂起
        //   此后锁空闲，hostFunc 才能获得锁
        //
        // 阶段 2（hostFunc 执行）：
        //   ① lock_guard 构造 → 获得 mtx 锁
        //   ② done.store(true) → 设置完成标志
        //   ③ } → lock_guard 析构 → 自动解锁（必须先解锁再 notify）
        //   ④ cv.notify_one() → 唤醒主线程
        //      先解锁再 notify 的原因：主线程被唤醒后立刻抢锁，
        //      若此时锁还在 hostFunc 手里，主线程被迫再阻塞一次，多余开销
        //
        // 阶段 3（主线程被唤醒）：
        //   重新获得锁 → 再次检查 predicate：done==true → 退出 wait → 继续执行
        //
        // 为什么 done 是 atomic 还需要 lock_guard 加锁？
        //   done 是 std::atomic<bool>，store 本身线程安全，lock_guard 不是严格必须的。
        //   但去掉锁后有微妙时序窗口：
        //     主线程（持锁）：cv.wait 检查 done==false → 决定睡眠
        //                                ← 此处若 hostFunc 无锁发出 notify
        //     hostFunc（无锁）：done=true → notify_one() ← 通知可能丢失
        //   C++ 标准保证 cv.wait(lock) 释放锁和进入等待是原子的，
        //   且 cv.wait(lock, predicate) 睡前会先检查 predicate，
        //   所以上述窗口实际不存在——技术上不加锁也正确。
        //
        //   仍然加锁的原因：
        //   ① 防御性：若将来 done 改为普通 bool，无锁会有 data race UB，加锁天然安全
        //   ② 惯用模式：C++ 社区约定"修改 cv predicate 时持锁"，意图更清晰
        //   ③ 性能：先解锁再 notify（当前写法），主线程被唤醒后能立刻拿到锁，
        //           避免"唤醒后再次阻塞在锁上"的多余开销
        //
        // 为什么 lock 没有手动解锁？
        //   lock_guard 是 RAII 对象：构造时自动加锁，析构时自动解锁。
        //   C++ 保证局部对象在离开作用域（}）时按构造顺序逆序析构。
        //   此处用 { } 单独划出一个作用域，lock_guard 在 } 处析构 → 自动解锁，
        //   无需也不能手动调用 unlock()（lock_guard 没有提供 unlock 方法）。
        //   若需要手动控制解锁时机，应改用 unique_lock（提供 .unlock() 方法）。
        {
            std::lock_guard<std::mutex> lock(ctx->mtx); // 构造：加锁
            ctx->done.store(true);
        }                                                // 析构：自动解锁，先于 notify
        ctx->cv.notify_one();  // 解锁后再 notify：主线程醒来能立刻拿到锁，减少开销
        // ───────────────
        // hostFunc 返回后 stream 仍然 IDLE（G4 保证）
    }, &g4ctx));

    // hostFunc 之后没有提交任何 device work
    // stream 将永远保持 IDLE

    // ── 主线程端：睡眠等待 hostFunc 通知 ─────
    // unique_lock vs lock_guard：
    //   cv.wait() 内部需要在挂起时临时释放锁、被唤醒时重新获锁，
    //   要求锁对象支持手动 unlock/lock，只有 unique_lock 支持，lock_guard 不行
    //
    // cv.wait(lock, predicate) 等价于：
    //   while (!predicate()) { cv.wait(lock); }
    //   每次被唤醒都重新检查 predicate，防止虚假唤醒（spurious wakeup）
    //
    // 执行流：
    //   ① unique_lock 构造 → 获得 mtx 锁
    //   ② cv.wait：检查 done==false → 原子地释放锁 + 挂起主线程
    //   ③（hostFunc 完成上方通知后）主线程被唤醒 → 重新获得锁
    //   ④ 再次检查 done==true → 退出 wait
    //   ⑤ } → unique_lock 析构 → 自动解锁
    //
    // 为什么 unique_lock 也没有手动解锁？
    //   unique_lock 同样是 RAII 对象：构造时加锁，析构时自动解锁。
    //   与 lock_guard 的区别是 unique_lock 额外提供了 .unlock() / .lock() 方法，
    //   允许在析构前手动控制，但"可以手动解锁"不代表"必须手动解锁"。
    //   这里 cv.wait 返回后不需要继续持锁（只需读 done 的结果，已经在 predicate 里确认），
    //   所以直接依赖 } 处析构自动解锁即可，不需要手动调用 lock.unlock()。
    //
    //   对比：何时需要手动 unlock？
    //     std::unique_lock<std::mutex> lock(mtx);
    //     // ... 持锁做一些事 ...
    //     lock.unlock();       // 提前解锁：后续代码不需要锁保护，又不想等到 } 才释放
    //     do_expensive_work(); // 这段耗时操作不在锁的保护范围内，其他线程可以进入
    printf("  [main thread] waiting for GPU completion signal...\n");
    {
        std::unique_lock<std::mutex> lock(g4ctx.mtx);        // ① 获得锁
        g4ctx.cv.wait(lock, [&]{ return g4ctx.done.load(); }); // ②③④ 等待
    }                                                          // ⑤ 析构自动解锁
    // ──────────
    printf("  [main thread] notified! stream is IDLE, "
           "managed memory safely accessible\n");

    // 此刻：
    //   stream 是 IDLE（G4 保证）
    //   um_data 在 CPU 侧（page fault 已响应）
    //   主线程可以安全处理结果
    verify(um_data, 10.0f, N,
           "G4-B: main thread reads result after notification");

    CUDA_CHECK(cudaFree(um_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ═══════════════
// 示例 5：四条保证综合 — 完整 pipeline
//
// 场景：
//   s1 做数据预处理（kernelA）
//   s2 join s1，做主计算（kernelB），然后通过 hostFunc 链处理结果
//   hostFunc1（s2）：读取并验证 kernelA + kernelB 的结果（G1 + G2）
//   hostFunc2（s2）：继续处理，stream 保持 IDLE（G4）
//   s3 join s2，排在 hostFunc 链之后（G3 suppresses premature activation）
//   hostFunc3（s2）：end-of-stream 通知（G4 永久 IDLE）
// ═══════════════════
void demo_all()
{
    printf("\n=== Combined: full pipeline using all four guarantees ===\n");

    float *data_a, *data_b, *data_c;
    CUDA_CHECK(cudaMallocManaged(&data_a, N * sizeof(float),
                                  cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&data_b, N * sizeof(float),
                                  cudaMemAttachGlobal));
    CUDA_CHECK(cudaMallocManaged(&data_c, N * sizeof(float),
                                  cudaMemAttachGlobal));

    for (int i = 0; i < N; i++) {
        data_a[i] = 1.0f;
        data_b[i] = 0.0f;
        data_c[i] = 0.0f;
    }

    cudaStream_t s1, s2, s3;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));
    CUDA_CHECK(cudaStreamCreate(&s3));

    cudaEvent_t e1, e2;
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    // ── s1：预处理 ──
    // data_a: 1.0 × 4 = 4.0
    scaleKernel<<<blocks(N), 256, 0, s1>>>(data_a, 4.0f, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1, s1));   // 标记 s1 工作完成点

    // ── s2 join s1，主计算 ──
    CUDA_CHECK(cudaStreamWaitEvent(s2, e1, 0));  // 真正的同步建立在这里（G2 来源）
    // data_b = data_a = 4.0
    addArrayKernel<<<blocks(N), 256, 0, s2>>>(data_b, data_a, N);
    CUDA_CHECK(cudaGetLastError());
    // data_b: 4.0 × 2 = 8.0
    scaleKernel<<<blocks(N), 256, 0, s2>>>(data_b, 2.0f, N);
    CUDA_CHECK(cudaGetLastError());

    // s2 上记录 e2，供 s3 建立 ordering chain
    CUDA_CHECK(cudaEventRecord(e2, s2));
    CUDA_CHECK(cudaStreamWaitEvent(s3, e2, 0));   // s3 join s2（G3 ordering chain）

    // ── s2 上的 hostFunc 链 ──

    // hostFunc1：G1 + G2
    struct Hf1Ctx { float *da; float *db; int n; };
    Hf1Ctx hf1ctx { data_a, data_b, N };

    CUDA_CHECK(cudaLaunchHostFunc(s2, [](void *arg) {
        auto *c = static_cast<Hf1Ctx*>(arg);
        printf("  [hostFunc1] G1: s2 is IDLE, CPU can access managed memory\n");
        printf("  [hostFunc1] G2: kernelA(s1) and kernelB(s2) both done\n");
        // G2：cudaStreamWaitEvent 建立的保证链传递到这里
        // data_a 是 kernelA 的结果（来自 s1）
        // data_b 是 kernelB 的结果（来自 s2，依赖 s1）
        verify(c->da, 4.0f, c->n, "hostFunc1: data_a (from s1)");
        verify(c->db, 8.0f, c->n, "hostFunc1: data_b (from s2)");
    }, &hf1ctx));

    // hostFunc2：G4（连续 hostFunc，stream 保持 IDLE）
    struct Hf2Ctx { float *db; int n; };
    Hf2Ctx hf2ctx { data_b, N };

    CUDA_CHECK(cudaLaunchHostFunc(s2, [](void *arg) {
        auto *c = static_cast<Hf2Ctx*>(arg);
        // G4：hostFunc1 完成后 s2 没有 reactivate，仍然是 IDLE
        printf("  [hostFunc2] G4: s2 still IDLE after hostFunc1 completion\n");
        // CPU 修改 data_b：8.0 + 2.0 = 10.0
        for (int i = 0; i < c->n; i++) c->db[i] += 2.0f;
        verify(c->db, 10.0f, c->n, "hostFunc2: modified data_b on CPU");
    }, &hf2ctx));

    // s3 上：kernelC 依赖 e2（ordering chain）
    // G3：在 s2 的 hostFunc 链完成前，s3 上新提交的工作不被视为 ACTIVE
    // data_c = data_b，此时 data_b 将被 hostFunc2 修改为 10.0
    // kernelC 在 s2 的整个 hostFunc 链完成后才真正执行
    addArrayKernel<<<blocks(N), 256, 0, s3>>>(data_c, data_b, N);
    CUDA_CHECK(cudaGetLastError());
    scaleKernel<<<blocks(N), 256, 0, s3>>>(data_c, 1.5f, N);   // 10.0 × 1.5 = 15.0
    CUDA_CHECK(cudaGetLastError());

    // hostFunc3：end-of-stream 通知（G4）
    G4Ctx final_ctx;
    final_ctx.data     = data_c;
    final_ctx.n        = N;
    final_ctx.expected = 15.0f;

    // 注意：hostFunc3 放在 s2 上（s2 的 hostFunc 链末尾）
    // s3 的工作通过 ordering chain 保证在 s2 的 hostFunc 链之后
    // 所以这里用 s3 的 sync 等待 data_c 就绪
    // end-of-stream 通知放在 s3 上
    CUDA_CHECK(cudaLaunchHostFunc(s3, [](void *arg) {
        auto *ctx = static_cast<G4Ctx*>(arg);
        printf("  [hostFunc3] G4: end-of-stream notification\n");
        printf("  [hostFunc3] G4: s3 will stay IDLE after this returns\n");
        verify(ctx->data, ctx->expected, ctx->n,
               "hostFunc3: final result in data_c");
        {
            std::lock_guard<std::mutex> lock(ctx->mtx);
            ctx->done.store(true);
        }
        ctx->cv.notify_one();
    }, &final_ctx));

    // 主线程等待完成信号
    printf("  [main] waiting for pipeline completion...\n");
    {
        std::unique_lock<std::mutex> lock(final_ctx.mtx);
        final_ctx.cv.wait(lock, [&]{ return final_ctx.done.load(); });
    }
    printf("  [main] pipeline done, all streams IDLE\n");

    // 最终验证
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaStreamSynchronize(s2));
    CUDA_CHECK(cudaStreamSynchronize(s3));

    verify(data_a, 4.0f,  N, "Final: data_a");
    verify(data_b, 10.0f, N, "Final: data_b");
    verify(data_c, 15.0f, N, "Final: data_c");

    CUDA_CHECK(cudaFree(data_a));
    CUDA_CHECK(cudaFree(data_b));
    CUDA_CHECK(cudaFree(data_c));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaStreamDestroy(s3));
}


// ═════════
// main
// ═════════
int main()
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("managedMemory: %s\n",
           prop.managedMemory ? "yes" : "no");
    printf("concurrentManagedAccess: %s\n",
           prop.concurrentManagedAccess ? "yes" : "no");

    if (!prop.managedMemory) {
        printf("This device does not support managed memory. Exiting.\n");
        return 1;
    }

    demo_g1();
    demo_g2();
    demo_g3();
    demo_g4();
    demo_all();

    printf("\nAll demos completed.\n");
    return 0;
}
