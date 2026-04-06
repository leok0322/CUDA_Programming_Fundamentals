/**
 * cuda_pipeline_hostfunc.cu
 *
 * 模式：流水线阶段通知
 *   GPU 完成一个 batch 后，通过 cudaLaunchHostFunc 通知 CPU，
 *   CPU 准备下一个 batch 的数据，GPU 和 CPU 重叠执行。
 *
 * 流水线设计：
 *   双缓冲（ping-pong）：两块 device buffer 交替使用，
 *   GPU 处理 buf[cur] 时，CPU 同时向 buf[nxt] 传输下一批数据。
 *
 * 时间轴（理想情况）：
 *
 *   batch:      0          1          2          3
 *   GPU:    [kernel 0] [kernel 1] [kernel 2] [kernel 3]
 *               ↓notify    ↓notify    ↓notify
 *   CPU:        [H2D 1]    [H2D 2]    [H2D 3]
 *
 *   GPU 和 CPU 重叠，吞吐量接近两者中较快的一方。
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 -std=c++20 cuda_pipeline_hostfunc.cu -o cuda_pipeline_hostfunc
 *
 * 运行：
 *   ./cuda_pipeline_hostfunc
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <semaphore>    // C++20：std::counting_semaphore
#include <thread>
#include <atomic>
#include <vector>

// ────────────────
// 错误检查宏 CUDA_CHECK
//
// 用法：CUDA_CHECK(cudaXxx(...));
//   将任意返回 cudaError_t 的 CUDA API 调用包裹起来，出错时打印详情并退出。
//
// #define 是预处理器宏，不是 C++ 标识符：
//   · 没有类型、没有链接性（linkage），不受 ODR 约束
//   · 多个 .cpp 文件 include 同一头文件中的宏，各自独立展开，不会冲突
//   · 编译阶段开始前已被替换为文本，链接器完全看不到它
//
// do { ... } while (0) 惯用法：
//   · 让宏展开后是一个完整语句，可安全用于 if/else 分支
//   · 若直接写 { ... }，在 if (cond) CUDA_CHECK(x); else ... 中会语法错误
//     例：if (cond) { ... }  ← 多出一个分号导致 else 孤立
//         if (cond) do{...}while(0);  ← 正确，整体是一条语句
//
// 宏参数 (call) 加括号：
//   · 防止调用表达式含逗号或运算符时被宏参数解析错误
//
// EXIT_FAILURE：
//   · 定义在 <stdlib.h>：#define EXIT_FAILURE 1
//   · 同样是预处理宏，不是变量，无链接性
//   · 表示"程序异常退出"的标准状态码，传给 exit() 通知操作系统
// ────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);          /* 执行 CUDA API，捕获返回值 */ \
        if (err != cudaSuccess) {          /* cudaSuccess == 0，非零即错 */ \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",               \
                    __FILE__,              /* 预定义宏：当前文件名        */ \
                    __LINE__,              /* 预定义宏：当前行号          */ \
                    cudaGetErrorName(err), /* 错误枚举名，如 cudaErrorXxx */ \
                    cudaGetErrorString(err)); /* 人类可读的错误描述       */ \
            exit(EXIT_FAILURE);            /* 立即终止进程，不做清理      */ \
        }                                                                   \
    } while (0)

// ─────────────────
// kernel：模拟 batch 计算
// ─────────────────
__global__ void processKernel(float *out, const float *in, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;
}

static const int    BATCH_SIZE  = 1 << 18;          // 每批元素数
static const size_t BATCH_BYTES = BATCH_SIZE * sizeof(float);
static const int    NUM_BATCHES = 6;                 // 总批次数
static const int    NUM_BUFS   = 2;                  // 双缓冲（ping-pong）


// ════════════════════
// PipelineCtx：hostFunc 的回调数据
// ═════════════════════
//
//   每次调用 cudaLaunchHostFunc 时，传入一个 PipelineCtx 实例。
//   hostFunc 执行时通过 sem->release() 通知 CPU 预处理线程。
//
struct PipelineCtx {
    std::counting_semaphore<1> *sem;      // 通知信号：GPU batch 已完成
    int                         batch_id; // 刚完成的 batch 编号（用于调试/日志）
};

// ───────────────
// hostFunc：GPU batch 完成后的回调
//
//   执行时机：stream 中此命令之前的所有操作（kernel、memcpy）已完成。
//   约束：不能调用任何 CUDA API，必须尽快返回。
//   只做一件事：release semaphore，通知 CPU 预处理线程。
// ─────────────────
void notify_batch_done(void *data)
{
    auto *ctx = static_cast<PipelineCtx *>(data);

    // 只通知，立刻返回
    // release：计数 0 → 1，唤醒正在 sem->acquire() 等待的 CPU 线程
    ctx->sem->release();

    // 不调用 cudaMemcpy、cudaMalloc 等任何 CUDA API（否则死锁）
}


// ═══════════════════
// 示例 1：简单流水线
//
//   GPU 处理 batch N 时，CPU 同时准备 batch N+1 的数据（H2D）。
//   通过 hostFunc + semaphore 协调两者节奏。
// ═══════════════════
void demo_simple_pipeline(void)
{
    printf("\n─── 示例 1：简单流水线（GPU 处理 + CPU 准备重叠）───\n");

    // ── 分配双缓冲 device 内存 ───────────────────────────────────────
    //
    //   buf[0]、buf[1] 交替使用（ping-pong）：
    //     偶数 batch → buf[0]
    //     奇数 batch → buf[1]
    //   GPU 处理 buf[cur] 时，CPU 向 buf[nxt] 传输下一批数据。
    float *d_in[NUM_BUFS], *d_out[NUM_BUFS];
    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaMalloc((void **)&d_in[i],  BATCH_BYTES));
        CUDA_CHECK(cudaMalloc((void **)&d_out[i], BATCH_BYTES));
    }

    // ── 分配 pinned host 内存（H2D 必须用 pinned 才能异步传输）────────
    //
    //   h_in[i]：CPU 预处理线程把数据写到这里，再 H2D 到 d_in[i]
    //   h_out：  D2H 结果缓冲（所有 batch 共用，演示用）
    float *h_in[NUM_BUFS], *h_out;
    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaMallocHost((void **)&h_in[i], BATCH_BYTES));
    }
    CUDA_CHECK(cudaMallocHost((void **)&h_out, BATCH_BYTES));

    // ── 创建 stream ───────
    //
    //   用一个 non-blocking stream 串行化所有操作（H2D、kernel、hostFunc、D2H）。
    //   stream 保证命令按提交顺序执行，无需额外同步。
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // ── 两个 semaphore，构成双向握手 ──────────────────────────────────
    //
    //   gpuDone  （初始 1）：GPU → prepThread
    //     hostFunc 执行后 release，prepThread acquire 后才开始准备下一批。
    //     初始值 1：允许 prepThread 立刻开始准备 batch 1（不等 GPU）。
    //
    //   h2dReady （初始 0）：prepThread → 主线程
    //     prepThread 完成 H2D 后 release，主线程 acquire 后才提交对应 kernel。
    //     初始值 0：主线程提交 kernel 1 前必须等 prepThread 完成 H2D(batch 1)。
    //
    //   握手流程（解决竞态）：
    //     prepThread:  gpuDone.acquire() → 准备数据 → H2D → h2dReady.release()
    //     主线程:      h2dReady.acquire() → 提交 kernel r+1 → stream
    //
    //   时序：
    //     GPU:        [kernel 0]──hostFunc──release gpuDone
    //                                  [kernel 1]──hostFunc──release gpuDone
    //     prepThread:          acquire──[H2D 1]──release h2dReady
    //                                           acquire──[H2D 2]──release h2dReady
    //     主线程:     submit 0  acquire h2dReady──submit 1  acquire h2dReady──submit 2
    std::counting_semaphore<1> gpuDone(1);   // GPU 完成信号（初始 1）
    std::counting_semaphore<1> h2dReady(0);  // H2D 就绪信号（初始 0）

    // ── PipelineCtx 数组：每个 batch 一个，生命周期覆盖整个流水线 ────
    //
    //   核心要求：ctx 必须在 hostFunc 执行时依然有效（不能是已销毁的临时变量）。
    //   这里用 std::vector，等价于固定大小数组 PipelineCtx ctxs[NUM_BATCHES]。
    //
    //   vector vs 数组的选择：
    //
    //     NUM_BATCHES 是编译期常量（static const int），两种写法都可以：
    //
    //       PipelineCtx ctxs[NUM_BATCHES];          // ✅ 编译期大小，栈上数组
    //       std::vector<PipelineCtx> ctxs(NUM_BATCHES); // ✅ 等价，堆上分配
    //
    //     必须用 vector 的场景：批次数在运行时才确定
    //       int num = get_batch_count();             // 运行时值
    //       PipelineCtx ctxs[num];                  // ❌ VLA，C++ 标准不支持
    //       std::vector<PipelineCtx> ctxs(num);     // ✅ 运行时大小，标准 C++
    //
    //   两者生命周期都是函数作用域（函数返回前不销毁），满足 hostFunc 的要求。
    std::vector<PipelineCtx> ctxs(NUM_BATCHES);
    for (int i = 0; i < NUM_BATCHES; i++) {
        ctxs[i] = {&gpuDone, i};  // hostFunc 通知 gpuDone，不再是 sem
        // {&sem, i}：初始化 PipelineCtx{sem*, batch_id}
        //   &sem：所有元素的 sem 指针都指向同一个 sem 对象（不是副本）
        //
        //   内存示意：
        //     ctxs[0].sem ──┐
        //     ctxs[1].sem ──┤
        //     ctxs[2].sem ──┼──→ sem（唯一的 counting_semaphore<1> 对象）
        //     ctxs[3].sem ──┤
        //     ctxs[4].sem ──┘
        //
        //   为什么共用一个 semaphore 就够？
        //     流水线是串行节奏：stream 保证 hostFunc 按顺序执行，
        //     同一时刻只有一个 hostFunc 在 release，prepThread 也只有一个在 acquire，
        //     release 和 acquire 严格一一对应，不会信号混淆。
        //
        //   什么时候需要多个 semaphore？
        //     多个 stream 并发触发各自的 hostFunc 时，
        //     每个 stream 应有独立的 semaphore，避免不同 stream 的信号互相干扰。
    }

    int threads = 256, blocks = (BATCH_SIZE + threads - 1) / threads;

    // ── 预填充第 0 个 batch 的数据 ───────
    //
    //   流水线启动前，必须先把第 0 批数据准备好，
    //   否则 GPU 第一轮 kernel 没有输入数据。
    for (int i = 0; i < BATCH_SIZE; i++) h_in[0][i] = 1.0f;
    CUDA_CHECK(cudaMemcpyAsync(d_in[0], h_in[0], BATCH_BYTES,
                               cudaMemcpyHostToDevice, stream));

    // ── CPU 预处理线程 ────────
    //
    //   职责：等 GPU 完成 batch r，准备 batch r+1 的数据并做 H2D。
    //   与 GPU 并发执行，隐藏数据传输延迟。
    std::thread prepThread([&]() {
        for (int r = 0; r < NUM_BATCHES - 1; r++) {
            // 等 GPU 完成 batch r（hostFunc release gpuDone）
            // 初始 gpuDone=1，第一次 acquire 立刻通过，无需等 GPU
            gpuDone.acquire();

            int nxt = (r + 1) % NUM_BUFS;

            // 模拟 CPU 端数据预处理
            float val = (float)(r + 1) * 0.5f;
            for (int i = 0; i < BATCH_SIZE; i++) h_in[nxt][i] = val;

            // H2D：用同步 cudaMemcpy，不用 cudaMemcpyAsync
            //
            //   cudaMemcpy（同步）：
            //     阻塞 prepThread，直到数据真正写入 GPU 显存后才返回。
            //     返回后立刻 h2dReady.release()，主线程提交的 kernel 读到的
            //     d_in[nxt] 是完整的数据，安全。
            //
            //   为何不用 cudaMemcpyAsync 提交到 computeStream：
            //     stream 有内部锁，多线程提交不会崩溃，但入队顺序不确定：
            //
            //       主线程:    processKernel → computeStream ──┐ 竞争同一把锁
            //       prepThread: cudaMemcpyAsync → computeStream─┘ 谁先拿到谁先入队
            //
            //     OS 调度决定哪个线程先拿到锁，可能出现：
            //       情况 A（正确）：[H2D][kernel]  ← prepThread 先入队
            //       情况 B（错误）：[kernel][H2D]  ← 主线程先入队，kernel 读脏数据
            //
            //     锁只保证入队操作本身不冲突，不保证入队顺序符合预期。
            //
            //   为何不用 cudaMemcpyAsync（默认 stream）：
            //     cudaMemcpyAsync 入队后立刻返回，拷贝尚未完成，
            //     若此时 release h2dReady，主线程提交 kernel 会读到未写完的数据。
            //     必须在 release 前加 cudaStreamSynchronize(0) 等拷贝真正完成，
            //     但这与直接用 cudaMemcpy 完全等价（cudaMemcpy 内部就是
            //     cudaMemcpyAsync + cudaStreamSynchronize 的封装），
            //     所以这里直接用 cudaMemcpy 更简洁。
            CUDA_CHECK(cudaMemcpy(d_in[nxt], h_in[nxt], BATCH_BYTES,
                                  cudaMemcpyHostToDevice));

            printf("  prepThread：batch %d H2D 完成（val=%.1f）\n", r + 1, val);

            // d_in[nxt] 已完整写入 GPU 显存，通知主线程可以安全提交 kernel r+1
            h2dReady.release();
        }
    });

    // ── 主线程：逐批提交 kernel，提交前等 h2dReady ───────────────────
    //
    //   修复竞态：主线程不再一次性提交所有 kernel。
    //   对 batch r > 0，先等 prepThread 完成 H2D（h2dReady.acquire()），
    //   确认 d_in[r] 数据就绪后再提交 kernel r。
    //
    //   流水线仍然有效：
    //     GPU 执行 kernel r 时，prepThread 同时在做 H2D(r+1)，
    //     主线程等待 h2dReady 的时间很短（H2D 比 kernel 快），
    //     GPU 几乎不会空等。
    for (int r = 0; r < NUM_BATCHES; r++) {
        int cur = r % NUM_BUFS;

        if (r > 0) {
            // r > 0 才等 h2dReady，原因：
            //
            //   r == 0：d_in[0] 由主线程在循环前已预填充 + H2D，
            //           不需要等 prepThread，直接提交 kernel 0。
            //
            //   r > 0 ：d_in[cur] 由 prepThread 负责准备，
            //           必须等 prepThread 的 h2dReady.release() 后才能提交。
            //
            //   如果去掉 if (r > 0)，r == 0 时主线程在 h2dReady.acquire() 阻塞，
            //   而 prepThread 的第一轮（准备 batch 1）会 release h2dReady，
            //   导致主线程等到 batch 1 的 H2D 完成才提交 kernel 0，
            //   batch 0 的数据早就就绪了却白白多等一轮——流水线启动被不必要地延迟。
            h2dReady.acquire();
        }

        // d_in[cur] 已就绪，安全提交 kernel
        processKernel<<<blocks, threads, 0, stream>>>(
            d_out[cur], d_in[cur], 2.0f, BATCH_SIZE);

        // cudaGetLastError()：检查 kernel 启动时的错误，不是同步点
        //
        //   kernel<<<...>>>() 本身是异步的：命令入队后 CPU 立刻返回，
        //   GPU 不一定已开始执行。启动过程中若参数非法（grid/block 超限等），
        //   驱动会在入队时将错误写入线程本地的错误标志。
        //   cudaGetLastError() 读取并清除该标志，纯 CPU 操作，立刻返回，
        //   不等 GPU 执行，不是同步点，不阻塞主线程。
        //
        //   能捕获：启动参数非法、设备资源不足等入队阶段的错误。
        //   不能捕获：kernel 执行时的越界访问、非法指令等运行时错误，
        //             这类错误要在 cudaStreamSynchronize 之后再调用
        //             cudaGetLastError 才能看到。
        CUDA_CHECK(cudaGetLastError());

        // kernel 完成后通知 prepThread 开始准备下一批
        CUDA_CHECK(cudaLaunchHostFunc(stream, notify_batch_done, &ctxs[r]));

        // D2H：把结果拷贝回 host
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out[cur], BATCH_BYTES,
                                   cudaMemcpyDeviceToHost, stream));

        printf("  主线程：kernel %d 已提交\n", r);
    }

    printf("  主线程：所有 %d 个 batch 已提交，等待完成...\n", NUM_BATCHES);

    // ── 同步点 1：CPU 线程同步 ──────
    //   prepThread.join()：
    //     纯 CPU 同步点，主线程阻塞直到 prepThread 函数体执行完毕。
    //     与 GPU 无关，不检查任何 CUDA 错误。
    //     确保 prepThread 的所有 H2D 操作已完成后，主线程才继续。
    prepThread.join();

    // ── 同步点 2：GPU 同步 ────────
    //   cudaStreamSynchronize：
    //     CPU 阻塞直到 stream 中所有操作全部完成
    //     （kernel、cudaLaunchHostFunc、cudaMemcpyAsync 均已执行完毕）。
    //     返回值包含 stream 中所有操作的执行错误（含 kernel 运行时错误）：
    //       cudaSuccess            → 全部成功
    //       其他错误码             → CUDA_CHECK 调用 exit()，程序终止
    //     因此 CUDA_CHECK(cudaStreamSynchronize) 已足够捕获所有错误，
    //     其后无需再调用 cudaGetLastError()——若 Synchronize 返回错误，
    //     CUDA_CHECK 已 exit()，后续代码根本不会执行。
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("  流水线完成，h_out[0] = %.2f\n", h_out[0]);

    // ── 清理 ───────────────────
    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaFreeHost(h_in[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════════════════
// 示例 2：多 stream 流水线（H2D stream 与 compute stream 分离）
//
//   用两个 stream 分别负责数据传输和计算，真正实现重叠：
//     transferStream：H2D 传输（由 main 线程预填充 batch0，之后由 prepThread 提交）
//     computeStream ：kernel 计算（由 main 线程提交）
//
//   同步机制：
//     GPU 侧：cudaStreamWaitEvent（computeStream 等 transferStream 的 H2D event）
//     CPU 侧：sem（hostFunc 通知 prepThread，GPU kernel 已完成）
//             wevInserted（prepThread 通知 main，WaitEvent 已插入 computeStream）
//
//   ── 为什么需要 wevInserted ──────────────────────────────────────────────
//
//   竞态根源：main 线程提交 kernel 的速度（纯 CPU，微秒级）远快于
//             prepThread 插入 WaitEvent 的速度（需等 GPU kernel 完成，毫秒级）。
//
//   错误时序（无 wevInserted）：
//     main 线程在几微秒内把所有 kernel 提交到 computeStream：
//       computeStream：[WaitEv(ev0)]→[k0]→[hf0]→[k1]→[hf1]→[k2]→[hf2]→[k3]→[hf3]
//     prepThread 在 k0 的 GPU 执行完成（毫秒后）才能插入 WaitEv(ev1)：
//       computeStream：...→[k3]→[hf3]→[WaitEv(ev1)]→[WaitEv(ev2)]→...
//     WaitEvent 排在 kernel 后面，对 kernel 无任何保护，k1/k2/k3 读到的是脏数据！
//
//   修复：wevInserted 初始值 0，main 循环用 if (r > 0) acquire()
//     r=0：main 自己插了 WaitEv(ev0)，直接提交 kernel 0，跳过 acquire
//     r>0：prepThread 在 cudaStreamWaitEvent 之后 release，main acquire 后才提交 kernel r
//   → 保证 WaitEv(h2dDone[r]) 一定排在 kernel r 前面入队
//   → 值域始终在 [0,1]，不会超过 counting_semaphore<1> 的最大值，无 UB
//
//   正确时序（有 wevInserted）：
//     computeStream：[WaitEv0]→[k0]→[hf0]→[WaitEv1]→[k1]→[hf1]→[WaitEv2]→[k2]→...
//                                           ↑
//                              prepThread release → main acquire → 才提交 k1
//
//   完整流水线时序图（NUM_BATCHES=4，buf 0/1 交替）：
//
//   时间 ────────────────────────────→
//
//   main(setup): H2D[0]→ev0→WaitEv(compute,ev0)  [wevInserted=1]
//
//   main 循环:   (r=0,跳过acq)→k0→hf0   acq(等prepThread)→k1→hf1   acq→k2→...
//                                  ↓notify(GPU完成)
//   prepThread:  acq(sem=1)             acq(sem)                    acq(sem)
//                →prep h1               →prep h0'                   →prep h1'
//                →H2D1→ev1              →H2D0'→ev0                  →H2D1'→ev1
//                →WaitEv(compute,ev1)   →WaitEv(compute,ev0)        →WaitEv(compute,ev1)
//                →rel(wevIns)           →rel(wevIns)                →rel(wevIns)
//
//   transferStream: [H2D 0]─ev0  [H2D 1]─ev1    [H2D 0']─ev0    [H2D 1']─ev1
//                          │            │                 │                │
//                     (WaitEvent)  (WaitEvent)       (WaitEvent)      (WaitEvent)
//                          ↓            ↓                 ↓                ↓
//   computeStream:  [WaitEv0─k0─hf0] [WaitEv1─k1─hf1] [WaitEv0─k2─hf2] [WaitEv1─k3─hf3]
//
//   说明：
//     ① main 线程预提交 H2D[0]，插入 WaitEv(ev0)，wevInserted 初始值 1 覆盖此次。
//     ② main 循环 r=0：acquire(wevInserted=1) 立即通过，提交 k0。
//        k0 完成 → hostFunc release(sem) → prepThread acquire → 准备 batch 1。
//     ③ prepThread：H2D[1] → EventRecord(ev1) → WaitEv(compute,ev1) → release(wevInserted)。
//        main 循环 r=1：acquire(wevInserted) 通过，提交 k1（此时 WaitEv(ev1) 已在 k1 前面）。
//     ④ transferStream 和 computeStream 真正并发：
//        GPU 执行 kernel r 时，transferStream 同时在传输 batch r+1 的数据。
//     ⑤ WaitEv 是 GPU 侧同步，CPU 不阻塞；wevInserted 是 CPU 侧握手，保证入队顺序。
// ════════════════════════
void demo_multi_stream_pipeline(void)
{
    printf("\n─── 示例 2：多 stream 流水线（传输与计算分离）───\n");

    float *d_in[NUM_BUFS], *d_out[NUM_BUFS];
    float *h_in[NUM_BUFS], *h_out;
    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaMalloc((void **)&d_in[i],  BATCH_BYTES));
        CUDA_CHECK(cudaMalloc((void **)&d_out[i], BATCH_BYTES));
        CUDA_CHECK(cudaMallocHost((void **)&h_in[i], BATCH_BYTES));
    }
    CUDA_CHECK(cudaMallocHost((void **)&h_out, BATCH_BYTES));

    // 两个独立 stream
    cudaStream_t transferStream, computeStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream,  cudaStreamNonBlocking));

    // event：标记 H2D 完成，让 computeStream 等待
    // cudaEventDisableTiming：只用于同步，不记时间戳，开销更低
    cudaEvent_t h2dDone[NUM_BUFS];
    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone[i], cudaEventDisableTiming));
    }

    // sem：GPU kernel 完成 → 通知 prepThread 可以准备下一批
    std::counting_semaphore<1> sem(1);  // 初始 1：允许立刻准备第 0 批（不等 GPU）
    std::vector<PipelineCtx> ctxs(NUM_BATCHES);
    for (int i = 0; i < NUM_BATCHES; i++) ctxs[i] = {&sem, i};

    // wevInserted：prepThread 插入 WaitEvent → 通知 main 可以提交对应 kernel
    // 初始值 0：r=0 由 main 自己插入 WaitEv，跳过 acquire；r>0 等 prepThread release
    // 为什么不用初始值 1 + 始终 acquire()？
    //   sem 初始值也是 1，prepThread 无需等 GPU 即可立即运行。
    //   若 prepThread 在 main 执行 r=0 的 acquire() 之前就调用了 release()，
    //   wevInserted 值从 1 变成 2，超过 counting_semaphore<1> 的最大值 1，
    //   属于未定义行为（UB）。
    //   用初始值 0 + if (r > 0) 则完全规避此问题：
    //     r=0：main 跳过 acquire，不存在 release 竞争
    //     r>0：prepThread release(0→1)，main acquire(1→0)，值域始终在 [0,1]
    // counting_semaphore 本身不是锁，是"事件通知"原语：
    //   · 锁（mutex）：保护共享资源，防止并发访问，持有者必须是解锁者
    //   · 信号量：传递"事件已发生"的信号，release/acquire 可由不同线程调用
    //   wevInserted 没有保护任何共享变量，只是 prepThread 告诉 main "可以继续了"
    std::counting_semaphore<1> wevInserted(0);

    // wevInserted.acquire() 的等待策略：spin-then-sleep（混合策略）
    //   ① 先 spin 若干次（忙等）：期望很快被 release，避免 OS 调度开销
    //   ② spin 超时仍未获取，调用 OS 原语（futex）睡眠，不再消耗 CPU
    //   适合本场景：prepThread 通常很快完成 cudaStreamWaitEvent 并 release，
    //   spin 阶段直接拿到，延迟极低；若意外延迟则退化为睡眠，不浪费 CPU。
    //
    // 对比 cv.wait(lock, pred) 的等待策略：纯 sleep
    //   调用后立即释放 mutex 并挂起到 OS 等待队列，
    //   线程从 CPU 调度中移除，不消耗 CPU，但唤醒延迟较高（微秒～毫秒级）。
    //   适合等待时间较长或不确定的场景。
    //
    // 本场景用 semaphore 比 cv.wait 更合适：等待时间短，spin 直接命中。

    int threads = 256, blocks = (BATCH_SIZE + threads - 1) / threads;

    // CPU 预处理线程：等 GPU kernel r 完成，准备 batch r+1 并插入 WaitEvent
    std::thread prepThread([&]() {
        for (int r = 0; r < NUM_BATCHES; r++) {
            sem.acquire();  // 等 GPU kernel r 完成（r=0 时初始值 1，立即通过）

            if (r < NUM_BATCHES - 1) {
                // 准备下一批 host 数据（模拟预处理）
                int nxt = (r + 1) % NUM_BUFS;
                float val = (float)(r + 1);
                for (int i = 0; i < BATCH_SIZE; i++) h_in[nxt][i] = val;

                // H2D 到 transferStream
                CUDA_CHECK(cudaMemcpyAsync(d_in[nxt], h_in[nxt], BATCH_BYTES,
                                           cudaMemcpyHostToDevice, transferStream));

                // 在 transferStream 上记录 event，标记此次 H2D 完成
                CUDA_CHECK(cudaEventRecord(h2dDone[nxt], transferStream));

                // 让 computeStream 等待此 H2D event（GPU 侧同步，CPU 不阻塞）
                CUDA_CHECK(cudaStreamWaitEvent(computeStream, h2dDone[nxt], 0));

                // WaitEvent 已插入 computeStream，通知 main 可以提交 kernel r+1
                // 必须在 cudaStreamWaitEvent 之后 release，保证 WaitEvent 先于 kernel 入队
                wevInserted.release();

                printf("  prepThread：batch %d H2D 已提交，WaitEvent 已插入\n", r + 1);
            }
        }
    });

    // 预填充第 0 批，main 自己插入 WaitEv(ev0)，r=0 跳过 acquire 无需 release
    for (int i = 0; i < BATCH_SIZE; i++) h_in[0][i] = 1.0f;
    CUDA_CHECK(cudaMemcpyAsync(d_in[0], h_in[0], BATCH_BYTES,
                               cudaMemcpyHostToDevice, transferStream));
    CUDA_CHECK(cudaEventRecord(h2dDone[0], transferStream));
    CUDA_CHECK(cudaStreamWaitEvent(computeStream, h2dDone[0], 0));

    // 主线程：每次提交 kernel 前先确认对应 WaitEvent 已入队
    for (int r = 0; r < NUM_BATCHES; r++) {
        int cur = r % NUM_BUFS;

        // r=0：WaitEv(ev0) 由 main 自己在循环前插入，直接提交 kernel，无需等待
        // r>0：等 prepThread 插完 WaitEv(h2dDone[r]) 后 release，再提交 kernel r
        if (r > 0) wevInserted.acquire();

        // 此时 computeStream 队列末尾已有 WaitEv(h2dDone[cur])，
        // kernel r 排在其后，GPU 侧保证 H2D 完成才执行 kernel
        processKernel<<<blocks, threads, 0, computeStream>>>(
            d_out[cur], d_in[cur], 2.0f, BATCH_SIZE);
        // 捕获启动错误（grid/block 参数非法等入队阶段的错误）
        // 不阻塞，不等 GPU 执行，纯 CPU 操作
        CUDA_CHECK(cudaGetLastError());

        // kernel 完成后通知 prepThread 准备下一批
        CUDA_CHECK(cudaLaunchHostFunc(computeStream, notify_batch_done, &ctxs[r]));
    }

    // D2H 最后一批结果
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out[(NUM_BATCHES - 1) % NUM_BUFS],
                               BATCH_BYTES, cudaMemcpyDeviceToHost, computeStream));

    printf("  主线程：所有 kernel 已入队\n");

    prepThread.join();
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(transferStream));

    printf("  多 stream 流水线完成，h_out[0] = %.2f\n", h_out[0]);

    for (int i = 0; i < NUM_BUFS; i++) {
        CUDA_CHECK(cudaEventDestroy(h2dDone[i]));
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaFreeHost(h_in[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
}


int main(void)
{
    demo_simple_pipeline();
    demo_multi_stream_pipeline();

    // ── 流水线设计要点 ───────
    //
    //   1. hostFunc 只做通知（sem.release()），不调用任何 CUDA API
    //      → 避免死锁（驱动 worker thread 持有内部 mutex 时调用 hostFunc）
    //
    //   2. PipelineCtx 生命周期必须覆盖 hostFunc 执行时间
    //      → 用 std::vector 存储，不用栈上临时变量
    //
    //   3. 双缓冲（ping-pong）：GPU 处理 buf[cur]，CPU 填充 buf[nxt]
    //      → cur = r % 2，nxt = (r+1) % 2，两者交替
    //
    //   4. semaphore 初始值 = 1：允许流水线启动时 CPU 立刻准备第 1 批
    //      → 不需要等 GPU 完成第 0 批才开始准备第 1 批
    //
    //   5. 多 stream 方案：transferStream 和 computeStream 分离
    //      用 cudaStreamWaitEvent 在 GPU 侧保证 H2D 先于 kernel（CPU 不阻塞）
    //
    // ── 通知原语选择 ────────
    //
    //   场景                          推荐原语
    //   ──────────────────────────    ───────────────────────────────────
    //   只需通知"完成"，无附加数据    counting_semaphore（本文件用法）
    //   需要传递 batch_id 等状态      condition_variable + 共享结构体
    //   极高频通知，追求最低开销      atomic<bool> + wait/notify（C++20）

    printf("\n全部示例完成。\n");
    return 0;
}
