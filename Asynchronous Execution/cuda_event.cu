/**
 * cuda_event.cu
 *
 * 详细说明 CUDA Event 相关 API：
 *
 *   cudaEventCreate / cudaEventCreateWithFlags
 *   cudaEventRecord
 *   cudaEventSynchronize
 *   cudaEventQuery
 *   cudaEventElapsedTime
 *   cudaEventDestroy
 *   cudaStreamWaitEvent
 *
 * 示例：
 *   1. 基本计时：cudaEventRecord + cudaEventElapsedTime
 *   2. 两种同步方式：cudaEventSynchronize（阻塞）vs cudaEventQuery（轮询）
 *   3. cudaEventRecord 循环记录（覆盖语义）
 *   4. cudaEventCreateWithFlags：Default vs BlockingSync
 *   5. cudaStreamWaitEvent vs cudaEventSynchronize 对比
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_event.cu -o cuda_event
 *
 * 运行：
 *   ./cuda_event
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ────────
// 错误检查宏
// ─────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",               \
                    __FILE__, __LINE__,                                     \
                    cudaGetErrorName(err), cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ───────────
// kernel：模拟耗时计算
// ───────────
__global__ void dummyKernel(float *out, float val, int n, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = val;
    for (int i = 0; i < iters; i++)
        x = x * 1.00001f + 0.00001f;
    out[idx] = x;
}

static const int    N      = 1 << 20;
static const size_t SIZE   = N * sizeof(float);
static const int    ITERS  = 2000;


// ════════════
// API 签名解析
// ═══════════════
//
// ── cudaEventCreate ──────
//
//   cudaError_t cudaEventCreate(cudaEvent_t *event)
//
//   创建一个 event，等价于：
//     cudaEventCreateWithFlags(event, cudaEventDefault)
//
//   cudaEvent_t 与 cudaStream_t 一样，是不透明句柄（opaque handle）：
//     typedef struct CUevent_st *cudaEvent_t;
//     CUevent_st 定义在驱动内部，外部不可见。
//     cudaEvent_t 变量存的是指向内部结构体的指针。
//
// ── cudaEventCreateWithFlags ──────
//
//   cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
//
//   flags 可选值：
//
//     cudaEventDefault      = 0x00
//       默认行为。
//       cudaEventSynchronize 内部使用 CPU 忙等（busy-wait / spin-wait）：
//         CPU 在一个紧密循环里不断轮询 GPU 状态，直到 event 完成。
//         延迟最低（响应快），但 CPU 核心被完全占用（100% CPU 占用率）。
//       适合：需要最低延迟的场景，CPU 不做其他事。
//
//     cudaEventBlockingSync = 0x01
//       cudaEventSynchronize 内部使用 OS 阻塞等待（让出 CPU）：
//         CPU 调用系统调用挂起线程，等 GPU 完成后由驱动唤醒。
//         CPU 核心释放给其他线程使用（CPU 占用率接近 0）。
//         但唤醒有 OS 调度延迟（通常 10~100 µs），响应慢于 spin-wait。
//       适合：不在乎微秒级同步精度，希望节省 CPU 资源的场景。
//
//     cudaEventDisableTiming = 0x02
//       不记录时间戳，不能用于 cudaEventElapsedTime。
//       开销更低（GPU 插入 event 时无需写时间戳），适合只用于同步的 event。
//
//     cudaEventInterprocess  = 0x04
//       用于进程间共享 event（IPC），需与 cudaEventDisableTiming 同时使用。
//
//   ── spin-wait vs blocking-wait 示意 ─────
//
//     cudaEventDefault（spin-wait）：
//       CPU: while (event_not_done) { /* 空转 */ }   ← 占满一个 CPU 核心
//       GPU: ─────────[kernel]─────── event ───────
//
//     cudaEventBlockingSync（blocking-wait）：
//       CPU: syscall(sleep) ──────── wakeup ←──── 驱动信号
//       GPU: ─────────[kernel]─────── event ───────
//       OS : 线程挂起               OS 调度唤醒（有延迟）
//
// ── cudaEventRecord ────────
//
//   cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
//
//   向 stream 的命令队列中插入一条"打时间戳"命令。
//   GPU 执行到这条命令时，将当前时刻的 GPU 时钟记录到 event 中。
//
//   重要：cudaEventRecord 是异步的。
//     调用 cudaEventRecord 时，命令仅入队，GPU 不一定立刻执行。
//     event 中的时间戳要等 GPU 实际执行到这条命令时才写入。
//
//   stream 参数：
//     传 0 或 nullptr → 记录到 Default Stream
//     传具体 stream   → 记录到该 stream
//
//   循环记录（pending ↔ completed 状态机）：
//
//     event 在生命周期中在两个状态之间反复切换：
//
//       ┌─────────────────────────────────────────────────────────────┐
//       │  状态          触发条件              cudaEventQuery 返回值  │
//       │  ──────────    ──────────────────    ─────────────────────  │
//       │  completed     GPU 执行完 Record     cudaSuccess            │
//       │  pending       CPU 调用 cudaEventRecord（命令入队）         │
//       │                                      cudaErrorNotReady      │
//       └─────────────────────────────────────────────────────────────┘
//
//     切换流程：
//       1. cudaEventCreate     → 初始状态：completed（未记录过，视为"已完成"）
//       2. cudaEventRecord(ev) → 状态立刻变为 pending（CPU 调用时即切换，不等 GPU）
//       3. GPU 执行 Record 命令 → 状态变为 completed，时间戳写入
//       4. 再次 cudaEventRecord(ev) → 状态再次变为 pending，旧时间戳被作废
//       5. GPU 再次执行 Record → 状态再次变为 completed，新时间戳写入
//       … 如此循环
//
//     关键：状态切换发生在 CPU 调用 cudaEventRecord 的瞬间（第 2、4 步），
//     不是在 GPU 执行完的时候。所以 Record 入队后立刻查询会得到 pending。
//
//     示例（带状态标注）：
//       // 初始：ev = completed（空）
//       cudaEventRecord(ev, stream);   // ev → pending（t1 入队）
//       kernel1<<<..., stream>>>();
//       cudaEventRecord(ev, stream);   // ev → pending（t2 入队，t1 被作废）
//       // GPU 执行顺序：Record(t1) → kernel1 → Record(t2)
//       // 但 t1 已被作废，ev 最终 = completed，时间戳 = t2
//
//     为什么重复 Record 是安全的？
//       stream 保证命令顺序执行，第 2 次 Record 命令一定在第 1 次 Record 命令之后执行，
//       驱动无需额外同步，直接覆盖时间戳即可。
//       如果需要保留多个时间点，必须创建多个 event（每个 event 独立状态机）。
//
// ── cudaEventSynchronize ───────
//
//   cudaError_t cudaEventSynchronize(cudaEvent_t event)
//
//   CPU 阻塞，直到 event 被 GPU 记录完成（时间戳已写入）。
//   等待粒度：单个 event，不等整个 stream。
//
//   等待方式由创建 event 时的 flags 决定：
//     cudaEventDefault      → spin-wait（忙等，低延迟）
//     cudaEventBlockingSync → blocking-wait（让出 CPU，低功耗）
//
//   典型用法：在读取 event 时间戳前调用，确保时间戳已写入：
//     cudaEventRecord(ev_end, stream);
//     cudaEventSynchronize(ev_end);          // 等 ev_end 被 GPU 记录
//     cudaEventElapsedTime(&ms, ev_start, ev_end);  // 此时读取安全
//
// ── cudaEventQuery ───────
//
//   cudaError_t cudaEventQuery(cudaEvent_t event)
//
//   非阻塞查询 event 状态，立刻返回：
//     cudaSuccess        event 已被 GPU 记录完成
//     cudaErrorNotReady  event 尚未被 GPU 记录
//
//   CPU 不阻塞，适合"做其他工作 + 间歇检查"的模式。
//
// ── cudaEventElapsedTime ───────
//
//   cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
//
//   计算两个 event 之间的 GPU 时间差，单位：毫秒（ms），精度约 0.5 µs。
//
//   调用前提：start 和 end 都必须已被 GPU 记录完成（否则返回错误）。
//   通常在 cudaEventSynchronize(end) 之后调用。
//
//   注意：测量的是 GPU 时钟时间，不是 CPU 时间。
//   若 start 和 end 在不同 stream 上，结果仍是 GPU 时间差（不受 stream 顺序影响）。
//
// ── cudaEventDestroy ──────
//
//   cudaError_t cudaEventDestroy(cudaEvent_t event)
//
//   销毁 event，释放驱动内部资源。
//   若 event 仍有未完成的 Record 或 WaitEvent，驱动会等其完成后再销毁。
//
// ── cudaStreamWaitEvent ─────────
//
//   cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
//
//   让 stream 等待 event（GPU 侧同步，CPU 不阻塞）：
//     stream 中在此命令之后提交的所有操作，都必须等到 event 被记录完成后才能执行。
//     CPU 调用 cudaStreamWaitEvent 后立刻返回，不阻塞。
//     等待发生在 GPU 内部（stream 的命令队列在 GPU 上暂停）。
//
//   flags：目前只有 0（保留参数，传 0）。
//
//   与 cudaEventSynchronize 的核心区别：
//
//     cudaEventSynchronize(event)
//       CPU 侧同步：CPU 等待 event，CPU 线程阻塞直到 event 完成。
//       其他 stream、其他 CPU 线程不受影响（只有调用线程阻塞）。
//       用途：CPU 需要知道 GPU 某个阶段已完成（如读取结果前）。
//
//     cudaStreamWaitEvent(stream, event, 0)
//       GPU 侧同步：stream 等待 event，CPU 不阻塞，继续执行。
//       stream 内部的命令队列在 GPU 上暂停，等 event 完成后继续。
//       用途：跨 stream 的 GPU 依赖，保证 stream B 在 stream A 的某个点之后才执行。
//
//   ── 对比表 ─────
//
//     函数                    谁在等      CPU 是否阻塞   典型用途
//     ───────────────────     ────────    ────────────   ──────────────────────────
//     cudaEventSynchronize    CPU 等      是（阻塞）      CPU 需要确认 GPU 完成某阶段
//     cudaStreamWaitEvent     GPU stream  否（不阻塞）    跨 stream 的 GPU 依赖关系
//
// ═════════════════


// ─────────────────
// 示例 1：基本计时
//
//   cudaEventRecord 标记 kernel 开始和结束，
//   cudaEventElapsedTime 计算耗时。
// ───────────────────
void demo_basic_timing(void)
{
    printf("\n─── 示例 1：基本计时 ───\n");

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 创建两个 event 用于计时
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ① Record start：向 stream 插入"打时间戳"命令
    //   此时 GPU 还未执行，只是命令入队
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    // ② 提交 kernel
    dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N, ITERS);

    // ③ Record end：kernel 之后插入时间戳命令
    //   GPU 按顺序执行：ev_start → kernel → ev_end
    CUDA_CHECK(cudaEventRecord(ev_end, stream));

    // ④ CPU 等待 ev_end 被 GPU 记录完成
    //   未调用此函数就直接 ElapsedTime 会得到错误（时间戳未写入）
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    // ⑤ 计算时间差（单位 ms）
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
    printf("  kernel 耗时：%.3f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ─────────────────────────────────────────────────────────────────────
// 示例 2：两种同步方式
//
//   方式 A：cudaEventSynchronize（阻塞等待，CPU 不做其他事）
//   方式 B：cudaEventQuery（非阻塞轮询，CPU 间歇检查）
// ─────────────────────────────────────────────────────────────────────
void demo_sync_vs_query(void)
{
    printf("\n─── 示例 2：cudaEventSynchronize vs cudaEventQuery ───\n");

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ── 方式 A：cudaEventSynchronize（阻塞）─────────────────────────
    {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));

        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N, ITERS * 5);
        CUDA_CHECK(cudaEventRecord(ev, stream));

        // CPU 在此阻塞，直到 ev 被 GPU 记录完成
        // 内部使用 spin-wait（cudaEventDefault），CPU 核心持续忙等
        CUDA_CHECK(cudaEventSynchronize(ev));

        printf("  [Synchronize] 阻塞等待完成，CPU 在等待期间被占用\n");
        CUDA_CHECK(cudaEventDestroy(ev));
    }

    // ── 方式 B：cudaEventQuery（非阻塞轮询）─────────────────────────
    {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));

        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 2.0f, N, ITERS * 5);
        CUDA_CHECK(cudaEventRecord(ev, stream));

        // CPU 不阻塞，循环检查 event 状态，同时可做其他工作
        int poll_count = 0;
        while (true) {
            cudaError_t status = cudaEventQuery(ev);

            if (status == cudaSuccess) {
                // event 已被 GPU 记录完成
                printf("  [Query] 轮询完成，共查询 %d 次，CPU 在等待期间可做其他工作\n",
                       poll_count);
                break;
            } else if (status == cudaErrorNotReady) {
                // event 尚未完成，CPU 继续做其他工作
                poll_count++;
                // 此处可插入 CPU 侧的实际工作（如准备下一批数据）
            } else {
                CUDA_CHECK(status);  // 其他错误
            }
        }

        CUDA_CHECK(cudaEventDestroy(ev));
    }

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ─────────────────────────────────────────────────────────────────────
// 示例 3：cudaEventRecord 循环记录（覆盖语义）
//
//   同一个 event 可以被反复 Record，每次覆盖上一次的时间戳。
//   常见用法：在循环里复用 event 而无需每次创建销毁。
// ─────────────────────────────────────────────────────────────────────
void demo_reuse_event(void)
{
    printf("\n─── 示例 3：cudaEventRecord 循环记录（覆盖语义）───\n");

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // 只创建一对 event，在循环里复用
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    const int ROUNDS = 3;
    for (int r = 0; r < ROUNDS; r++) {
        // 每轮 Record 都覆盖上一轮写入的时间戳
        // 驱动保证本轮的 Record 在上一轮的操作完成后才执行（stream 内顺序）
        CUDA_CHECK(cudaEventRecord(ev_start, stream));

        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, (float)r, N, ITERS);

        CUDA_CHECK(cudaEventRecord(ev_end, stream));

        // 等待本轮 ev_end 完成后读取时间（覆盖上一轮的值）
        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        printf("  第 %d 轮 kernel 耗时：%.3f ms\n", r, ms);

        // 无需 destroy/recreate event，直接进入下一轮
        // 下一轮 Record 时会自动覆盖当前时间戳
    }

    // ── 覆盖语义的进一步说明 ──────────────────────────────────────
    //
    //   同一个 event 在 stream 里多次 Record，命令队列示意：
    //
    //     stream: [Record ev t0] [kernel0] [Record ev t1] [kernel1] [Record ev t2]
    //                                                                ↑ 最终时间戳
    //
    //   GPU 执行完所有命令后，ev 里存的是 t2（最后一次 Record 的时间戳）。
    //   t0、t1 被覆盖，无法再读取。
    //   如果需要保留多个时间点，应创建多个 event。
    //
    printf("  （覆盖语义：同一 event 多次 Record，最终值为最后一次的时间戳）\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ─────────────────────────────────────────────────────────────────────
// 示例 4：cudaEventCreateWithFlags
//
//   cudaEventDefault     → spin-wait（忙等），低延迟，高 CPU 占用
//   cudaEventBlockingSync → blocking-wait（让出 CPU），有调度延迟，低 CPU 占用
// ─────────────────────────────────────────────────────────────────────
void demo_event_flags(void)
{
    printf("\n─── 示例 4：cudaEventCreateWithFlags ───\n");

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ── cudaEventDefault（spin-wait）────────────────────────────────
    {
        cudaEvent_t ev_start, ev_end;
        // cudaEventDefault = 0x00：cudaEventSynchronize 内部忙等
        // CPU 核心在等待期间持续轮询，响应最快但消耗 CPU 资源
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_start, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_end,   cudaEventDefault));

        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N, ITERS);
        CUDA_CHECK(cudaEventRecord(ev_end, stream));

        // spin-wait：CPU 持续忙等，直到 ev_end 完成
        // 适合：需要最低同步延迟（如实时渲染、高频控制循环）
        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        printf("  [Default/spin-wait]    耗时：%.3f ms  （CPU 核心忙等，响应快）\n", ms);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_end));
    }

    // ── cudaEventBlockingSync（blocking-wait）───────────────────────
    {
        cudaEvent_t ev_start, ev_end;
        // cudaEventBlockingSync = 0x01：cudaEventSynchronize 内部让出 CPU
        // 线程挂起，等 GPU 完成后由驱动通过 OS 信号唤醒
        // 唤醒有 OS 调度延迟（通常 10~100 µs），但 CPU 核心可被其他线程使用
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_start, cudaEventBlockingSync));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_end,   cudaEventBlockingSync));

        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 2.0f, N, ITERS);
        CUDA_CHECK(cudaEventRecord(ev_end, stream));

        // blocking-wait：CPU 线程挂起，释放 CPU 核心给其他线程
        // 适合：多线程程序、服务端推理等 CPU 资源紧张的场景
        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        printf("  [BlockingSync/sleep]   耗时：%.3f ms  （CPU 核心让出，有唤醒延迟）\n", ms);

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_end));
    }

    // ── cudaEventDisableTiming（纯同步用途）─────────────────────────
    {
        cudaEvent_t ev;
        // cudaEventDisableTiming = 0x02：不记录时间戳
        // GPU 插入 event 时无需写时间戳，开销略低
        // 不能用于 cudaEventElapsedTime，只能用于同步（cudaStreamWaitEvent 等）
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

        dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 3.0f, N, ITERS);
        CUDA_CHECK(cudaEventRecord(ev, stream));
        CUDA_CHECK(cudaEventSynchronize(ev));
        printf("  [DisableTiming]        同步完成（无时间戳，不可 ElapsedTime）\n");

        CUDA_CHECK(cudaEventDestroy(ev));
    }

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ──────────────────────
// 示例 5：cudaStreamWaitEvent vs cudaEventSynchronize
//
//   问题场景：
//     stream A 生产数据（kernel A），stream B 消费数据（kernel B）。
//     需要保证 kernel B 在 kernel A 完成后才执行。
//
//   方案 1：cudaEventSynchronize（CPU 等 GPU，再提交 B）
//     CPU 在 kernel A 完成前阻塞，完成后才提交 kernel B。
//     两段 GPU 工作之间有 CPU 介入，存在额外延迟。
//
//   方案 2：cudaStreamWaitEvent（GPU 等 GPU，CPU 全程不阻塞）
//     CPU 一次性提交 kernel A、WaitEvent、kernel B，全部入队后立刻返回。
//     GPU 内部保证 kernel B 在 kernel A 完成后才执行，CPU 无需介入。
// ──────────────────────
void demo_stream_wait_event(void)
{
    printf("\n─── 示例 5：cudaStreamWaitEvent vs cudaEventSynchronize ───\n");

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_b, SIZE));

    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ── 方案 1：cudaEventSynchronize（CPU 等 GPU）───────────────────
    //
    //   时间轴：
    //     sA:  [kernel A] ─────────────────
    //     CPU: ─── EventSynchronize(阻塞) ─── 提交 kernel B ──
    //     sB:                               ─ [kernel B] ───────
    //
    //   缺点：CPU 被阻塞，GPU 可能出现空泡（CPU 解阻后才提交 B）。
    {
        cudaEvent_t ev_a_done;
        CUDA_CHECK(cudaEventCreate(&ev_a_done));

        // sA 提交 kernel A，然后 Record ev_a_done
        dummyKernel<<<blocks, threads, 0, sA>>>(d_a, 1.0f, N, ITERS);
        CUDA_CHECK(cudaEventRecord(ev_a_done, sA));

        // CPU 阻塞等 kernel A 完成
        CUDA_CHECK(cudaEventSynchronize(ev_a_done));  // CPU 在此等待

        // CPU 解阻后，向 sB 提交 kernel B
        dummyKernel<<<blocks, threads, 0, sB>>>(d_b, 2.0f, N, ITERS);

        CUDA_CHECK(cudaStreamSynchronize(sB));
        printf("  [方案 1 EventSynchronize] CPU 等 kernel A → 再提交 kernel B\n");
        printf("    CPU 被阻塞，kernel A 和 kernel B 之间可能有空泡\n");

        CUDA_CHECK(cudaEventDestroy(ev_a_done));
    }

    // ── 方案 2：cudaStreamWaitEvent（GPU 等 GPU，CPU 不阻塞）────────
    //
    //   时间轴：
    //     sA:  [kernel A] ───────
    //     sB:  ──[WaitEvent]────────────── [kernel B] ───
    //     CPU: 提交完所有命令后立刻返回，不阻塞
    //
    //   GPU 内部机制：
    //     sB 的命令队列在 WaitEvent 处暂停，
    //     等 sA 的 ev_a_done 被记录后，sB 自动继续执行 kernel B。
    //     CPU 全程不介入，GPU 流水线更紧凑。
    {
        cudaEvent_t ev_a_done;
        // 只用于同步，不需要时间戳，使用 DisableTiming 减少开销
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_a_done, cudaEventDisableTiming));

        // sA 提交 kernel A，然后 Record ev_a_done
        dummyKernel<<<blocks, threads, 0, sA>>>(d_a, 1.0f, N, ITERS);
        CUDA_CHECK(cudaEventRecord(ev_a_done, sA));

        // 告知 sB：在 ev_a_done 完成之前，sB 中的后续命令不得执行
        // CPU 不阻塞，立刻返回
        CUDA_CHECK(cudaStreamWaitEvent(sB, ev_a_done, 0));

        // 立刻提交 kernel B（它在 GPU 侧等待 ev_a_done，不在 CPU 侧等）
        dummyKernel<<<blocks, threads, 0, sB>>>(d_b, 2.0f, N, ITERS);

        // 等两个 stream 都完成
        CUDA_CHECK(cudaStreamSynchronize(sA));
        CUDA_CHECK(cudaStreamSynchronize(sB));
        printf("  [方案 2 StreamWaitEvent] GPU 内部等待，CPU 不阻塞\n");
        printf("    CPU 提交完所有命令后立刻返回，GPU 自动保证 A → B 顺序\n");

        CUDA_CHECK(cudaEventDestroy(ev_a_done));
    }

    // ── 对比总结 ────────────
    //
    //   维度              cudaEventSynchronize    cudaStreamWaitEvent
    //   ──────────────    ────────────────────    ──────────────────────────
    //   等待发生在        CPU 侧                  GPU 侧（stream 内部）
    //   CPU 是否阻塞      是                      否
    //   GPU 流水线        可能有空泡              紧凑（无 CPU 介入）
    //   典型用途          CPU 需要读取 GPU 结果   跨 stream 的纯 GPU 依赖
    //   额外开销          CPU 调度 + 唤醒延迟      几乎无（GPU 内部信号）
    //
    printf("  （详见注释中的对比表）\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
}


int main(void)
{
    demo_basic_timing();
    demo_sync_vs_query();
    demo_reuse_event();
    demo_event_flags();
    demo_stream_wait_event();

    // ── 速查：Event 创建 flags ────────
    //
    //   flag                    值      含义
    //   ─────────────────────   ─────   ────────────
    //   cudaEventDefault        0x00    spin-wait，低延迟，高 CPU 占用
    //   cudaEventBlockingSync   0x01    blocking-wait，低 CPU 占用，有调度延迟
    //   cudaEventDisableTiming  0x02    不记时间戳，只用于同步，开销低
    //   cudaEventInterprocess   0x04    进程间共享（须与 DisableTiming 同用）
    //
    // ── 速查：Event 同步方式 ─────────
    //
    //   函数                    阻塞    等待粒度    返回条件
    //   ─────────────────────   ──────  ──────────  ────────
    //   cudaEventSynchronize    CPU     单个 event  event 被 GPU 记录完成
    //   cudaEventQuery          否      单个 event  立刻，返回状态码
    //   cudaStreamWaitEvent     GPU     单个 event  GPU stream 内部等待
    //
    // ── 速查：cudaStreamWaitEvent vs cudaEventSynchronize ───────────
    //
    //   场景                                推荐
    //   ────────────────────────────────    ─────────────
    //   CPU 需要读取 GPU 计算结果            cudaEventSynchronize
    //   跨 stream 的 GPU 数据依赖            cudaStreamWaitEvent
    //   循环流水线（双缓冲、生产消费）        cudaStreamWaitEvent（更高效）

    printf("\n全部示例完成。\n");
    return 0;
}
