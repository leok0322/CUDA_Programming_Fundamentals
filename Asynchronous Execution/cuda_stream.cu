/**
 * cuda_stream.cu
 *
 * 详细说明 CUDA Stream 相关 API，并给出并发示例：
 *
 * API：
 *   cudaStreamCreate
 *   cudaStreamCreateWithFlags
 *   cudaStreamCreateWithPriority
 *   cudaStreamSynchronize
 *   cudaStreamQuery
 *   cudaDeviceSynchronize
 *   cudaStreamDestroy
 *
 * 示例：
 *   1. Default Stream（隐式同步屏障）
 *   2. Blocking Stream（与 Default Stream 同步）
 *   3. Non-Blocking Stream（与 Default Stream 独立并发）
 *   4. 带优先级的 Stream
 *   5. cudaStreamQuery 非阻塞轮询
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_stream.cu -o cuda_stream
 *
 * 运行：
 *   ./cuda_stream
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ───────────────
// 错误检查宏
// ─────────────
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

// ────────────
// kernel：模拟耗时计算（空循环）
// ────────────
__global__ void dummyKernel(float *out, float val, int n, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = val;
    for (int i = 0; i < iters; i++)
        x = x * 1.00001f + 0.00001f;  // 防止编译器优化掉循环
    out[idx] = x;
}

static const int    N      = 1 << 20;
static const size_t SIZE   = N * sizeof(float);
static const int    ITERS  = 1000;   // 控制 kernel 耗时

// ═════════════════
// API 签名解析
// ═════════════════

// ── cudaStreamCreate ──────────

//   cudaError_t cudaStreamCreate(cudaStream_t *pStream)

//   pStream：[out] 新建的 stream 句柄写入此指针

//   创建一个 blocking stream（默认行为）：
//     与 Default Stream 之间存在隐式同步屏障。
//     等价于 cudaStreamCreateWithFlags(pStream, cudaStreamDefault)

// ── cudaStreamCreateWithFlags ────────

//   cudaError_t cudaStreamCreateWithFlags(
//       cudaStream_t *pStream,
//       unsigned int  flags
//   )

//   flags：
//     cudaStreamDefault     = 0x00  blocking stream，与 Default Stream 同步
//     cudaStreamNonBlocking = 0x01  non-blocking stream，与 Default Stream 独立

//   blocking vs non-blocking 的核心区别：

//     Default Stream（stream 0）是一个特殊的隐式同步点：
//       - 向 Default Stream 提交操作前，所有 blocking stream 的已提交操作必须完成
//       - Default Stream 的操作完成后，所有 blocking stream 才能继续执行

//     blocking stream：受 Default Stream 的隐式同步影响
//     non-blocking stream：完全独立，不受 Default Stream 影响，可与其真正并发

//   示意图：

//     blocking stream 模式：
//       stream A (blocking): [kernel A1] ───────────────── [kernel A2]
//       Default  stream   :             [barrier][kernel D][barrier]
//       stream B (blocking): [kernel B1] ───────────────── [kernel B2]
//                                        ↑ A1/B1 必须完成  ↑ D 必须完成
//                                          Default 才能开始  A2/B2 才能开始

//     non-blocking stream 模式：
//       stream A (non-blocking): [kernel A1][kernel A2]  （完全不受 Default 影响）
//       Default  stream        :     [kernel D]
//       stream B (non-blocking): [kernel B1][kernel B2]  （完全不受 Default 影响）

// ── cudaStreamCreateWithPriority ────────

//   cudaError_t cudaStreamCreateWithPriority(
//       cudaStream_t *pStream,
//       unsigned int  flags,
//       int           priority
//   )

//   priority：
//     数值越小优先级越高（与直觉相反）。
//     合法范围通过 cudaDeviceGetStreamPriorityRange 查询：
//       int leastPriority, greatestPriority;
//       cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
//       // greatestPriority <= priority <= leastPriority
//       // 典型值：greatestPriority = -1，leastPriority = 0

//   优先级影响 GPU 调度器在多个 stream 竞争资源时的选择顺序，
//   不保证低优先级 stream 完全饿死，只是高优先级 stream 优先获得资源。

// ── cudaStreamSynchronize ──────

//   cudaError_t cudaStreamSynchronize(cudaStream_t stream)

//   stream 参数：按值传递（pass by value），不是指针也不是引用。

//   为什么传值就等于传了"句柄"？
//     cudaStream_t 本质是一个指针类型，定义如下：

//       typedef struct CUstream_st *cudaStream_t;
//       │       │                  │
//       │       │                  └─ * 表示"指针"，包含在 typedef 里
//       │       └─ 只声明结构体名，不暴露内部成员（定义在 CUDA 驱动内部）
//       └─ 给"指向 CUstream_st 的指针"这整个类型起别名 cudaStream_t

//     拆开等价于：
//       struct CUstream_st;                  // 前向声明，内容不可见（不透明）
//       typedef struct CUstream_st * cudaStream_t;  // 指针别名

//     所以 cudaStream_t s 实际上是：
//       struct CUstream_st *s;   // s 存的是驱动内部结构体的地址

//     cudaStream_t 的值本身就是一个指针（内存地址），
//     按值传递这个地址，函数内部可通过它找到对应的 stream 对象。
//     无需再传 cudaStream_t*（二级指针）。

//     这种设计叫 Opaque Handle（不透明句柄）：
//       你持有地址，但看不到也改不了结构体内部，所有操作只能通过 CUDA API 完成。
//       FILE* 是同类设计：typedef struct _IO_FILE FILE; 然后用 FILE *fp。
//       区别在于 FILE 别名不含 *，用时要写 FILE *fp；
//       而 cudaStream_t 别名含 *，所以直接写 cudaStream_t s（不加 *）。

//   对比 cudaStreamCreate 为何需要 cudaStream_t*：
//     cudaStreamCreate 要向外写出一个新建的句柄，
//     所以需要 cudaStream_t*（指向句柄变量的指针）才能修改调用方的变量。
//     Synchronize / Query 只是读取句柄，传值即可。

//   Default Stream 传 0（或 nullptr）：
//     cudaStreamSynchronize(0);   // 0 即空指针，代表 Default Stream

//   CPU 阻塞，直到指定 stream 中所有已提交的操作全部完成。
//   只等待这一个 stream，其他 stream 继续执行。

// ── cudaStreamQuery ──────

//   cudaError_t cudaStreamQuery(cudaStream_t stream)

//   stream 参数：同 cudaStreamSynchronize，按值传递 cudaStream_t 句柄。

//   非阻塞查询 stream 状态，立刻返回：
//     cudaSuccess        stream 中所有操作已完成
//     cudaErrorNotReady  stream 中还有操作未完成

//   CPU 不阻塞，适合轮询或"做其他工作 + 间歇检查"的模式。

//   ── cudaStream_t 参数传递总结 ──────────────────────────────────────

//     函数                          参数类型           原因
//     ──────────────────────────    ────────────────   ──────────────────────────
//     cudaStreamCreate              cudaStream_t *     需要写出新建的句柄（输出参数）
//     cudaStreamCreateWithFlags     cudaStream_t *     同上
//     cudaStreamCreateWithPriority  cudaStream_t *     同上
//     cudaStreamSynchronize         cudaStream_t       只读句柄，传值即可
//     cudaStreamQuery               cudaStream_t       只读句柄，传值即可
//     cudaStreamDestroy             cudaStream_t       只读句柄，传值即可
//     cudaMemcpyAsync（最后参数）   cudaStream_t       只读句柄，传值即可

// ── cudaDeviceSynchronize ─────

//   cudaError_t cudaDeviceSynchronize(void)

//   CPU 阻塞，直到该设备上**所有 stream**的所有操作全部完成。
//   比 cudaStreamSynchronize 粒度更粗，等待范围更广。
//   适合：调试、程序退出前的最终同步。

// ── cudaStreamDestroy ────

//   cudaError_t cudaStreamDestroy(cudaStream_t stream)

//   销毁 stream，释放相关资源。
//   若 stream 中还有未完成的操作，cudaStreamDestroy 会等待它们完成后再销毁。
//   销毁后 stream 句柄失效，不能再使用。

// ════════════


// ─────────
// 示例 1：Default Stream 的隐式同步屏障
//
//   Default Stream（stream 0）是隐式同步点：
//     向 Default Stream 提交操作时，所有 blocking stream 的已有操作必须先完成；
//     Default Stream 操作完成后，blocking stream 才能继续。
//
//   时间轴：
//     stream A (blocking): [H2D A]──────────────────────[kernel A]
//     Default  stream    :         [barrier][kernel D][barrier]
//     stream B (blocking): [H2D B]──────────────────────[kernel B]
// ────────────
void demo_default_stream(void)
{
    printf("\n─── 示例 1：Default Stream 隐式同步屏障 ───\n");

    float *h_a, *h_b, *d_a, *d_b;
    CUDA_CHECK(cudaMallocHost((void **)&h_a, SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_b, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_b, SIZE));

    // 初始化 host 数据
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // cudaStreamCreate 创建 blocking stream（默认）
    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreate(&sA));   // blocking stream
    CUDA_CHECK(cudaStreamCreate(&sB));   // blocking stream

    int threads = 256, blocks = (N + threads - 1) / threads;

    // sA 和 sB 各自提交 H2D 传输（两者可并行）
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, SIZE, cudaMemcpyHostToDevice, sA));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, SIZE, cudaMemcpyHostToDevice, sB));

    // 向 Default Stream 提交 kernel：
    // 此时 CUDA 运行时隐式插入屏障：
    //   等 sA 和 sB 的 H2D 都完成后，Default Stream 的 kernel 才开始
    dummyKernel<<<blocks, threads>>>(d_a, 1.0f, N, ITERS);  // Default Stream
    CUDA_CHECK(cudaGetLastError());
    printf("  Default Stream kernel 提交，隐式等待 sA、sB 的 H2D 完成\n");

    // Default Stream kernel 完成后，sA/sB 的后续操作才能执行
    dummyKernel<<<blocks, threads, 0, sA>>>(d_a, 2.0f, N, ITERS);
    dummyKernel<<<blocks, threads, 0, sB>>>(d_b, 3.0f, N, ITERS);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  所有操作完成（Default Stream 充当了全局屏障）\n");

    CUDA_CHECK(cudaFreeHost(h_a)); CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFree(d_a));     CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
}

// ─────────────
// 示例 2：Blocking Stream vs Non-Blocking Stream 并发对比
//
//   blocking stream：受 Default Stream 隐式屏障影响，不能与 Default 真正并发
//   non-blocking stream：完全独立，不受 Default Stream 影响
//
//   验证方法：
//     在 non-blocking stream 上提交长 kernel，
//     然后向 Default Stream 提交操作，观察两者是否真正并发。
// ─────────────
void demo_blocking_vs_nonblocking(void)
{
    printf("\n─── 示例 2：Blocking vs Non-Blocking Stream ───\n");

    float *d_buf0, *d_buf1, *d_buf2;
    CUDA_CHECK(cudaMalloc((void **)&d_buf0, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_buf1, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_buf2, SIZE));

    // blocking stream：cudaStreamCreate 或 cudaStreamCreateWithFlags(..., cudaStreamDefault)
    cudaStream_t sBlocking;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sBlocking, cudaStreamDefault));

    // non-blocking stream：cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)
    cudaStream_t sNonBlocking;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sNonBlocking, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ── blocking stream 测试 ──────────────────────────────────────
    // sBlocking 提交 kernel，然后向 Default Stream 提交操作
    // Default Stream 必须等 sBlocking 的 kernel 完成后才能开始

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, sBlocking));
    dummyKernel<<<blocks, threads, 0, sBlocking>>>(d_buf0, 1.0f, N, ITERS);
    // Default Stream 操作：隐式等待 sBlocking 完成
    dummyKernel<<<blocks, threads>>>(d_buf1, 2.0f, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev1, 0));  // Default Stream
    // blocking 模式下 Default Stream 必须等 sBlocking 完成才能开始，
    // 所以只需同步 Default Stream（stream 0），sBlocking 必然已完成。
    CUDA_CHECK(cudaStreamSynchronize(0));

    float ms_blocking = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_blocking, ev0, ev1));
    printf("  [Blocking]     sBlocking + Default Stream 总耗时：%.2f ms"
           "（串行，Default 等 sBlocking）\n", ms_blocking);

    // ── non-blocking stream 测试 ──────────────────────────────────
    // sNonBlocking 提交 kernel，Default Stream 不等它，两者真正并发

    cudaEvent_t ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventCreate(&ev3));

    CUDA_CHECK(cudaEventRecord(ev2, sNonBlocking));
    dummyKernel<<<blocks, threads, 0, sNonBlocking>>>(d_buf0, 1.0f, N, ITERS);
    // Default Stream 操作：不等 sNonBlocking，立刻开始
    dummyKernel<<<blocks, threads>>>(d_buf2, 2.0f, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev3, 0));  // Default Stream
    // non-blocking 模式下两者完全独立，必须分别同步：
    // 先等 Default Stream（ev3 在此 stream，保证计时可读），
    // 再等 sNonBlocking（与 Default 无依赖，需单独等）。
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaStreamSynchronize(sNonBlocking));

    float ms_nonblocking = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_nonblocking, ev2, ev3));
    printf("  [Non-Blocking] sNonBlocking + Default Stream 总耗时：%.2f ms"
           "（并发，Default 不等 sNonBlocking）\n", ms_nonblocking);
    printf("  Non-Blocking 耗时应短于 Blocking（两个 kernel 并行执行）\n");

    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaEventDestroy(ev2)); CUDA_CHECK(cudaEventDestroy(ev3));
    CUDA_CHECK(cudaFree(d_buf0));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));
    CUDA_CHECK(cudaStreamDestroy(sBlocking));
    CUDA_CHECK(cudaStreamDestroy(sNonBlocking));
}

// ────────────────
// 示例 3：带优先级的 Stream
//
//   高优先级 stream 的 kernel 在 GPU 调度时优先获得资源。
//   数值越小优先级越高（greatestPriority 通常为 -1，leastPriority 为 0）。
// ──────────────────
void demo_priority_stream(void)
{
    printf("\n─── 示例 3：带优先级的 Stream ───\n");

    // 查询设备支持的优先级范围
    int leastPriority    = 0;
    int greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    printf("  优先级范围：greatest（最高）= %d，least（最低）= %d\n",
           greatestPriority, leastPriority);
    // 典型输出：greatest = -1，least = 0
    // 数值越小优先级越高

    // 创建高优先级 stream（non-blocking，不受 Default Stream 干扰）
    cudaStream_t sHigh, sLow;
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &sHigh,
        cudaStreamNonBlocking,
        greatestPriority    // 最高优先级
    ));

    // 创建低优先级 stream
    CUDA_CHECK(cudaStreamCreateWithPriority(
        &sLow,
        cudaStreamNonBlocking,
        leastPriority       // 最低优先级
    ));

    float *d_high, *d_low;
    CUDA_CHECK(cudaMalloc((void **)&d_high, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_low,  SIZE));

    int threads = 256, blocks = (N + threads - 1) / threads;

    cudaEvent_t ev_start, ev_high_done, ev_low_done;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_high_done));
    CUDA_CHECK(cudaEventCreate(&ev_low_done));

    // 同时提交高低优先级 kernel，观察完成顺序
    CUDA_CHECK(cudaEventRecord(ev_start, sHigh));
    dummyKernel<<<blocks, threads, 0, sLow >>>(d_low,  1.0f, N, ITERS * 2);
    dummyKernel<<<blocks, threads, 0, sHigh>>>(d_high, 2.0f, N, ITERS * 2);
    CUDA_CHECK(cudaEventRecord(ev_high_done, sHigh));
    CUDA_CHECK(cudaEventRecord(ev_low_done,  sLow));

    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_high = 0.f, ms_low = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_high, ev_start, ev_high_done));
    CUDA_CHECK(cudaEventElapsedTime(&ms_low,  ev_start, ev_low_done));
    printf("  高优先级 stream 完成耗时：%.2f ms\n", ms_high);
    printf("  低优先级 stream 完成耗时：%.2f ms\n", ms_low);
    printf("  高优先级应更早完成（GPU 调度优先分配资源）\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_high_done));
    CUDA_CHECK(cudaEventDestroy(ev_low_done));
    CUDA_CHECK(cudaFree(d_high));
    CUDA_CHECK(cudaFree(d_low));
    CUDA_CHECK(cudaStreamDestroy(sHigh));
    CUDA_CHECK(cudaStreamDestroy(sLow));
}

// ────────────
// 示例 4：cudaStreamQuery 非阻塞轮询
//
//   cudaStreamQuery 不阻塞 CPU，立刻返回 stream 当前状态。
//   适合"CPU 做其他工作，间歇检查 GPU 是否完成"的模式。
// ──────────────
void demo_stream_query(void)
{
    printf("\n─── 示例 4：cudaStreamQuery 非阻塞轮询 ───\n");

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // 提交一个耗时 kernel
    dummyKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N, ITERS * 10);

    // CPU 不阻塞，轮询检查 stream 状态，同时做"其他工作"
    int poll_count = 0;
    while (true) {
        cudaError_t status = cudaStreamQuery(stream);

        if (status == cudaSuccess) {
            // stream 中所有操作已完成
            printf("  stream 完成，共轮询 %d 次\n", poll_count);
            break;
        } else if (status == cudaErrorNotReady) {
            // stream 还在执行，CPU 继续做其他工作
            poll_count++;
            // 模拟 CPU 做其他工作（此处仅计数）
        } else {
            // 其他错误
            CUDA_CHECK(status);
        }
    }

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ───────────────
// 示例 5：cudaStreamSynchronize vs cudaDeviceSynchronize 粒度对比
//
//   cudaStreamSynchronize：只等指定 stream，其他 stream 继续执行
//   cudaDeviceSynchronize：等所有 stream 的所有操作全部完成
// ─────────────────
void demo_sync_granularity(void)
{
    printf("\n─── 示例 5：cudaStreamSynchronize vs cudaDeviceSynchronize ───\n");

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_b, SIZE));

    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    dummyKernel<<<blocks, threads, 0, sA>>>(d_a, 1.0f, N, ITERS);
    dummyKernel<<<blocks, threads, 0, sB>>>(d_b, 2.0f, N, ITERS * 3);  // sB 更慢

    // 只等 sA，不等 sB
    // sB 的 kernel 此时仍在执行
    CUDA_CHECK(cudaStreamSynchronize(sA));
    printf("  cudaStreamSynchronize(sA) 返回：sA 完成，sB 可能仍在执行\n");

    // 等所有 stream（包括 sB）全部完成
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  cudaDeviceSynchronize() 返回：所有 stream 全部完成\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
}

int main(void)
{
    demo_default_stream();
    demo_blocking_vs_nonblocking();
    demo_priority_stream();
    demo_stream_query();
    demo_sync_granularity();

    // ── 速查：Stream 创建函数对比 ────────
    //
    //   函数                          flags                  priority
    //   ─────────────────────────     ──────────────────     ────────
    //   cudaStreamCreate              cudaStreamDefault      默认
    //   cudaStreamCreateWithFlags     可指定 Non-Blocking    默认
    //   cudaStreamCreateWithPriority  可指定 Non-Blocking    可指定
    //
    // ── 速查：同步函数对比 ────────────
    //
    //   函数                      阻塞范围              返回时机
    //   ───────────────────────   ─────────────────     ──────────────────
    //   cudaStreamSynchronize     指定 stream           该 stream 全部完成
    //   cudaDeviceSynchronize     所有 stream           所有操作全部完成
    //   cudaStreamQuery           不阻塞（立刻返回）    立刻，返回状态码
    //
    // ── 速查：Blocking vs Non-Blocking ────────
    //
    //   类型              与 Default Stream 关系
    //   ───────────────   ─────────────────
    //   Blocking stream   Default Stream 前后有隐式屏障，互相等待
    //   Non-Blocking      完全独立，不受 Default Stream 影响，真正并发

    printf("\n全部示例完成。\n");
    return 0;
}
