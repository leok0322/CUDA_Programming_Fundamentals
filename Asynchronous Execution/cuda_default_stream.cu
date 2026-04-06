/**
 * cuda_default_stream.cu
 *
 * cudaStreamLegacy 与 cudaStreamPerThread 详解与示例
 *
 * 编译（Legacy 模式，默认）：
 *   nvcc -arch=native -std=c++17 cuda_default_stream.cu -o cds_legacy
 *
 * 编译（Per-Thread 模式）：
 *   nvcc -arch=native -std=c++17 --default-stream per-thread \
 *        cuda_default_stream.cu -o cds_perthread
 *
 * 两条命令编译出不同行为的二进制，用于对比。
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ────────────
// 错误检查宏
// ────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",                  \
                    __FILE__, __LINE__,                                       \
                    cudaGetErrorName(err), cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


// ════════════
// 背景：CUDA 中的"默认 stream"是什么
//
// 每次 kernel 启动都必须指定一个 stream。若不指定（或传 0/NULL），
// 使用的就是"默认 stream"（default stream）。
//
// CUDA 提供两种默认 stream 语义，通过编译选项选择：
//
// ① cudaStreamLegacy（传统默认 stream，又称 NULL stream）
//   · 标识符：cudaStreamLegacy，等价于 (cudaStream_t)0 或 NULL
//   · 编译选项：--default-stream legacy（nvcc 默认行为）
//   · 语义：隐式全局同步
//       - 提交到 Legacy stream 的操作，会等待同 context 内所有其他
//         非阻塞 stream 中已入队的操作全部完成后才执行
//       - 同 context 内所有其他非阻塞 stream，会等待 Legacy stream
//         中的操作完成后才继续执行
//       - 效果：Legacy stream 是一个"全局屏障点"，破坏并发性
//
// ② cudaStreamPerThread（per-thread 默认 stream）
//   · 标识符：cudaStreamPerThread
//   · 编译选项：--default-stream per-thread
//             或 -DCUDA_API_PER_THREAD_DEFAULT_STREAM
//   · 语义：每个 host 线程拥有独立的默认 stream
//       - 不同线程的默认 stream 互相独立，不隐式同步
//       - 行为与普通用 cudaStreamCreate 创建的 stream 相同
//       - 多线程场景下真正实现 GPU 并发
// ══════════════


static const int N       = 1 << 20;
static const int THREADS = 256;
static const int BLOCKS  = (N + THREADS - 1) / THREADS;

// 简单的向量加法 kernel，sleep_iters 控制执行时长（模拟耗时操作）
__global__ void addKernel(float *out, const float *in,
                          float val, int n, int sleep_iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 模拟耗时：空循环，防止编译器优化掉
    // volatile 阻止编译器将循环优化为空操作
    volatile int s = 0;
    for (int i = 0; i < sleep_iters; i++) s += i;
    (void)s;

    out[idx] = in[idx] + val;
}


// ═══════════
// 示例 1：cudaStreamLegacy 的隐式同步行为
//
// 场景：
//   streamA（非阻塞 stream）先提交一个耗时 kernel
//   然后在 Legacy stream（默认 stream）提交一个 kernel
//   最后在 streamB（非阻塞 stream）提交一个 kernel
//
// 预期行为（Legacy 语义）：
//   Legacy stream 的 kernel 会等待 streamA 的 kernel 完成
//   streamB 的 kernel 会等待 Legacy stream 的 kernel 完成
//   时间线：[streamA kernel] → [Legacy kernel] → [streamB kernel]
//           完全串行，无并发
//
// 对比（如果三个都用非阻塞 stream）：
//   时间线：[streamA kernel]
//           [Legacy kernel ]  ← 三者并发
//           [streamB kernel]
// ════════════
void demo_legacy_stream(float *d_in, float *d_outA,
                        float *d_outL, float *d_outB)
{
    printf("\n─── 示例 1：cudaStreamLegacy 隐式同步 ───\n");

    // 创建两个普通非阻塞 stream
    // cudaStreamNonBlocking：stream 标志位，表示此 stream 不与 Legacy stream 隐式同步
    // 不加此标志则为阻塞 stream（blocking stream），同样会与 Legacy stream 同步
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking));

    // ① streamA 提交耗时 kernel
    printf("  [1] streamA 提交耗时 kernel\n");
    addKernel<<<BLOCKS, THREADS, 0, streamA>>>(d_outA, d_in, 1.0f, N, 10000);
    CUDA_CHECK(cudaGetLastError());

    // ② Legacy stream 提交 kernel
    // cudaStreamLegacy：Legacy 默认 stream 的显式标识符
    //   等价写法：
    //     addKernel<<<BLOCKS, THREADS>>>(...)          // 不指定 stream
    //     addKernel<<<BLOCKS, THREADS, 0, 0>>>(...)    // 传 0
    //     addKernel<<<BLOCKS, THREADS, 0, NULL>>>(...)  // 传 NULL
    //   以上三种在 --default-stream legacy 模式下均使用 Legacy stream
    //
    // Legacy stream 隐式同步规则（提交时生效）：
    //   · 等待 streamA 中已入队的所有操作完成后，才开始执行本 kernel
    //   · streamB 中后续提交的操作，要等本 kernel 完成后才能执行
    printf("  [2] Legacy stream 提交 kernel（将等待 streamA 完成）\n");
    addKernel<<<BLOCKS, THREADS, 0, cudaStreamLegacy>>>(d_outL, d_in, 2.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    // ③ streamB 提交 kernel（会等待 Legacy stream 完成）
    printf("  [3] streamB 提交 kernel（将等待 Legacy stream 完成）\n");
    addKernel<<<BLOCKS, THREADS, 0, streamB>>>(d_outB, d_in, 3.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    // 等待所有操作完成
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
    CUDA_CHECK(cudaStreamSynchronize(streamB));

    printf("  [PASS] Legacy stream 示例完成\n");
    printf("  执行顺序：streamA kernel → Legacy kernel → streamB kernel（串行）\n");

    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
}


// ═══════════
// 示例 2：cudaStreamPerThread 的独立语义
//
// 场景与示例 1 相同，但将 Legacy stream 替换为 cudaStreamPerThread
//
// 预期行为（Per-Thread 语义）：
//   cudaStreamPerThread 与 streamA、streamB 之间没有隐式同步
//   三个 stream 真正并发执行
//   时间线：[streamA kernel]
//           [PerThread kernel]  ← 三者并发，无隐式等待
//           [streamB kernel]
//
// cudaStreamPerThread 的本质：
//   每个 host 线程有一个独属的默认 stream，
//   行为与 cudaStreamCreate 创建的普通 stream 完全相同，
//   不与任何其他 stream 发生隐式同步。
// ═════════════
void demo_per_thread_stream(float *d_in, float *d_outA,
                             float *d_outP, float *d_outB)
{
    printf("\n─── 示例 2：cudaStreamPerThread 独立语义 ───\n");

    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking));

    // ① streamA 提交耗时 kernel
    printf("  [1] streamA 提交耗时 kernel\n");
    addKernel<<<BLOCKS, THREADS, 0, streamA>>>(d_outA, d_in, 1.0f, N, 10000);
    CUDA_CHECK(cudaGetLastError());

    // ② cudaStreamPerThread 提交 kernel
    // cudaStreamPerThread：当前 host 线程的 per-thread 默认 stream
    //   · 不等待 streamA，与 streamA 并发执行
    //   · streamB 也不等待它，三者真正并发
    //   · 仅在 --default-stream per-thread 编译时，
    //     不指定 stream 的 kernel 启动才使用此 stream；
    //     legacy 模式下不指定 stream 使用的是 cudaStreamLegacy
    //   · cudaStreamPerThread 作为显式标识符，两种编译模式下均可使用
    printf("  [2] cudaStreamPerThread 提交 kernel（不等待 streamA）\n");
    addKernel<<<BLOCKS, THREADS, 0, cudaStreamPerThread>>>(d_outP, d_in, 2.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    // ③ streamB 提交 kernel（不等待 cudaStreamPerThread）
    printf("  [3] streamB 提交 kernel（不等待 per-thread stream）\n");
    addKernel<<<BLOCKS, THREADS, 0, streamB>>>(d_outB, d_in, 3.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(streamB));

    printf("  [PASS] Per-thread stream 示例完成\n");
    printf("  执行顺序：三者并发，无隐式等待\n");

    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
}


// ═════════════
// 示例 3：blocking stream vs non-blocking stream
//
// cudaStreamCreate 创建的是 blocking stream（阻塞 stream）：
//   · 与 Legacy stream 之间有隐式同步（双向）
//   · 等价于 cudaStreamCreateWithFlags(&s, cudaStreamDefault)
//
// cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) 创建非阻塞 stream：
//   · 与 Legacy stream 之间没有隐式同步
//   · 与其他 non-blocking stream 之间也没有隐式同步
//
// 注意：cudaStreamPerThread 的行为类似 non-blocking stream，
//       不与 Legacy stream 发生隐式同步。
// ════════════
void demo_blocking_vs_nonblocking(float *d_in, float *d_outA, float *d_outB)
{
    printf("\n─── 示例 3：blocking stream vs non-blocking stream ───\n");

    // blocking stream：与 Legacy stream 隐式同步
    cudaStream_t blockingStream;
    CUDA_CHECK(cudaStreamCreate(&blockingStream));
    // 等价于：cudaStreamCreateWithFlags(&blockingStream, cudaStreamDefault);

    // non-blocking stream：与 Legacy stream 无隐式同步
    cudaStream_t nonBlockingStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&nonBlockingStream, cudaStreamNonBlocking));

    // Legacy stream 提交 kernel
    printf("  [1] Legacy stream 提交 kernel\n");
    addKernel<<<BLOCKS, THREADS, 0, cudaStreamLegacy>>>(d_outA, d_in, 1.0f, N, 10000);
    CUDA_CHECK(cudaGetLastError());

    // blocking stream 提交 kernel：等待 Legacy stream 完成
    printf("  [2] blocking stream 提交 kernel（等待 Legacy stream）\n");
    addKernel<<<BLOCKS, THREADS, 0, blockingStream>>>(d_outA, d_in, 2.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    // non-blocking stream 提交 kernel：不等待 Legacy stream
    printf("  [3] non-blocking stream 提交 kernel（不等待 Legacy stream）\n");
    addKernel<<<BLOCKS, THREADS, 0, nonBlockingStream>>>(d_outB, d_in, 3.0f, N, 100);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
    CUDA_CHECK(cudaStreamSynchronize(blockingStream));
    CUDA_CHECK(cudaStreamSynchronize(nonBlockingStream));

    printf("  [PASS] blocking vs non-blocking 示例完成\n");
    printf("  blocking stream  与 Legacy stream 串行\n");
    printf("  non-blocking stream 与 Legacy stream 并发\n");

    CUDA_CHECK(cudaStreamDestroy(blockingStream));
    CUDA_CHECK(cudaStreamDestroy(nonBlockingStream));
}


// ═════════════
// 验证结果
// ═════════════
static void verify(const float *d_out, float expected, int n, const char *label)
{
    float *h = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h[i] - expected) > 1e-3f) {
            printf("  [FAIL] %s: h[%d]=%.2f expected=%.2f\n",
                   label, i, h[i], expected);
            ok = false; break;
        }
    }
    if (ok) printf("  [PASS] %s (expected=%.2f)\n", label, expected);
    free(h);
}


int main(void)
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // ── 编译模式说明 ──────────────────────────────────────────────────────
    // CUDA_API_PER_THREAD_DEFAULT_STREAM 宏由 --default-stream per-thread 定义
    // 可用于在运行时打印当前编译模式
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    printf("编译模式：per-thread default stream\n");
    printf("  不指定 stream 的 kernel 使用 cudaStreamPerThread\n");
#else
    printf("编译模式：legacy default stream（默认）\n");
    printf("  不指定 stream 的 kernel 使用 cudaStreamLegacy\n");
#endif

    float *d_in, *d_outA, *d_outB, *d_outC;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outC, N * sizeof(float)));

    // 初始化输入为 0.0f
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    // 示例 1：Legacy stream 隐式同步
    demo_legacy_stream(d_in, d_outA, d_outB, d_outC);
    verify(d_outA, 1.0f, N, "streamA out");
    verify(d_outB, 2.0f, N, "Legacy out");
    verify(d_outC, 3.0f, N, "streamB out");

    // 示例 2：Per-thread stream 独立语义
    demo_per_thread_stream(d_in, d_outA, d_outB, d_outC);
    verify(d_outA, 1.0f, N, "streamA out");
    verify(d_outB, 2.0f, N, "per-thread out");
    verify(d_outC, 3.0f, N, "streamB out");

    // 示例 3：blocking vs non-blocking stream
    demo_blocking_vs_nonblocking(d_in, d_outA, d_outB);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_outA));
    CUDA_CHECK(cudaFree(d_outB));
    CUDA_CHECK(cudaFree(d_outC));

    printf("\n全部示例完成。\n");
    return 0;
}
