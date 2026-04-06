/**
 * cuda_memcpy_async.cu
 *
 * 详细解析 cudaMemcpyAsync 的函数签名，并给出常见用法示例：
 *   1. 基本 H2D / D2H 异步传输
 *   2. 与 kernel 在同一 stream 内串行
 *   3. 多 stream 并行传输（overlap H2D + kernel）
 *   4. cudaMemcpy（同步版）与 cudaMemcpyAsync 的对比
 *   5. cudaMemcpyAsync 的方向自动推断（cudaMemcpyDefault）
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_memcpy_async.cu -o memcpy_async
 *
 * 运行：
 *   ./memcpy_async
 */

#include <cuda_runtime.h>  // cudaMemcpyAsync、cudaMemcpy、cudaStream_t 等
#include <stdio.h>         // printf、fprintf
#include <stdlib.h>        // exit、EXIT_FAILURE

// ─────────────────────────────────────────────
// 错误检查宏
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
// kernel：向量加一
// ─────────────────────────────────────────────
__global__ void addOneKernel(float *ptr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] += 1.0f;
}

// .cu 文件由 NVCC 按 C++ 规则编译，C++ 规定：
//   const 全局变量默认就是 internal linkage（除非显式加 extern）
//   因此这里的 static 是多余的，去掉后行为完全相同：
//     const int N = 1 << 22;         // C++ 中已是 internal linkage
//     static const int N = 1 << 22;  // 等价，static 无额外效果
//
// 对比 C 的规则（.c 文件）：
//   C 中 const 全局变量默认是 external linkage，必须加 static 才能限制可见性：
//     const int N = 1 << 22;         // C 中是 external linkage，多文件会冲突
//     static const int N = 1 << 22;  // C 中必须加 static 才是 internal linkage
//
// CUDA 代码中 static const 常见的原因：
//   历史习惯——很多 CUDA 开发者有 C 背景，在 C 中必须加 static，
//   迁移到 .cu 后沿用了这个写法，虽然多余但无害。
static const int    N    = 1 << 22;          // 1<<22 = 2^22 = 4M 个 float ≈ 16 MB
static const size_t SIZE = N * sizeof(float);




// ──────────────────────────
// 示例 1：基本 H2D + D2H 异步传输
//
//   同一 stream 内：H2D → kernel → D2H 串行执行。
//   CPU 提交所有命令后立刻继续，stream 在后台按序执行。
// ────────────────────────
void demo_basic_async(void)
{
    printf("\n─── 示例 1：基本 H2D + D2H 异步传输 ───\n");

    // pinned memory：cudaMemcpyAsync 异步的前提
    float *h_in  = NULL;
    float *h_out = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_in,  SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_out, SIZE));

    float *d_buf = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // H2D：把 h_in 传到 GPU
    // CPU 提交后立刻返回，DMA 在 stream 后台开始传输
    CUDA_CHECK(cudaMemcpyAsync(
        d_buf,                    // dst：GPU 显存
        h_in,                     // src：pinned host 内存
        SIZE,                     // count：字节数
        cudaMemcpyHostToDevice,   // kind：H2D
        stream                    // 提交到此 stream
    ));

    // kernel：在同一 stream 内，自动等 H2D 完成后才执行
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads, 0, stream>>>(d_buf, N);
    CUDA_CHECK(cudaGetLastError());

    // D2H：在同一 stream 内，自动等 kernel 完成后才执行
    CUDA_CHECK(cudaMemcpyAsync(
        h_out,                    // dst：pinned host 内存
        d_buf,                    // src：GPU 显存
        SIZE,                     // count：字节数
        cudaMemcpyDeviceToHost,   // kind：D2H
        stream                    // 同一 stream，保证顺序
    ));

    // CPU 到这里已经提交完所有命令，stream 还在后台跑
    // 必须同步，才能安全读 h_out
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_out[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ────────────────────────
// 示例 2：多 stream 并行（H2D 与 kernel 重叠）
//
//   把数据分成两半，stream0 传输第一半时，stream1 可以同时计算第一半。
//   注意：要真正重叠需要 GPU 有足够资源，且两个 stream 分别用不同 buffer。
// ───────────────────────
void demo_multi_stream(void)
{
    printf("\n─── 示例 2：多 stream 并行（H2D + kernel 重叠）───\n");

    const int    HALF      = N / 2;
    const size_t HALF_SIZE = HALF * sizeof(float);

    float *h_in0, *h_in1, *h_out0, *h_out1;
    CUDA_CHECK(cudaMallocHost((void **)&h_in0,  HALF_SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_in1,  HALF_SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_out0, HALF_SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_out1, HALF_SIZE));

    float *d_buf0, *d_buf1;
    CUDA_CHECK(cudaMalloc((void **)&d_buf0, HALF_SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_buf1, HALF_SIZE));

    cudaStream_t s0, s1;
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaStreamCreate(&s1));

    for (int i = 0; i < HALF; i++) { h_in0[i] = 1.0f; h_in1[i] = 2.0f; }

    int threads = 256;
    int blocks  = (HALF + threads - 1) / threads;

    // s0 和 s1 的命令交替提交，让调度器尽量重叠执行：
    //   s0: H2D[0]           → kernel[0]           → D2H[0]
    //   s1:       H2D[1]           → kernel[1]           → D2H[1]
    // 两个 stream 在 GPU 上可以并行执行（取决于硬件资源）

    CUDA_CHECK(cudaMemcpyAsync(d_buf0, h_in0, HALF_SIZE,
                               cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaMemcpyAsync(d_buf1, h_in1, HALF_SIZE,
                               cudaMemcpyHostToDevice, s1));

    addOneKernel<<<blocks, threads, 0, s0>>>(d_buf0, HALF);
    addOneKernel<<<blocks, threads, 0, s1>>>(d_buf1, HALF);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_out0, d_buf0, HALF_SIZE,
                               cudaMemcpyDeviceToHost, s0));
    CUDA_CHECK(cudaMemcpyAsync(h_out1, d_buf1, HALF_SIZE,
                               cudaMemcpyDeviceToHost, s1));

    // 等两个 stream 都完成
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    int errors = 0;
    for (int i = 0; i < HALF; i++) {
        if (h_out0[i] != 2.0f) errors++;
        if (h_out1[i] != 3.0f) errors++;
    }
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    CUDA_CHECK(cudaFreeHost(h_in0));  CUDA_CHECK(cudaFreeHost(h_in1));
    CUDA_CHECK(cudaFreeHost(h_out0)); CUDA_CHECK(cudaFreeHost(h_out1));
    CUDA_CHECK(cudaFree(d_buf0));     CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaStreamDestroy(s0)); CUDA_CHECK(cudaStreamDestroy(s1));
}

// ──────────────────────────
// 示例 3：cudaMemcpy（同步）vs cudaMemcpyAsync（异步）对比
//
//   cudaMemcpy：CPU 阻塞直到传输完成
//   cudaMemcpyAsync：CPU 提交后立刻返回，传输在后台进行
// ──────────────────────────
void demo_sync_vs_async(void)
{
    printf("\n─── 示例 3：cudaMemcpy vs cudaMemcpyAsync ───\n");

    float *h_buf = NULL;
    float *d_buf = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_buf, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));
    for (int i = 0; i < N; i++) h_buf[i] = (float)i;

    // ── 同步版：cudaMemcpy ────────
    // 签名：cudaError_t cudaMemcpy(void *dst, const void *src,
    //                              size_t count, enum cudaMemcpyKind kind)
    // 没有 stream 参数，内部等价于：
    //   cudaMemcpyAsync(dst, src, count, kind, 0) + cudaDeviceSynchronize()
    // CPU 调用后阻塞，直到 DMA 传输完成才返回。
    // h_buf 可以是 pageable（CUDA 内部自动使用 staging buffer）。

    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, SIZE, cudaMemcpyHostToDevice));
    // 到这里传输已完成，d_buf 可以立刻使用
    printf("  cudaMemcpy 返回后传输已完成，CPU 一直等待\n");

    // ── 异步版：cudaMemcpyAsync ─────
    // h_buf 必须是 pinned，否则 CUDA 内部 staging 使行为退化为同步。
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_buf, h_buf, SIZE,
                               cudaMemcpyHostToDevice, stream));
    // 到这里传输可能尚未完成，CPU 继续执行其他工作
    printf("  cudaMemcpyAsync 返回后传输可能仍在进行，CPU 继续\n");

    // 需要显式同步后才能使用 d_buf
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  cudaStreamSynchronize 后传输完成，d_buf 可安全使用\n");

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ────────────────
// 示例 4：cudaMemcpyDefault（自动推断方向）
//
//   CUDA 6.0+ 引入统一虚拟地址（UVA），所有 host pinned 内存和 GPU 显存
//   都在同一个虚拟地址空间中，CUDA 运行时可以通过地址本身判断内存位置，
//   无需程序员显式指定 kind。
//   适合：Unified Memory 场景、不确定指针来源的通用传输函数。
// ──────────────────
void demo_memcpy_default(void)
{
    printf("\n─── 示例 4：cudaMemcpyDefault（自动推断方向）───\n");

    float *h_buf = NULL;
    float *d_buf = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_buf, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));
    for (int i = 0; i < N; i++) h_buf[i] = (float)i;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 不指定方向，CUDA 根据 src/dst 地址自动判断是 H2D 还是 D2H
    // 要求：编译时需开启 UVA（默认开启，sm_20 及以上）
    CUDA_CHECK(cudaMemcpyAsync(
        d_buf,
        h_buf,
        SIZE,
        cudaMemcpyDefault,   // 自动推断：h_buf 是 host，d_buf 是 device → H2D
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  cudaMemcpyDefault H2D 完成\n");

    // 反向：D2H，同样用 Default，CUDA 自动判断
    CUDA_CHECK(cudaMemcpyAsync(
        h_buf,
        d_buf,
        SIZE,
        cudaMemcpyDefault,   // 自动推断：d_buf 是 device，h_buf 是 host → D2H
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_buf[i] != (float)i) errors++;
    printf("  cudaMemcpyDefault D2H 完成，验证：%s\n",
           errors == 0 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ──────────────
// 示例 5：D2D（显存到显存）传输
//
//   D2D 不经过 host，走 GPU 内部总线（带宽远高于 PCIe）。
//   cudaMemcpyAsync 的 src 和 dst 都是 cudaMalloc 分配的显存地址。
// ───────────────
void demo_d2d(void)
{
    printf("\n─── 示例 5：D2D 显存到显存传输 ───\n");

    float *d_src = NULL;
    float *d_dst = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_src, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, SIZE));

    // 用 H2D 初始化 d_src
    float *h_buf = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_buf, SIZE));
    for (int i = 0; i < N; i++) h_buf[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_src, h_buf, SIZE, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // D2D：不经过 host，走 GPU 内部互联总线
    // 典型带宽：A100 HBM2 ~2 TB/s，远高于 PCIe 的 ~32 GB/s
    CUDA_CHECK(cudaMemcpyAsync(
        d_dst,
        d_src,
        SIZE,
        cudaMemcpyDeviceToDevice,  // D2D
        stream
    ));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 验证：把 d_dst 搬回 host 检查
    CUDA_CHECK(cudaMemcpy(h_buf, d_dst, SIZE, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_buf[i] != (float)i) errors++;
    printf("  D2D 验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(void)
{
    demo_basic_async();
    demo_multi_stream();
    demo_sync_vs_async();
    demo_memcpy_default();
    demo_d2d();

    // ── 速查：kind 枚举值 ─────
    //
    //   kind                        src        dst        走哪条路
    //   ─────────────────────────   ────────   ────────   ──────────────────
    //   cudaMemcpyHostToDevice      pinned     显存        PCIe（DMA）
    //   cudaMemcpyDeviceToHost      显存       pinned      PCIe（DMA）
    //   cudaMemcpyDeviceToDevice    显存       显存        GPU 内部总线
    //   cudaMemcpyHostToHost        host       host        CPU memcpy
    //   cudaMemcpyDefault           任意       任意        运行时自动推断
    //
    // ── 常见错误 ─────────
    //
    //   1. H2D 传 pageable 指针：async 退化为 sync，性能下降
    //   2. 提交 async 后不同步直接读 host buffer：读到未完成的数据
    //   3. kind 与实际指针类型不匹配：返回 cudaErrorInvalidMemcpyDirection
    //   4. count 单位是字节，误传元素个数：实际只传了 1/sizeof(T) 的数据

    printf("\n全部示例完成。\n");
    return 0;
}
