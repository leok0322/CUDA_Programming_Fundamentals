/**
 * pinned_host_kernel.cu
 *
 * 验证：cudaMallocHost 分配的 pinned 内存，其指针可以直接传入 kernel，
 * 无需 cudaMemcpy。
 *
 * 核心机制：
 *   cudaMallocHost 分配的是页锁定（page-locked）host 内存。
 *   在 UVA（Unified Virtual Addressing）体系下，该内存同时在
 *   CPU 页表和 GPU 页表中建立映射，GPU 通过 PCIe 零拷贝直接访问。
 *
 * 编译：
 *   nvcc -arch=native -std=c++17 pinned_host_kernel.cu -o pinned_host_kernel
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",            \
                    __FILE__, __LINE__,                                 \
                    cudaGetErrorName(err), cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ────────────────────
// Kernel：直接读写 pinned host 指针
//
// GPU 访问 pinned host 内存的底层路径：
//   1. GPU 的 GMMU 查页表 → 找到该 UVA 地址对应的 CPU DRAM 物理页
//   2. 通过 PCIe（或 NVLink）直接读写 CPU DRAM
//   3. 无需任何数据迁移，无需 cudaMemcpy
//
// 代价：每次访问都经过 PCIe，带宽约 16~32 GB/s，
//       远低于 GPU 本地显存（~900 GB/s）。
//       适合访问次数少、数据量小、或不值得拷贝的场景。
// ────────────────────
__global__ void writeKernel(float* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = (float)i * 2.0f;   // GPU 直接写入 CPU DRAM（通过 PCIe）
}

__global__ void copyKernel(const float* src, float* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[i] + 1.0f;   // src、dst 均可以是 pinned host 指针
}

// ──────────────────────
// 辅助函数
// ──────────────────────
static constexpr int vLen = 1 << 20;   // 1M 个 float，供两个演示函数共用

static void initVector(float* v, int n)
{
    for (int i = 0; i < n; ++i) v[i] = (float)i;
}

// 验证 a[i] == b[i] + 1（copyKernel 语义：dst = src + 1）
static void checkAnswer(const float* a, const float* b, int n)
{
    for (int i = 0; i < n; ++i) {
        if (fabsf(a[i] - (b[i] + 1.0f)) > 1e-5f) {
            printf("✗  不一致：a[%d]=%.1f，期望 %.1f\n", i, a[i], b[i] + 1.0f);
            return;
        }
    }
    printf("✓\n");
}

// ──────────────────────
// cudaHostAlloc + cudaHostAllocMapped 演示
//
// cudaHostAlloc 函数签名：
//   cudaError_t cudaHostAlloc(
//       void**       pHost,  // [out] 返回 host 侧指针
//       size_t       size,   // [in]  字节数
//       unsigned int flags   // [in]  行为标志（可 OR 组合）
//   );
//
// 与 cudaMallocHost 的关系：
//   cudaMallocHost(ptr, size)
//     ≡ cudaHostAlloc(ptr, size, cudaHostAllocDefault)
//   是 cudaHostAlloc 的简化包装，flags 固定为 0（默认行为）。
//   两者分配的内存均为 pinned，均须用 cudaFreeHost 释放。
//
// cudaHostAlloc 的 flags：
//   cudaHostAllocDefault    = 0x00  页锁定，可加速 cudaMemcpy（DMA 直传），
//                                   与 cudaMallocHost 等价。
//                                   UVA 下 GPU 也可直接访问（同 Mapped）。
//   cudaHostAllocMapped     = 0x02  显式声明：将此内存映射到 GPU 地址空间，
//                                   cudaHostGetDevicePointer 可查询 GPU 侧指针。
//                                   UVA 下与 Default 实际效果相同（地址已统一）；
//                                   非 UVA 旧硬件上才有实质区别。
//   cudaHostAllocWriteCombined = 0x04  写合并内存：CPU 写入不经 L1/L2 cache，
//                                   直接合并后批量刷到内存总线，
//                                   CPU 读取极慢，适合"CPU 只写、GPU 只读"场景。
//   cudaHostAllocPortable   = 0x01  分配的内存对所有 CUDA Context 可见
//                                   （默认只对分配时的 Context 有效）。
// ──────────────────────
static void usingCudaHostAlloc()
{
    printf("\n════════════════\n");
    printf("  cudaHostAlloc + cudaHostAllocMapped\n");
    printf("════════════════\n");

    float* a = nullptr;
    float* b = nullptr;

    // cudaHostAllocMapped：显式请求映射到 GPU 地址空间。
    // UVA 环境下与 cudaMallocHost 行为相同：
    //   · host 指针即 device 指针（同一虚拟地址）
    //   · 无需 cudaHostGetDevicePointer，直接把 a/b 传入 kernel
    CUDA_CHECK(cudaHostAlloc(&a, vLen * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&b, vLen * sizeof(float), cudaHostAllocDefault));

    initVector(b, vLen);           // CPU 初始化 b：b[i] = i
    memset(a, 0, vLen * sizeof(float));   // a 清零

    // 查询指针属性，对比 cudaMallocHost 结果
    cudaPointerAttributes attrA{}, attrB{};
    CUDA_CHECK(cudaPointerGetAttributes(&attrA, a));
    CUDA_CHECK(cudaPointerGetAttributes(&attrB, b));
    printf("  a: hostPointer=%p  devicePointer=%p  same=%s\n",
           attrA.hostPointer, attrA.devicePointer,
           attrA.hostPointer == attrA.devicePointer ? "✓" : "✗");
    printf("  b: hostPointer=%p  devicePointer=%p  same=%s\n",
           attrB.hostPointer, attrB.devicePointer,
           attrB.hostPointer == attrB.devicePointer ? "✓" : "✗");

    // kernel 直接使用 host 指针，无需任何转换
    int threads = 256;
    int blocks  = vLen / threads;
    copyKernel<<<blocks, threads>>>(b, a, vLen);   // dst=a, src=b
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Using cudaHostAlloc: ");
    checkAnswer(a, b, vLen);   // 验证 a[i] == b[i] + 1

    // cudaHostAlloc 分配的内存同样用 cudaFreeHost 释放
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
}

int main()
{
    cudaInitDevice(0, 0, 0);   // 显式初始化，消除第一次 API 调用的 Context 创建开销

    constexpr size_t sz = vLen * sizeof(float);   // vLen 来自全局常量（1M 个 float = 4 MB）

    // ── 分配 pinned host 内存 ────────────
    //
    // cudaMallocHost 函数签名：
    //   cudaError_t cudaMallocHost(void** ptr, size_t size);
    //
    // 与普通 malloc 的区别：
    //   malloc          → 可分页（pageable）内存，OS 可随时换出到磁盘
    //                     GPU 无法直接访问（没有稳定的物理地址）
    //   cudaMallocHost  → 页锁定（pinned）内存，OS 保证物理页不被换出
    //                     驱动在 GPU 页表中建立映射 → GPU 可直接访问
    //
    // UVA 下，cudaMallocHost 返回的指针：
    //   · CPU 可直接读写（普通指针操作）
    //   · GPU 可直接读写（kernel 内直接用）
    //   · cudaMemcpy 可用（也支持显式拷贝路径）
    //   · 同一个指针值，CPU 和 GPU 用的是同一个虚拟地址
    float* a = nullptr;
    float* b = nullptr;
    CUDA_CHECK(cudaMallocHost(&a, sz));   // pinned，GPU 可直接访问
    CUDA_CHECK(cudaMallocHost(&b, sz));

    // ── CPU 初始化 a ───
    for (int i = 0; i < vLen; ++i)
        a[i] = 0.0f;

    // ── 验证 1：GPU 直接写入 pinned host 指针 a ──────────────────
    //
    // 关键：a 是 cudaMallocHost 分配的 pinned 指针，
    //       直接传入 kernel，GPU 通过 PCIe 写入 CPU DRAM。
    //       无需任何 cudaMemcpy。
    int threads = 256;
    int blocks  = (vLen + threads - 1) / threads;
    writeKernel<<<blocks, threads>>>(a, vLen);
    // kernel 启动本身是异步的，<<<>>> 语法不返回错误码。
    // cudaGetLastError() 检查启动配置错误（如 blocks/threads 非法、
    // 共享内存超限等），这类错误在启动时同步设置到 sticky error 槽，
    // 不需要等 GPU 执行完就能检测到。
    // 注意：kernel 执行期间的运行时错误（越界访问、非法指令等）
    // 需要 cudaDeviceSynchronize() 之后才能通过 GetLastError 检测到。
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU 直接读取 a（无需 memcpy，GPU 已写入到 CPU DRAM）
    bool ok1 = true;
    for (int i = 0; i < vLen && ok1; ++i)
        if (fabsf(a[i] - (float)i * 2.0f) > 1e-5f) ok1 = false;
    printf("[验证 1] GPU writeKernel 直接写 pinned a：%s\n", ok1 ? "✓" : "✗");
    printf("         a[0]=%.1f  a[1]=%.1f  a[2]=%.1f\n", a[0], a[1], a[2]);

    // ── 验证 2：同一指针 a 作为 src，b 作为 dst，GPU kernel 直接操作 ──
    //
    // src 和 dst 都是 pinned host 指针，kernel 内直接用，
    // GPU 通过 PCIe 读 a（CPU DRAM），写 b（CPU DRAM）。
    copyKernel<<<blocks, threads>>>(a, b, vLen);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool ok2 = true;
    for (int i = 0; i < vLen && ok2; ++i)
        if (fabsf(b[i] - ((float)i * 2.0f + 1.0f)) > 1e-5f) ok2 = false;
    printf("[验证 2] GPU copyKernel(pinned a → pinned b)：%s\n", ok2 ? "✓" : "✗");
    printf("         b[0]=%.1f  b[1]=%.1f  b[2]=%.1f\n", b[0], b[1], b[2]);

    // ── 查询指针属性，确认 UVA 映射 ──────
    //
    // cudaPointerAttributes 可以查询一个指针的类型和映射信息：
    //   type = cudaMemoryTypeHost  → host 端内存（pinned 或 managed）
    //   devicePointer              → GPU 侧对应的 UVA 地址
    //                                如果等于 hostPointer，说明 UVA 统一了两端地址
    cudaPointerAttributes attr{};
    CUDA_CHECK(cudaPointerGetAttributes(&attr, a));
    printf("\n[指针属性] a（cudaMallocHost）：\n");
    printf("  type          = %d  （%s）\n", attr.type,
           attr.type == cudaMemoryTypeHost   ? "cudaMemoryTypeHost（pinned host）" :
           attr.type == cudaMemoryTypeDevice ? "cudaMemoryTypeDevice" :
           attr.type == cudaMemoryTypeManaged? "cudaMemoryTypeManaged" : "unknown");
    printf("  hostPointer   = %p\n", attr.hostPointer);
    printf("  devicePointer = %p\n", attr.devicePointer);
    printf("  hostPointer == devicePointer：%s（UVA 统一地址）\n",
           attr.hostPointer == attr.devicePointer ? "✓ 是" : "✗ 否");

    // ── 释放 ───────────
    //
    // cudaMallocHost 分配的内存必须用 cudaFreeHost 释放（不能用 free()）：
    //   free(a)         → 只释放 CPU 侧 malloc 结构，GPU 页表映射泄漏
    //   cudaFreeHost(a) → 同时解除 GPU 页表映射，解锁物理页，归还 OS
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));

    printf("\n结论：cudaMallocHost 的 pinned 指针可直接传入 kernel，GPU 通过 PCIe 零拷贝访问。\n");

    usingCudaHostAlloc();
    return 0;
}
