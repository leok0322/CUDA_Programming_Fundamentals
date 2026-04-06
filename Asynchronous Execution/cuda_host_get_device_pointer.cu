/**
 * cuda_host_get_device_pointer.cu
 *
 * 演示 cudaHostGetDevicePointer 的函数签名与使用方式。
 *
 * 背景：
 *   cudaHostAllocMapped 分配的 pinned memory 同时存在两个地址：
 *     - host pointer：CPU 用来读写的地址（普通虚拟地址）
 *     - device pointer：GPU 用来读写的地址（GPU 虚拟地址空间）
 *   cudaHostGetDevicePointer 的作用就是：给定 host pointer，查询对应的 device pointer。
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_host_get_device_pointer.cu -o host_get_device_ptr
 *
 * 运行：
 *   ./host_get_device_ptr
 */

#include <cuda_runtime.h>  // cudaHostAlloc、cudaHostGetDevicePointer 等 CUDA 运行时 API
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
// kernel：通过 device pointer 直接读写 host 内存（零拷贝）
// ─────────────────────────────────────────────
__global__ void addOneKernel(float *d_ptr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_ptr[idx] += 1.0f;   // GPU 通过 PCIe 直接访问 host 内存
}

// ─────────────────────────────────────────────
// __device__ 全局变量：物理存储在 GPU 显存（VRAM）
// 供 cudaGetSymbolAddress 示例使用
// ─────────────────────────────────────────────
__device__ float d_symbol_array[1 << 20];   // 1M 个 float，存在显存中

// ═══════════════════════════════════════════════════════════════════════
// 示例 A：cudaHostAllocMapped + cudaHostGetDevicePointer（零拷贝）
//
//   物理内存在 host DRAM，GPU 通过 PCIe 远程访问，无需 cudaMemcpy。
//   程序员维护两个指针：h_ptr（CPU用）和 d_ptr（GPU用）。
// ═══════════════════════════════════════════════════════════════════════
void demo_mapped(int N, size_t size)
{
    printf("\n─── 示例 A：cudaHostAllocMapped + cudaHostGetDevicePointer ───\n");

    // 1. 分配 Mapped pinned memory
    float *h_ptr = NULL;
    CUDA_CHECK(cudaHostAlloc(
        (void **)&h_ptr,
        size,
        cudaHostAllocMapped   // 允许 GPU 直接访问此 host 内存
    ));

    // 2. 查询 GPU 侧映射地址
    //   签名：cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
    //   pDevice：输出，GPU 虚拟地址空间中的映射地址
    //   pHost  ：输入，cudaHostAllocMapped 分配的 host pointer
    //   flags  ：保留字段，必须为 0
    float *d_ptr = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer(
        (void **)&d_ptr,   // [out] GPU 侧地址
        (void  *) h_ptr,   // [in]  host 侧地址
        0                  // [in]  flags = 0
    ));

    printf("  h_ptr (host DRAM 地址)  : %p\n", (void *)h_ptr);
    printf("  d_ptr (GPU 映射地址)    : %p\n", (void *)d_ptr);
    // 地址不同：同一物理页在两个虚拟地址空间中的不同映射

    // 3. CPU 直接写 h_ptr，无需任何拷贝
    for (int i = 0; i < N; i++)
        h_ptr[i] = (float)i;

    // 4. GPU 用 d_ptr 访问同一块内存（每次访问走 PCIe）
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads>>>(d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. CPU 直接读 h_ptr，无需 D2H 拷贝
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_ptr[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    // 6. 只释放 h_ptr；d_ptr 是映射地址，不独立分配，无需释放
    CUDA_CHECK(cudaFreeHost(h_ptr));
}

// ═══════════════════════════════════════════════════════════════════════
// 示例 B：cudaGetSymbolAddress + cudaMemcpy
//
//   __device__ 变量物理存储在 GPU 显存（VRAM），GPU 访问无 PCIe 开销。
//   但 CPU 无法直接读写，必须通过 cudaMemcpy 搬运数据。
//   程序员只有一个符号名（d_symbol_array），通过本函数获取其显存地址。
// ═══════════════════════════════════════════════════════════════════════
void demo_symbol(int N, size_t size)
{
    printf("\n─── 示例 B：cudaGetSymbolAddress + cudaMemcpy ───\n");

    // 1. 查询 __device__ 变量在显存中的地址
    //   签名：cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol)
    //   devPtr ：输出，__device__ 变量在 GPU 显存中的地址
    //   symbol ：输入，__device__ 或 __constant__ 声明的全局变量名
    //   无 flags 参数（内存类型固定为显存，无需额外选项）
    float *sym_ptr = NULL;
    CUDA_CHECK(cudaGetSymbolAddress(
        (void **)&sym_ptr,    // [out] 显存地址写入 sym_ptr
        d_symbol_array        // [in]  __device__ 变量符号
    ));

    printf("  d_symbol_array 显存地址 : %p\n", (void *)sym_ptr);

    // 2. CPU 准备 host 数据
    float *h_buf = (float *)malloc(size);
    for (int i = 0; i < N; i++)
        h_buf[i] = (float)i;

    // 3. H2D：必须通过 cudaMemcpy 把数据搬到显存
    //   （不能直接 h_buf → d_symbol_array，CPU 无法直接写显存）
    CUDA_CHECK(cudaMemcpy(sym_ptr, h_buf, size, cudaMemcpyHostToDevice));

    // 4. GPU kernel 访问显存，延迟低（无 PCIe 开销）
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads>>>(sym_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. D2H：必须通过 cudaMemcpy 把结果搬回 host
    CUDA_CHECK(cudaMemcpy(h_buf, sym_ptr, size, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_buf[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    free(h_buf);
}

// ══════════════════════════
// 示例 C：cudaMallocManaged（Unified Memory）
//
//   一个指针，CPU 和 GPU 都可直接使用。
//   运行时按 Page Fault 自动在 host DRAM 和 GPU 显存之间迁移数据页。
//   无需手动 cudaMemcpy，也无需维护两个指针。
// ═════════════════════════
void demo_managed(int N, size_t size)
{
    printf("\n─── 示例 C：cudaMallocManaged（Unified Memory）───\n");

    // 1. 分配 Unified Memory
    //   签名：cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
    //   devPtr ：输出，统一虚拟地址（CPU 和 GPU 共用同一个地址）
    //   size   ：分配字节数
    //   flags  ：cudaMemAttachGlobal（默认，所有设备可见）
    //            cudaMemAttachHost（仅 host 可见，适合流式传输）
    float *um_ptr = NULL;
    CUDA_CHECK(cudaMallocManaged(
        (void **)&um_ptr,       // [out] 统一虚拟地址
        size,
        cudaMemAttachGlobal     // [in]  所有 GPU 均可访问
    ));

    printf("  um_ptr (统一虚拟地址)   : %p\n", (void *)um_ptr);
    // CPU 和 GPU 使用同一个地址值，运行时负责背后的页迁移

    // 2. CPU 直接写（此时页在 host，触发 host 侧访问）
    for (int i = 0; i < N; i++)
        um_ptr[i] = (float)i;

    // 3. GPU 直接用同一个指针（首次访问触发 Page Fault，运行时迁移页到显存）
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads>>>(um_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynchronize 后页迁移完成，CPU 可再次访问

    // 4. CPU 直接读同一个指针（运行时将页迁移回 host）
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (um_ptr[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    // 5. 只需一次 cudaFree，无需区分 host/device
    CUDA_CHECK(cudaFree(um_ptr));
}



int main(void)
{
    const int    N    = 1 << 20;
    const size_t size = N * sizeof(float);

    // 检查设备是否支持 Mapped 内存（示例 A 需要）
    int canMap = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, 0));
    if (!canMap) {
        fprintf(stderr, "该设备不支持 Mapped host memory，跳过示例 A。\n");
    } else {
        demo_mapped(N, size);   // 示例 A：零拷贝，数据永远在 host DRAM
    }

    demo_symbol(N, size);       // 示例 B：__device__ 变量，数据在 GPU 显存
    demo_managed(N, size);      // 示例 C：Unified Memory，运行时自动迁移

    // ── 三种方式对比 ────────────────────────────────────────────────
    //
    //  方式                      数据位置        指针数量  cudaMemcpy  GPU访问延迟
    //  ─────────────────────── ──────────────  ────────  ──────────  ───────────
    //  A. HostAllocMapped        host DRAM       2个       不需要      高（PCIe）
    //     + HostGetDevicePointer
    //  B. GetSymbolAddress       GPU 显存(VRAM)  1个       需要        低（本地）
    //     + cudaMemcpy
    //  C. cudaMallocManaged      自动迁移        1个       不需要      迁移后低
    //     (Unified Memory)

    return 0;
}
