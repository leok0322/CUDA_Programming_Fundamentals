/**
 * cuda_free_comparison.cu
 *
 * 对比三种释放函数的使用场景：
 *   - free()         释放 malloc 分配的 pageable host 内存
 *   - cudaFreeHost() 释放 cudaMallocHost / cudaHostAlloc 分配的 pinned host 内存
 *   - cudaFree()     释放 cudaMalloc / cudaMallocManaged 分配的 device / unified 内存
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_free_comparison.cu -o free_comparison
 *
 * 运行：
 *   ./free_comparison
 */

#include <cuda_runtime.h>  // cudaMalloc、cudaFree、cudaMallocHost、cudaFreeHost 等
#include <stdio.h>         // printf、fprintf
#include <stdlib.h>        // malloc、free、exit、EXIT_FAILURE

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


static const int   N    = 1 << 20;
static const size_t SIZE = N * sizeof(float);

// ────────────────────────
// 示例 1：malloc + free
//
//   malloc 分配 pageable 内存：
//     - OS 以"虚拟内存"方式管理，物理页可能被换出到磁盘（swap）
//     - CUDA DMA 无法直接访问，cudaMemcpy 时内部需额外拷贝到 staging buffer
//     - free() 通知 C 运行时堆管理器回收虚拟地址范围，物理页由 OS 按需回收
// ─────────────────────────────────────────────────────────────────────
void demo_malloc_free(void)
{
    printf("\n─── 示例 1：malloc + free（pageable host 内存）───\n");

    // malloc：在 host 堆上分配 pageable 虚拟内存
    float *h_pageable = (float *)malloc(SIZE);
    if (!h_pageable) { fprintf(stderr, "malloc 失败\n"); return; }
    printf("  malloc    ptr : %p\n", (void *)h_pageable);

    for (int i = 0; i < N; i++)
        h_pageable[i] = (float)i;

    // free：释放 pageable 内存
    //   签名：void free(void *ptr)
    //   - 无返回值，无法感知错误
    //   - ptr 必须是 malloc/calloc/realloc 返回的原始指针
    //   - free(NULL) 是安全的，什么都不做
    free(h_pageable);
    h_pageable = NULL;   // 置 NULL 防止悬空指针
    printf("  free() 完成，pageable 内存已归还堆\n");
}

// ─────────────────────────────────────────────────────────────────────
// 示例 2：cudaMallocHost + cudaFreeHost
//
//   cudaMallocHost 分配 pinned（page-locked）内存：
//     - OS 将物理页锁定，不可换出
//     - DMA 可直接访问，cudaMemcpy 无需 staging buffer，带宽更高
//     - cudaFreeHost() 先解锁物理页，再归还给 OS；
//       不能用 free()，因为 C 堆管理器不知道这块内存，会破坏 CUDA 内部记录
// ─────────────────────────────────────────────────────────────────────
void demo_cudaMallocHost_cudaFreeHost(void)
{
    printf("\n─── 示例 2：cudaMallocHost + cudaFreeHost（pinned host 内存）───\n");

    // cudaMallocHost：分配 pinned 内存
    float *h_pinned = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_pinned, SIZE));
    printf("  cudaMallocHost ptr : %p\n", (void *)h_pinned);

    for (int i = 0; i < N; i++)
        h_pinned[i] = (float)i;

    // cudaFreeHost：释放 pinned 内存
    //   签名：cudaError_t cudaFreeHost(void *ptr)
    //   - 有返回值，可检测错误（如传入非 pinned 指针）
    //   - ptr 必须是 cudaMallocHost 或 cudaHostAlloc 返回的原始指针
    //   - 内部两步：① 解除物理页锁定（unpin） ② 归还内存
    //   - cudaFreeHost(NULL) 安全，返回 cudaSuccess
    CUDA_CHECK(cudaFreeHost(h_pinned));
    h_pinned = NULL;
    printf("  cudaFreeHost() 完成，物理页已解锁并归还 OS\n");
}

// ─────────────────────────────────────────────────────────────────────
// 示例 3：cudaHostAlloc(Mapped) + cudaFreeHost
//
//   cudaHostAlloc 无论传何种 flags，分配的都是 pinned 内存，
//   因此释放时同样用 cudaFreeHost，而不是 free()。
//   d_ptr 是 h_ptr 的映射地址，不是独立分配，不需要单独释放。
// ─────────────────────────────────────────────────────────────────────
void demo_cudaHostAlloc_cudaFreeHost(void)
{
    printf("\n─── 示例 3：cudaHostAlloc(Mapped) + cudaFreeHost───\n");

    float *h_mapped = NULL;
    CUDA_CHECK(cudaHostAlloc((void **)&h_mapped, SIZE, cudaHostAllocMapped));

    float *d_mapped = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_mapped, h_mapped, 0));

    printf("  h_mapped (host ptr)   : %p\n", (void *)h_mapped);
    printf("  d_mapped (device ptr) : %p\n", (void *)d_mapped);

    // 只释放 h_mapped（host pointer 原始分配）
    // d_mapped 是映射地址，释放 h_mapped 时映射自动解除，不需要单独释放
    CUDA_CHECK(cudaFreeHost(h_mapped));
    h_mapped = NULL;
    // d_mapped 此时已失效，不需要也不能再调用任何释放函数
    printf("  cudaFreeHost(h_mapped) 完成，d_mapped 映射自动解除\n");
}

// ─────────────────────────────────────────────────────────────────────
// 示例 4：cudaMalloc + cudaFree
//
//   cudaMalloc 在 GPU 显存（VRAM）中分配内存：
//     - 地址在 GPU 虚拟地址空间，CPU 不能直接读写
//     - cudaFree() 归还显存给 CUDA 内存管理器
//     - 不能用 free()（host 堆管理器不认识该地址）
// ─────────────────────────────────────────────────────────────────────
void demo_cudaMalloc_cudaFree(void)
{
    printf("\n─── 示例 4：cudaMalloc + cudaFree（GPU 显存）───\n");

    // cudaMalloc：在 GPU 显存中分配内存
    float *d_dev = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_dev, SIZE));
    printf("  cudaMalloc ptr : %p\n", (void *)d_dev);

    // cudaFree：释放 GPU 显存
    //   签名：cudaError_t cudaFree(void *devPtr)
    //   - 有返回值，可检测错误
    //   - devPtr 必须是 cudaMalloc 返回的原始指针
    //   - cudaFree(NULL) 安全，返回 cudaSuccess
    CUDA_CHECK(cudaFree(d_dev));
    d_dev = NULL;
    printf("  cudaFree() 完成，显存已归还 CUDA 内存管理器\n");
}

// ─────────────────────────────────────────────────────────────────────
// 示例 5：cudaMallocManaged + cudaFree
//
//   cudaMallocManaged 分配 Unified Memory：
//     - CPU 和 GPU 共用同一个虚拟地址，运行时自动迁移页
//     - 释放时用 cudaFree，不用 free() 也不用 cudaFreeHost()
//     - cudaFree 会同时清理 host 和 device 侧的映射
// ─────────────────────────────────────────────────────────────────────
void demo_cudaMallocManaged_cudaFree(void)
{
    printf("\n─── 示例 5：cudaMallocManaged + cudaFree（Unified Memory）───\n");

    float *um_ptr = NULL;
    CUDA_CHECK(cudaMallocManaged((void **)&um_ptr, SIZE, cudaMemAttachGlobal));
    printf("  cudaMallocManaged ptr : %p\n", (void *)um_ptr);

    // CPU 和 GPU 都用同一个 um_ptr，无需区分
    for (int i = 0; i < N; i++)
        um_ptr[i] = (float)i;

    // cudaFree：同时释放 host 和 device 侧映射
    //   不能用 free()：um_ptr 不在 C 堆
    //   不能用 cudaFreeHost()：um_ptr 不是 pinned host 内存
    CUDA_CHECK(cudaFree(um_ptr));
    um_ptr = NULL;
    printf("  cudaFree() 完成，host + device 侧映射均已释放\n");
}

int main(void)
{
    demo_malloc_free();
    demo_cudaMallocHost_cudaFreeHost();
    demo_cudaHostAlloc_cudaFreeHost();
    demo_cudaMalloc_cudaFree();
    demo_cudaMallocManaged_cudaFree();

    // ── 配套关系速查 ────────────────────────────────────────────────
    //
    //   分配函数                          释放函数         内存位置
    //   ─────────────────────────────    ─────────────    ──────────────────
    //   malloc / calloc / realloc         free()           host pageable
    //   cudaMallocHost                    cudaFreeHost()   host pinned
    //   cudaHostAlloc(任意 flags)         cudaFreeHost()   host pinned
    //   cudaHostGetDevicePointer 的结果   不需要释放       映射地址，非独立分配
    //   cudaMalloc                        cudaFree()       GPU 显存
    //   cudaMallocManaged                 cudaFree()       Unified Memory

    printf("\n全部示例完成。\n");
    return 0;
}
