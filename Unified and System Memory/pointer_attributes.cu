/**
 * pointer_attributes.cu
 *
 * 使用 cudaPointerGetAttributes 查询指针的内存类型与归属 GPU
 *
 * 演示四类内存指针的属性：
 *   1. pageable host memory  （普通 new / malloc 分配）
 *   2. pinned host memory    （cudaMallocHost 分配）
 *   3. device memory         （cudaMalloc 分配）
 *   4. managed memory        （cudaMallocManaged 分配）
 *
 * 编译：
 *   nvcc -arch=native -std=c++17 pointer_attributes.cu -o pointer_attributes
 *
 * 运行：
 *   ./pointer_attributes
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ────────────────────
// 错误检查宏
// ────────────────────
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


// ══════════════════════════
// 背景：cudaPointerAttributes 结构体
//
// cudaPointerAttributes 是普通结构体，字段完全公开，可直接访问成员：
//   struct cudaPointerAttributes {
//       enum cudaMemoryType type;    // 内存类型
//       int                 device;  // 关联 GPU 编号
//       void               *devicePointer;
//       void               *hostPointer;
//   };
//
// 与不透明指针的区别：
//   不透明指针（如 cudaStream_t、cudaEvent_t）— 内部结构对用户隐藏，
//     只能作为句柄传给 CUDA API，无法直接访问成员。
//   cudaPointerAttributes — 普通结构体，调用 cudaPointerGetAttributes 后
//     可直接读取 attr.type、attr.device 等字段，这正是其设计目的。

// 函数签名：
//   cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
//                                        const void            *ptr)
//   参数：
//     attributes — 输出参数，函数将结果写入此结构体
//     ptr        — 待查询的指针，可以是任意类型的内存地址
//   返回值：
//     cudaSuccess            — 查询成功
//     cudaErrorInvalidValue  — ptr 为无效地址
//
// cudaPointerGetAttributes(&attr, ptr) 用于查询任意指针的元信息。
//
// attr.type 的取值：
//   cudaMemoryTypeUnregistered — 普通 pageable 主机内存（new / malloc）
//                                 GPU 不能直接访问；DMA 传输时驱动临时锁页
//   cudaMemoryTypeHost         — 固定内存（pinned / page-locked）
//                                 CPU 和 GPU 均可直接访问
//                                 attr.hostPointer 和 attr.devicePointer 均有效
//   cudaMemoryTypeDevice       — 设备内存（cudaMalloc）
//                                 仅 GPU 可直接访问
//                                 attr.hostPointer == nullptr
//   cudaMemoryTypeManaged      — 统一内存（cudaMallocManaged）
//                                 CPU 和 GPU 共享同一虚拟地址
//                                 attr.hostPointer == attr.devicePointer
//
// attr.device：
//   device / managed 内存对应的 GPU 编号（0, 1, ...）
//   host 内存时为 -1（或未定义，不应使用）
//
// attr.devicePointer：
//   该内存在 GPU 视角下的访问地址
//   pageable host 内存时为 nullptr（GPU 无法直接访问）
//
// attr.hostPointer：
//   该内存在 CPU 视角下的访问地址
//   纯 device 内存时为 nullptr
// ═════════════════════════


static const int N = 1 << 20;   // 1M 个 float，4 MB

// ────────────────────
// 打印单个指针的 cudaPointerAttributes
// ─────────────────────
static void printPointerAttributes(const char *label, const void *ptr)
{
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));

    const char *typeStr = "unknown";
    switch (attr.type) {
        case cudaMemoryTypeUnregistered: typeStr = "unregistered host (pageable)"; break;
        case cudaMemoryTypeHost:         typeStr = "pinned host (registered)";     break;
        case cudaMemoryTypeDevice:       typeStr = "device";                       break;
        case cudaMemoryTypeManaged:      typeStr = "managed (unified)";            break;
    }

    printf("[%s]\n", label);
    printf("  attr.type          = %s\n",  typeStr);
    printf("  attr.device        = %d\n",  attr.device);
    // attr.device 取值含义：
    //   >= 0               — device / managed 内存，值为对应 GPU 编号（0, 1, ...）
    //   -2 (cudaInvalidDeviceId) — host 内存（pageable），无关联 GPU
    //                             定义：driver_types.h: #define cudaInvalidDeviceId ((int)-2)
    printf("  attr.devicePointer = %p\n",  attr.devicePointer);   // nullptr 表示 GPU 不可直接访问
    printf("  attr.hostPointer   = %p\n\n", attr.hostPointer);    // nullptr 表示 CPU 不可直接访问
}


// ════════════════════════════════════════════════
// 示例 1：pageable host memory
//
// new / malloc 分配的普通主机内存。
// GPU 无法直接访问；cudaMemcpy 时驱动会先将数据复制到临时锁页缓冲区再传输。
// attr.type == cudaMemoryTypeUnregistered
// attr.devicePointer == nullptr
// ════════════════════════════════════════════════
static void demo_pageable_host()
{
    printf("─── 示例 1：pageable host memory ───\n");

    float *h = new float[N];
    printPointerAttributes("pageable host (new[])", h);

    delete[] h;
}

// ─── 示例 1：pageable host memory ───
// [pageable host (new[])]
//   attr.type          = unregistered host (pageable)
//   attr.device        = -2
//   attr.devicePointer = (nil)
//   attr.hostPointer   = 0x7eedf9bff010

// ════════════════════════════════════════════════
// 示例 2：pinned host memory（page-locked memory）
//
// cudaMallocHost 分配的锁页内存。
// CPU 和 GPU 均可直接访问（零拷贝路径或快速 DMA）。
// attr.type == cudaMemoryTypeHost
// attr.hostPointer 和 attr.devicePointer 均有效（指向同一物理页）
// ════════════════════════════════════════════════
static void demo_pinned_host()
{
    printf("─── 示例 2：pinned host memory ───\n");

    float *h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, N * sizeof(float)));
    printPointerAttributes("pinned host (cudaMallocHost)", h_pinned);

    CUDA_CHECK(cudaFreeHost(h_pinned));
}


// ─── 示例 2：pinned host memory ───
// [pinned host (cudaMallocHost)]
//   attr.type          = pinned host (registered)
//   attr.device        = 0
//   attr.devicePointer = 0x204c00000
//   attr.hostPointer   = 0x204c00000



// ════════════════════════════════════════════════
// 示例 3：device memory
//
// cudaMalloc 分配的纯设备内存。
// 仅 GPU 可直接访问；CPU 只能通过 cudaMemcpy 传输数据。
// attr.type   == cudaMemoryTypeDevice
// attr.device == 分配时所在 GPU 的编号
// attr.hostPointer == nullptr
// ════════════════════════════════════════════════
static void demo_device()
{
    printf("─── 示例 3：device memory ───\n");

    float *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
    printPointerAttributes("device (cudaMalloc)", d);

    CUDA_CHECK(cudaFree(d));
}



// ─── 示例 3：device memory ───
// [device (cudaMalloc)]
//   attr.type          = device
//   attr.device        = 0
//   attr.devicePointer = 0x706800000
//   attr.hostPointer   = (nil)



// ════════════════════════════════════════════════
// 示例 4：managed memory（Unified Memory）
//
// cudaMallocManaged 分配的统一内存。
// CPU 和 GPU 共享同一虚拟地址；驱动自动按需迁移数据页。
// attr.type == cudaMemoryTypeManaged
// attr.hostPointer == attr.devicePointer（同一地址）
// ════════════════════════════════════════════════
static void demo_managed()
{
    printf("─── 示例 4：managed memory ───\n");

    float *um = nullptr;
    CUDA_CHECK(cudaMallocManaged(&um, N * sizeof(float)));
    printPointerAttributes("managed (cudaMallocManaged)", um);

    CUDA_CHECK(cudaFree(um));
}


// ─── 示例 4：managed memory ───
// [managed (cudaMallocManaged)]
//   attr.type          = managed (unified)
//   attr.device        = 0
//   attr.devicePointer = 0x204c00000
//   attr.hostPointer   = 0x204c00000


int main(void)
{
    // dev：GPU 设备编号
    //   单卡机器固定为 0
    //   多卡机器编号为 0, 1, 2, ...，可通过 cudaGetDeviceCount(&count) 查询总数
    //   cudaGetDeviceProperties 根据 dev 查询对应 GPU 的名称、SM 版本、显存等属性
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    demo_pageable_host();
    demo_pinned_host();
    demo_device();
    demo_managed();

    printf("全部示例完成。\n");
    return 0;
}
