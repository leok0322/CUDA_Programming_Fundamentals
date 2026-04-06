/**
 * device_properties_um.cu
 *
 * 详解 cudaGetDeviceProperties 与统一内存相关属性，
 * 以及 UVA / Unified Memory / Limited UM / Full UM / HMM / ATS 概念关系。
 *
 * 编译：
 *   nvcc -arch=native -std=c++17 device_properties_um.cu -o device_properties_um
 *
 * 运行：
 *   ./device_properties_um
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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


// ════════════════════════
// 一、三个关键属性的含义
// ════════════════════════
//
// ── prop.managedMemory ───────────────────
//
//   是否支持统一内存（cudaMallocManaged）。
//   为 0 时 cudaMallocManaged 会返回错误，统一内存完全不可用。
//   条件：Kepler（SM 3.0）及以上 + 64 位程序。
//   现代 GPU 全部为 1。
//
// ── prop.concurrentManagedAccess ─────────
//
//   CPU 和 GPU 是否可以在同一时刻并发访问统一内存，即是否支持 Full UM。
//
//   = 0（Limited Unified Memory，旧行为）：
//     · GPU kernel 运行期间，CPU 不能访问统一内存（访问结果未定义）
//     · UVM 在 kernel 启动前把所有 managed 页整体迁移到 GPU
//     · kernel 结束后整体迁回 CPU
//     · 无按需缺页迁移，无细粒度控制
//     · 适用：Kepler/Maxwell GPU，或 Windows 平台（驱动限制）
//
//   = 1（Full Unified Memory，Pascal SM 6.0+ / Linux）：
//     · CPU 和 GPU 可同时访问，UVM 按需以页为单位迁移
//     · GPU 访问不在显存的页 → GPU page fault → UVM 迁移该页
//     · CPU 访问不在 DRAM 的页 → CPU page fault → UVM 迁移该页
//     · cudaMemAdvise / cudaMemPrefetchAsync 等 hint 全部有效
//     · 支持 oversubscription（managed 内存总量可超过 GPU 显存）
//
// ── prop.pageableMemoryAccess ────────────
//
//   GPU 是否可以直接访问 CPU 的可分页内存（普通 malloc 分配，未 pinned）。
//   这是 HMM（Heterogeneous Memory Management）的标志。
//
//   = 0（无 HMM）：
//     · GPU 只能访问 pinned 内存或统一内存
//     · 普通 malloc 内存若要传给 GPU，必须先 cudaHostRegister 或 cudaMemcpy
//
//   = 1（有 HMM）：
//     · GPU 可以直接访问普通 malloc 内存（通过 Linux 内核 HMM 机制）
//     · 驱动与内核协作，在 GPU GMMU 中建立指向可分页物理页的 PTE
//     · 无需 cudaHostAlloc 或 cudaMemcpy，零拷贝访问普通 host 内存
//     · 需要：Linux 5.14+ 内核 + 支持 HMM 的驱动 + Pascal+ GPU
//     · 注意：可分页内存物理地址可能变化（被 OS 换出换入），
//             驱动通过 MMU notifier 感知变化并更新 GPU PTE
//
// ── cudaDevAttrPageableMemoryAccessUsesHostPageTables（attr 100）────────
//
//   与 pageableMemoryAccess（attr 88）配套，进一步说明 GPU 访问可分页内存的
//   底层实现路径：GPU 是通过自己的 GMMU 页表访问，还是直接走 CPU 的页表。
//
//   = 0（软件 HMM 路径，GPU 维护独立 GMMU 页表）：
//     · GPU 有自己的 GMMU，驱动负责与 CPU 页表同步：
//         CPU 页表变化（页换出/迁移）
//           → Linux MMU notifier 通知 GPU 驱动
//           → 驱动更新 GPU GMMU 中对应的 PTE（软件介入）
//     · pageableMemoryAccess=1 时，此为常见路径（PCIe GPU + 软件 HMM）
//
//   = 1（硬件 ATS 路径，GPU 直接使用 CPU 页表）：
//     · GPU 不维护指向 CPU 内存的 GMMU 条目
//     · GPU TLB miss 时，通过 PCIe ATS（Address Translation Services）协议
//       向 CPU IOMMU 发送地址翻译请求，IOMMU 查 CPU 页表返回物理地址
//     · 硬件完成，无软件 MMU notifier 开销
//     · 需要：PCIe ATS 支持的硬件平台（如 IBM POWER + NVLink，或特定 x86 服务器）
//
//   两个属性的组合含义：
//   ┌──────────────────────┬─────────────────────────────────────────────────┐
//   │ pageableMemAccess=0  │ GPU 不能访问可分页内存，两个属性均无意义         │
//   │ pageableMemAccess=1  │ GPU 可访问可分页内存，进一步看 UsesHostPageTables│
//   │   UsesHostPageTables=0│ 软件 HMM：GPU 维护 GMMU，MMU notifier 同步     │
//   │   UsesHostPageTables=1│ 硬件 ATS：GPU 直接查 CPU IOMMU，无软件开销     │
//   └──────────────────────┴─────────────────────────────────────────────────┘


// ════════════════════════
// 二、判断逻辑与模式分类
// ════════════════════════
//
// cudaDeviceProp prop;
// cudaGetDeviceProperties(&prop, device);
//
// if (!prop.managedMemory) {
//     // 不支持统一内存（Kepler 以前或 32 位程序）
//
// } else if (prop.concurrentManagedAccess) {
//
//     if (prop.pageableMemoryAccess) {
//         // Full Unified Memory + HMM
//         // CPU 可分页内存对 GPU 直接可见
//         // 是否硬件缓存一致性取决于互连方式（NVLink C2C / ATS）
//         // 典型平台：GH200（Grace CPU + Hopper GPU，NVLink C2C）
//
//     } else {
//         // Full Unified Memory（软件 UVM，无 HMM）
//         // 按需缺页迁移，CPU/GPU 并发访问
//         // 典型平台：Pascal / Volta / Ampere + Linux + PCIe
//     }
//
// } else {
//     // Limited Unified Memory
//     // kernel 运行期间 CPU 不可访问，整体迁移，无细粒度控制
//     // 典型平台：Kepler / Maxwell，或任意 GPU + Windows
// }



// ── UVA 与 Unified Memory 的关系 ─────────────────────
//
//   UVA 是地址空间统一（虚拟地址唯一），不涉及内存迁移。
//   Unified Memory 建立在 UVA 之上，增加了自动迁移能力。
//   有 UVA 不一定有 Unified Memory（老平台有 UVA 但只有 Limited UM）。
//   有 Unified Memory 一定有 UVA。
//
// ── Full UM 与 HMM 的关系 ───
//
//   Full UM 是 CUDA 层面的概念（按需缺页 + CPU/GPU 并发）。
//   HMM 是 Linux 内核层面的机制，Full UM 可以利用 HMM 实现对
//   可分页内存的直接访问（pageableMemoryAccess = 1）。
//   Full UM 可以在无 HMM 的系统上运行（软件模式，只管理 managed 页）。
//     "只管理 managed 页"是指只管理 cudaMallocManaged 分配的页，
//     不是指 pinned 内存（pinned 内存由驱动静态建立 PTE，不属于 UVM 管理范畴）：
//
//       cudaMallocManaged → UVM 全权管理，按需缺页迁移，hint 全部有效        ✓
//       cudaHostAlloc     → 驱动静态 PTE，永远在 CPU DRAM，UVM 不介入        ✓（但不是 UVM 管理）
//       malloc            → 无 HMM 时 GPU 不可直接访问，需 cudaMemcpy 中转   ✗
//
//     HMM 的价值：把 UVM 能管理的范围从 cudaMallocManaged 扩展到普通 malloc 内存，
//     GPU 无需任何显式操作即可访问任意 CPU 内存（pageableMemoryAccess = 1）。
//
// ── ATS 与 HMM 的关系 ───────
//
//   两者都允许 GPU 访问 CPU 内存，但实现路径不同：
//
//   HMM — 软件路径：cudaMemAdvise
//     GPU 仍有自己的 GMMU 页表，驱动负责与 CPU 页表同步：
//       CPU 页表变化（页换出、迁移）
//         → Linux 内核通过 MMU notifier 通知 GPU 驱动
//         → 驱动更新 GPU GMMU 中对应的 PTE
//         → 软件介入，有通知开销
//     GPU 访问 CPU 内存时查自己的 GMMU，驱动提前建好 PTE。
//
//   ATS（Address Translation Services）— 硬件路径：
//     GPU 不再维护指向 CPU 内存的 GMMU 条目，
//     而是把地址翻译请求直接发给 CPU 的 IOMMU：
//       GPU TLB miss（访问 CPU 内存）
//         → GPU 通过 PCIe ATS 协议向 CPU IOMMU 发送地址翻译请求
//         → IOMMU 查 CPU 页表，返回物理地址给 GPU
//         → 硬件完成，无软件干预
//
//                 HMM                        ATS
//     谁维护映射   GPU 驱动（软件）             CPU IOMMU（硬件）
//     页表变化时   MMU notifier 通知驱动更新 PTE  IOMMU 直接感知，无需通知
//     开销         每次页表变化有软件通知开销      每次 TLB miss 有硬件请求开销
//
//   NVLink C2C — 硬件缓存一致性（GH200 Grace+Hopper）：
//     ATS 解决了地址翻译，但 CPU/GPU cache 仍独立，无一致性保证。
//     NVLink C2C 在此基础上更进一步：
//       CPU L3 cache 和 GPU L2 cache 共享同一套硬件一致性协议
//         → CPU 写入 cache line，GPU 读同一 cache line
//         → 硬件自动返回最新值，无迁移，无软件干预
//         → 真正的"同一块内存"语义
//
//   三者的层次关系：
//     HMM        — 软件同步页表，GPU 可访问 malloc 内存
//       ↓ 硬件化
//     ATS        — 硬件地址翻译，消除软件通知开销
//       ↓ 更紧密互连
//     NVLink C2C — 硬件 cache 一致性，消除迁移本身


// ════════════════════════
// 四、各平台典型配置
// ════════════════════════
//
//   平台                  managedMemory  concurrentManaged  pageableMemAccess  UsesHostPageTables
//   ─────────────────────────────────────────────────────────────────────────────────────────
//   Kepler / Maxwell      1              0                  0                  0  → Limited UM
//   任意 GPU + Windows    1              0                  0                  0  → Limited UM
//   Pascal+ / Linux PCIe  1              1                  0                  0  → Full UM（软件 UVM）
//   Ampere+ / Linux HMM   1              1                  1                  0  → Full UM + HMM（软件页表）
//   ATS 平台（POWER+NVLink）1            1                  1                  1  → Full UM + HMM + ATS（硬件页表）
//   GH200 NVLink C2C      1              1                  1                  1  → Full UM + 硬件缓存一致性
//   WSL2                  1              0                  0                  0  → Limited UM（驱动限制）


static void print_um_properties(int device)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // cudaDevAttrPageableMemoryAccessUsesHostPageTables 无对应 cudaDeviceProp 字段，
    // 需用 cudaDeviceGetAttribute 单独查询。
    int usesHostPageTables = 0;
    cudaDeviceGetAttribute(&usesHostPageTables,
                           cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);

    printf("GPU %d：%s  (SM %d.%d)\n", device, prop.name, prop.major, prop.minor);
    printf("  managedMemory               = %d\n", prop.managedMemory);
    printf("  concurrentManagedAccess     = %d\n", prop.concurrentManagedAccess);
    printf("  pageableMemoryAccess        = %d\n", prop.pageableMemoryAccess);
    printf("  pageableMemAccessUsesHostPT = %d\n", usesHostPageTables);

    // 判断模式
    printf("  → 统一内存模式：");
    if (!prop.managedMemory) {
        printf("不支持统一内存\n");
    } else if (prop.concurrentManagedAccess) {
        if (prop.pageableMemoryAccess) {
            if (usesHostPageTables)
                printf("Full UM + HMM + ATS（硬件页表，GPU 直接查 CPU IOMMU）\n");
            else
                printf("Full UM + HMM（软件页表，MMU notifier 同步 GMMU）\n");
        } else {
            printf("Full UM（软件 UVM，按需缺页迁移）\n");
        }
    } else {
        printf("Limited UM（整体迁移，GPU 独占访问）\n");
    }

    // 附加属性
    printf("  unifiedAddressing           = %d  (UVA)\n",   prop.unifiedAddressing);
    printf("  canMapHostMemory            = %d  (mapped pinned mem)\n", prop.canMapHostMemory);
    printf("\n");
}

int main(void)
{
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    printf("检测到 %d 张 GPU\n\n", gpuCount);

    for (int i = 0; i < gpuCount; i++)
        print_um_properties(i);

    return 0;
}


// "/home/liam/cpp_linux/CUDA Programming Guide/Programming GPUs in CUDA/cmake-build-debug/device_properties_um"
// 检测到 1 张 GPU
//
// GPU 0：NVIDIA GeForce RTX 3060 Laptop GPU  (SM 8.6)
//   managedMemory               = 1
//   concurrentManagedAccess     = 0
//   pageableMemoryAccess        = 0
//   pageableMemAccessUsesHostPT = 0
//   → 统一内存模式：Limited UM（整体迁移，GPU 独占访问）
//   unifiedAddressing           = 1  (UVA)
//   canMapHostMemory            = 1  (mapped pinned mem)