/**
 * prefetch_and_advise.cu
 *
 * 详解 cudaMemPrefetchAsync 与 cudaMemAdvise
 * 两者均作用于 cudaMallocManaged 分配的统一内存（Unified Memory）。
 *
 * 编译：
 *   nvcc -arch=native -std=c++20 prefetch_and_advise.cu -o prefetch_and_advise
 *
 * 运行：
 *   ./prefetch_and_advise
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

// ═════════════════════════════
// 背景概念
// ═════════════════════════════
//
// ── 一、UVA / UVM 区别 ──────────
//
//   UVA（Unified Virtual Addressing，CUDA 4.0）
//     统一 CPU 内存和 GPU 显存的虚拟地址空间。
//     实现方式：CUDA 驱动在软件层面协调 CPU 和 GPU 两套页表，
//     确保同一虚拟地址在两套页表中均有映射：
//       CPU 页表：虚拟地址 A → 物理页 P（CPU MMU 解析，访问 DRAM）
//       GPU 页表：虚拟地址 A → 物理页 P（GPU MMU 解析，经 PCIe 访问 DRAM）
//     CPU MMU 与 GPU MMU 仍是各自独立的硬件，各自查各自的页表；
//     "统一"是驱动让两套页表对同一虚拟地址填入一致的映射条目。
//     效果：同一指针值在 CPU 代码和 GPU kernel 中均可直接使用，无需两个指针。
//
//     无 UVA 时（CUDA 3.x 之前）：
//       同一块 pinned 物理页在两套页表中的虚拟地址不同：
//         CPU 页表：虚拟地址 A → 物理页 P
//         GPU 页表：虚拟地址 B → 物理页 P（同一物理页，但虚拟地址不同）
//       访问同一数据需要两个指针：h_ptr（CPU 用）、d_ptr（GPU 用）
//       d_ptr 须通过 cudaHostGetDevicePointer(&d_ptr, h_ptr, 0) 获取
//
//     UVA 只统一地址，不负责自动迁移，页仍需 pinned 或手动管理。
//
//   UVM（Unified Virtual Memory，即 cudaMallocManaged，CUDA 6.0）
//     在 UVA 基础上增加按需自动迁移（page fault 驱动）。
//     GPU 访问未就位的页 → 触发 page fault → UVM 驱动自动迁移或建立远程映射。
//     UVA：统一地址（一个指针）
//     UVM：统一地址 + 自动迁移（cudaMallocManaged 的核心价值）
//
//   UVM 的两条访问路径（由 hint 决定，详见 cudaMemAdvise 章节）：
//     路径 A（SetAccessedBy / hint）：预建立远程映射，GPU 经 PCIe 访问，不迁移
//     路径 B（默认，无 hint）      ：按需迁移，首次访问触发 page fault
//
// ── 二、cudaMallocManaged flags：可见性范围 vs 迁移策略 ──────────
//
//   cudaMallocManaged(void **ptr, size_t size, unsigned int flags)
//     flags 参数控制内存的可见性范围（accessibility scope）：
//       cudaMemAttachGlobal (默认)：内存对所有设备、所有 stream 立即可见
//       cudaMemAttachHost          ：内存初始仅对 CPU 可见；需调用
//                                    cudaStreamAttachMemAsync(stream, ptr) 后
//                                    才对指定 stream 的 GPU 可见
//
//   flags 与迁移策略是两个完全正交的维度，互不影响：
//     · flags（可见性）：控制"谁被允许访问"，是访问权限的门控
//     · 迁移策略      ：控制"访问时页在哪、怎么搬"
//   无论 flags 是 Global 还是 Host，一旦权限开放，UVM 的迁移机制
//   （demand paging + 驱动启发式）独立运作，与 flags 无关。
//
//   两个维度对照：
//     可见性范围：cudaMemAttachGlobal / cudaMemAttachHost
//                 + cudaStreamAttachMemAsync   → 谁被允许访问
//     迁移策略  ：cudaMemAdvise / cudaMemPrefetchAsync / UVM 启发式
//                                              → 访问时页在哪、怎么搬
//
// ── 三、默认行为：demand paging + UVM 启发式 ────
//
//   基础机制（demand paging）：
//     GPU 访问尚未迁移的页 → 触发 page fault → 驱动将页从 CPU 迁移到 GPU
//     CPU 访问 GPU 上的页  → 触发 page fault → 驱动将页从 GPU 迁移到 CPU
//     page fault 有延迟开销，频繁触发会显著降低性能
//
//   UVM 驱动叠加的启发式策略（自动、用户无感知）：
//     1. 局部性预取（locality prefetch）：
//        处理一个 page fault 时，驱动同时迁移相邻的若干页（空间局部性），
//        减少后续 fault 次数；迁移粒度通常为 2 MB（大页）而非 4 KB
//     2. 访问计数器驱动的主动迁移（Volta+ / SM 7.0+ 硬件支持）：
//        GPU 硬件内置访问计数器（Access Counters），统计每页的访问频率；
//        若某页被远程访问（PCIe 路径）次数超过阈值，驱动主动将其迁移到
//        访问方的本地内存，无需等待显式 page fault
//   两种启发式均是驱动内部优化，不改变 demand paging 的基础模型。
//   cudaMemAdvise / cudaMemPrefetchAsync 是用户层面的显式控制手段，
//   优先级高于驱动启发式。
//
// ── 四、两种显式优化 API 概览 ────
//
//   · cudaMemPrefetchAsync — 主动预取：提前将数据迁移到目标设备，避免 page fault
//   · cudaMemAdvise        — 访问提示：告知驱动数据的使用模式，驱动据此优化迁移策略


static const int N = 1 << 20;   // 1M 个 float，4 MB

// ────────────
// 简单的向量加法 kernel
// ────────────
__global__ void vecAdd(float *a, const float *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ════════════════
// cudaMemPrefetchAsync
// ════════════════
//
// 函数签名：
//   cudaError_t cudaMemPrefetchAsync(const void  *devPtr,
//                                    size_t       count,
//                                    int          dstDevice,
//                                    cudaStream_t stream)
//
// 参数：
//   devPtr    — 统一内存指针（cudaMallocManaged 分配），指定预取范围的起始地址
//   count     — 预取的字节数
//   dstDevice — 目标设备编号
//                 >= 0              : 预取到指定 GPU（编号 0, 1, 2, ...）
//                 cudaCpuDeviceId   : 预取到 CPU 主机内存（值为 -1）
//   stream    — 操作所在的 CUDA stream；传 0 使用默认 stream
//               预取操作在 stream 中异步执行，不阻塞 CPU
//
// 返回值：
//   cudaSuccess            — 成功提交预取请求
//   cudaErrorInvalidValue  — devPtr 非统一内存地址，或 dstDevice 无效
//
// 作用：
//   在 kernel 启动前将数据主动迁移到目标设备，
//   使 kernel 执行时数据已在本地，避免执行期间触发 page fault。
//   是异步操作：调用立即返回，迁移在 stream 中后台执行。
//
//   注意：向 GPU 预取要求设备支持 cudaDevAttrConcurrentManagedAccess。
//   WSL2 通常不支持此特性，调用会返回 cudaErrorInvalidDevice。
//   向 CPU 预取（cudaCpuDeviceId）在 WSL2 上同样受限，需统一跳过。
//
static void demo_prefetch(int dev)
{
    printf("─── 示例 1：cudaMemPrefetchAsync ───\n");

    // cudaDevAttrConcurrentManagedAccess = 0 时（如 WSL2），所有预取调用均跳过，
    // kernel 改由 page fault 按需迁移，结果仍然正确。
    int concurrentAccess = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentAccess,
                                     cudaDevAttrConcurrentManagedAccess, dev));
    if (!concurrentAccess)
        printf("  [注意] 设备不支持 concurrentManagedAccess，跳过 GPU 预取\n");

    float *a, *b;
    CUDA_CHECK(cudaMallocManaged(&a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, N * sizeof(float)));

    // CPU 初始化数据（此时数据在 CPU 侧）
    for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f;}

    // ── 预取到 GPU ──────────
    // 在 kernel 启动前将 a、b 迁移到 GPU，避免 kernel 执行时触发 page fault
    // dstDevice = dev（GPU 编号 0），stream = 0（默认 stream）
    if (concurrentAccess) {
        CUDA_CHECK(cudaMemPrefetchAsync(a, N * sizeof(float), dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(b, N * sizeof(float), dev, 0));
    }

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(a, b, N);
    CUDA_CHECK(cudaGetLastError());

    // ── 预取回 CPU ──────────
    // kernel 完成后将结果迁移回 CPU，避免 CPU 访问时触发 page fault
    // dstDevice = cudaCpuDeviceId（值为 -1，表示 CPU）
    CUDA_CHECK(cudaDeviceSynchronize());
    if (concurrentAccess)
        CUDA_CHECK(cudaMemPrefetchAsync(a, N * sizeof(float), cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  a[0] = %.1f（期望 3.0）\n", a[0]);
    printf("  [PASS] cudaMemPrefetchAsync 示例完成\n\n");

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
}

// ═════════════════════════════
// cudaMemAdvise
// ═════════════════════════════
//
// 函数签名：
//   cudaError_t cudaMemAdvise(const void           *devPtr,
//                             size_t                count,
//                             enum cudaMemoryAdvise advice,
//                             int                   device)
//
// 参数：
//   devPtr  — 统一内存指针，指定建议作用范围的起始地址
//   count   — 建议作用的字节数
//   advice  — 访问模式建议（见下方枚举说明）
//   device  — 建议关联的设备编号（部分 advice 需要指定设备）
//
// 返回值：
//   cudaSuccess            — 建议已登记
//   cudaErrorInvalidValue  — devPtr 非统一内存，或 advice 无效
//
// ── advice 枚举值说明 ────────────
//
//   (1) cudaMemAdviseSetReadMostly
//     数据以读为主，偶尔写入。
//     驱动在多个设备上创建只读副本，各设备可并发读取无需迁移。
//     写入时所有副本失效，驱动重新同步。
//     适合：权重、常量、只读查找表。
//
//     SetReadMostly(ptr, dev) 是否立即在 dev 上建立只读副本？
//       不是。调用时只登记 hint，副本在以下时机才建立：
//         · dev 第一次读取 ptr 时（page fault → UVM 建立只读副本到 dev VRAM）
//         · 显式调用 cudaMemPrefetchAsync(ptr, dev) 时（主动推送）
//
//     dev 参数的作用：
//       只对指定的 dev 启用只读副本优化。
//       其他未指定的 GPU 访问该数据时仍走普通迁移逻辑，不会自动建立只读副本。
//       若需要多张 GPU 都能建立只读副本，需对每张 GPU 分别调用 SetReadMostly。
//
//     只读副本的生命周期：
//       多个 GPU 可同时各持一份只读副本，互不干扰，并发读取无迁移开销。
//       任意设备写入 → UVM 作废所有只读副本（写入方自身除外）→ 退回普通迁移模式。
//       因此收益完全建立在"写入极少"这个前提上。
//
//     副本存储位置：
//       GPU 侧副本：GPU 普通全局内存（VRAM/global memory），不是 constant memory
//         · SetReadMostly 副本无大小限制，存储在普通 VRAM 页，
//           "只读"由 GPU MMU 页表权限位（read-only PTE）实现：
//           驱动将这些页的 PTE 标记为只读，GPU 写入时 MMU 触发 fault，
//           驱动捕获后使所有副本失效并重新同步
//         · GPU constant memory（__constant__）是编译期声明的静态区域，
//           容量仅 64 KB，有专用缓存（broadcast cache），用于所有线程读同一地址
//       CPU 侧副本：普通 CPU DRAM，"只读"同样由页表权限位控制
//
//     与 GPU constant memory 的区别：
//       SetReadMostly 副本  — 运行时 API 控制，任意大小，存普通 VRAM，MMU PTE 只读保护
//       __constant__ 内存   — 编译期声明，64 KB 上限，专用硬件缓存，所有线程广播读
//
//   (2) cudaMemAdviseUnsetReadMostly
//     撤销 SetReadMostly，恢复默认按需迁移。
//
//   (3) cudaMemAdviseSetPreferredLocation
//     设置数据的首选驻留位置为指定 device（驻留偏好 hint，不是强制锁定）。
//     设置时数据不立即迁移，只是向驱动登记偏好，影响后续迁移决策：
//       · 首选设备访问：命中本地内存，无额外开销
//       · 非首选设备访问：
//           GPU 访问方：驱动优先建立远程映射（peer mapping），而不是把数据从首选设备迁走
//             例：SetPreferredLocation(output, GPU0)，GPU1 访问 output 时，
//                 驱动在 GPU1 GMMU 写入 PTE → GPU0 BAR1 物理地址，
//                 GPU1 经 PCIe/NVLink 远程读 GPU0 VRAM，output 留在 GPU0，不迁移
//           CPU 访问方：即使设置了 SetPreferredLocation(GPU)，CPU 访问 GPU 驻留页
//                       仍触发迁移（UVM 把页从 GPU VRAM 搬到 CPU DRAM）。
//                       UVM 不会为 CPU 建立指向 GPU BAR1 的远程映射。
//                       SetPreferredLocation 对 CPU 访问方无保护效果。
//       · 内存压力需驱逐页时：首选设备上的页优先保留，非首选设备上的副本先被回收
//     与 SetAccessedBy 的配合：
//       SetPreferredLocation 决定数据的"家"（住在哪）
//       SetAccessedBy        确保访问方提前在 GMMU 中建好映射（怎么去）
//     适合：主要由某一 GPU 访问，偶尔被其他设备读取的数据。
//
//     ── peer access（多卡 GPU↔GPU 直接显存互访）──
//
//       严格定义：peer access 特指 GPU↔GPU 之间直接互访对方显存，不经过 CPU。
//       注意：GPU 访问 CPU 侧统一内存（单卡场景）≠ peer access，见下方说明。
//
//       多卡场景：
//         无 peer access：GPU0 显存 → CPU 内存 → GPU1 显存（两次 DMA，CPU 居中协调）
//         有 peer access：GPU0 显存 ──NVLink/PCIe──→ GPU1 显存（一次直接传输）
//
//       为什么无 peer access 时必须经过 CPU 内存：
//         PCIe 物理上支持设备间直接通信，但需要驱动显式建立跨设备地址映射。
//         未启用时驱动将每块 GPU 显存视为独立隔离的地址空间，
//         GPU0 无法寻址 GPU1 的显存（BAR 未映射），只能由 CPU 居中中转。
//         启用后（cudaDeviceEnablePeerAccess）驱动将 GPU1 的显存 BAR 映射到
//         GPU0 的地址空间，GPU0 即可通过 PCIe 直接 DMA 写入 GPU1 显存。
//
//       无 BAR 映射时的完整传输过程（GPU0 VRAM → CPU DRAM → GPU1 VRAM）：
//         前提：GPU0 GMMU 页表中没有指向 GPU1 BAR1 的 PTE，
//               GPU0 无任何合法虚拟地址能翻译到 GPU1 VRAM 的物理地址，
//               即使 PCIe 链路物理上互通，寻址机制缺失，GPU0 无法直接写 GPU1。
//         驱动介入：cudaMemcpyPeer 检测到无 peer access
//           → 在 CPU DRAM 中分配/复用 pinned 中转缓冲区（物理地址范围 [B, B+size]）
//         阶段 1：GPU0 VRAM → CPU DRAM（第一次 DMA）
//           · 驱动编程 GPU0 Copy Engine：源 = GPU0 VRAM[S]，目的 = CPU DRAM[B]
//           · GPU0 Copy Engine 向 PCIe 发出 Memory Write TLP，目标地址 = B
//           · PCIe 根复合体路由到 CPU 内存控制器 → 写入 CPU DRAM[B]
//         阶段 2：CPU DRAM → GPU1 VRAM（第二次 DMA）
//           · 驱动编程 GPU1 Copy Engine：源 = CPU DRAM[B]，目的 = GPU1 VRAM[D]
//           · GPU1 Copy Engine 向 PCIe 发出 Memory Read TLP，源地址 = B
//           · PCIe 根复合体路由到 CPU 内存控制器 → 读取 DRAM[B] → 返回 GPU1
//           · GPU1 内存控制器写入 GPU1 VRAM[D]
//         DMA 引擎归属：
//           · 每块 GPU 芯片内部集成独立的 Copy Engine（DMA 引擎），与 SM 分离
//             GeForce 消费级：通常 1–2 个；数据中心 GPU（A100/H100）：多个
//           · 两阶段分别由 GPU0 和 GPU1 各自的 Copy Engine 驱动，
//             CPU 核心全程不参与数据搬运，只负责编程 DMA 引擎（驱动调度）
//           · "CPU 居中协调"指 CPU DRAM 作为中转存储，而非 CPU 核心执行 memcpy
//
//       BAR 映射（peer access 的核心机制）：
//         BAR（Base Address Register）是 PCIe 标准的设备地址注册机制：
//           · 每个 PCIe 设备在系统枚举时声明若干内存窗口（BAR0、BAR1...）
//           · OS/BIOS 将这些窗口分配到系统物理地址空间的某段连续范围
//           · GPU 的 BAR1（framebuffer BAR）通常将整块显存暴露到 PCIe 总线地址空间
//         启用 peer access 时，驱动的操作：
//           1. 读取 GPU1 BAR1 在系统物理地址空间中的起始地址
//           2. 将该地址范围写入 GPU0 的 GMMU 页表（PTE）
//              · 此处是 GPU0 自身的 GMMU 映射，不是 CPU 的 MMU
//              · CPU MMU（x86 页表，OS 管理）：CPU 虚拟地址 → CPU/DRAM 物理地址
//              · GPU GMMU（GPU 页表，CUDA 驱动管理）：GPU 虚拟地址 → 物理地址
//                物理地址可指向本卡 VRAM / GPU1 BAR1（peer access）/ CPU DRAM（UVM）
//           3. GPU0 GMMU 现在能将虚拟地址翻译到 GPU1 BAR1 的物理地址
//           4. GPU0 发出的 PCIe TLP 被 PCIe 交换机路由到 GPU1 内存控制器
//         结果：GPU0 发出普通 load/store，硬件自动转换为指向 GPU1 BAR1 的 PCIe 事务。
//
//         BAR 映射不是自动完成的，需要显式 API 调用：
//           步骤 1 — 查询：cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1)
//           步骤 2 — 启用：cudaSetDevice(gpu0);
//                          cudaDeviceEnablePeerAccess(gpu1, 0);
//                    此调用触发驱动建立 BAR 映射，写入 GPU0 GMMU 页表
//           步骤 3 — 之后 GPU0 的 kernel 才能直接使用 GPU1 显存指针
//           （对称访问需反向再调用一次）
//         未调用 cudaDeviceEnablePeerAccess 时：
//           GPU0 GMMU 无 GPU1 BAR 的有效 PTE，kernel 访问触发 fault 或未定义行为。
//         例外：cudaMemcpyPeer 驱动内部临时处理，不需预先 EnablePeerAccess，
//               但那只是数据搬运，不是 kernel 直接寻址对端显存。
//
//       传输介质由硬件拓扑决定，PCIe 和 NVLink 均可：
//         · PCIe：多数服务器/工作站 GPU 均支持，带宽约 16–32 GB/s（PCIe 4.0 x16 单向）
//         · NVLink：高端 GPU 专属硬件（V100/A100/H100），NVLink 4.0 单向可达 ~450 GB/s
//         · cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1) 统一查询，无论 PCIe 还是 NVLink
//
//     ── 单卡场景：GPU 访问 CPU 侧统一内存（非 peer access）────────
//
//       GPU 访问 CPU 侧统一内存 ≠ peer access：
//         · peer access 定义为 GPU↔GPU 直接显存互访；CPU 内存不是"另一个 GPU 的 BAR"
//         · GPU 访问 CPU 内存是 PCIe 架构的天然路径，无需 BAR 反向映射
//
//       为什么 CPU 内存不需要 BAR 映射：
//         "天然暴露"含义：CPU DRAM 的物理地址本身就是 PCIe 总线上有效的目标地址，
//         PCIe 根复合体负责将 PCIe 事务路由到 CPU 内存控制器，
//         无需像 GPU 显存那样先通过 BAR 暴露到系统物理地址空间。
//         （GPU 显存没有 BAR1 映射时，系统物理地址空间中没有对应的地址范围，
//          其他 PCIe 设备无法寻址它；CPU DRAM 的物理地址天然存在于系统地址空间）
//         注意：GMMU 仍需要 PTE——"天然"是指 CPU 物理地址无需 BAR 暴露步骤，
//               但驱动仍须在 GMMU 页表中写入 PTE 才能完成虚拟地址翻译。
//
//       pinned 内存物理地址对 GPU 的可见性：
//         GPU GMMU PTE 中存放的"物理地址"实质上是 PCIe 总线地址。
//         CPU DRAM 物理地址天然就是合法的 PCIe 总线地址（根复合体可路由），
//         因此 pinned 内存的物理地址可直接填入 GPU GMMU PTE。
//
//         GPU GMMU 的 PTE 可指向三类物理地址：
//           本卡 VRAM 物理地址     → GPU 本地内存控制器（最快）
//           CPU DRAM 物理地址      → PCIe → 根复合体 → CPU 内存控制器
//                                    前提：页必须 pinned（物理地址稳定）
//           另一 GPU BAR1 物理地址 → PCIe → 对端 GPU 内存控制器
//                                    前提：cudaDeviceEnablePeerAccess 已调用
//
//         DRAM 物理页 ≠ pinned memory：
//           DRAM 物理页 = CPU DRAM 中的任意内存页（pageable 或 pinned 均是）
//           pinned memory = DRAM 页的子集，被 OS 锁定、不会被换出（swap out）
//           GPU 通过 PCIe 访问时需要稳定物理地址，所以必须是 pinned 的
//           · cudaMallocHost：永久 pinned，物理地址始终稳定
//           · cudaMallocManaged 的 CPU 侧页：UVM 驱动按需临时 pin，CUDA 代码无感知
//         non-pinned（pageable）内存不可填入 PTE：OS 随时可能换出该页并改变物理地址，
//           GPU 访问将命中错误的物理页，产生数据错误或系统崩溃。
//
//       GPU 访问 CPU pinned 内存的完整寻址过程：
//         步骤 1 — CUDA 驱动 pin 住 CPU 内存页，获得稳定的物理地址 X
//         步骤 2 — 驱动在 GPU0 GMMU 页表中写入 PTE：虚拟地址 V → 物理地址 X
//                  （GMMU 需要 PTE 才能翻译地址，此步骤是必需的）
//         步骤 3 — GPU kernel 访问虚拟地址 V
//                  → GPU0 GMMU 查页表，PTE 命中，翻译得到物理地址 X
//         步骤 4 — GPU0 内存子系统判断 X 不在本卡 VRAM 地址范围内
//                  → 向 PCIe 总线发出 Memory Read TLP，目标地址 = X
//         步骤 5 — PCIe 根复合体收到 TLP
//                  → 识别 X 属于 CPU DRAM 地址范围，路由到 CPU 内存控制器
//                  （若启用了 IOMMU：验证权限，pinned 内存通常 1:1 直通映射）
//         步骤 6 — CPU 内存控制器从 DRAM 读取数据，经 PCIe 返回给 GPU0
//
//       PTE 的有效位（valid bit）：
//         每条 PTE 包含一个 valid 位，表示该映射条目当前是否可用：
//           valid = 1（有效）：GMMU 直接完成翻译，GPU 继续执行，不产生任何中断
//           valid = 0（无效）：GMMU 无法完成翻译 → 产生硬件 page fault 信号
//                              → 陷入 UVM 驱动处理
//
//       UVM 两条访问路径（由 hint 决定）：
//         路径 A：预建立远程映射（remote mapping，无 page fault）
//           触发条件：使用了 SetAccessedBy / SetPreferredLocation 等 hint
//           驱动提前写入 valid PTE：虚拟地址 A → CPU 物理页 P（valid = 1）
//           GPU 访问时 GMMU 命中有效 PTE，直接经 PCIe 访问 CPU DRAM，不触发 fault
//           页始终驻留 CPU DRAM，GPU 每次访问都经 PCIe（带宽受限）
//           适合：数据只访问少量次数，迁移开销 > 远程访问开销
//         路径 B：按需迁移（demand migration，触发 page fault）
//           触发条件：GMMU 页表中无有效 PTE（默认行为，未使用任何 hint）
//           GPU 访问 → GMMU 查表，PTE 无效 → 产生 page fault
//           → UVM 驱动在 GPU VRAM 分配物理页，将数据从 CPU DRAM 复制过来
//           → 更新 PTE：虚拟地址 A → GPU VRAM 物理地址（valid = 1）
//           → GPU 重试访问，命中本地 VRAM，速度快，无 PCIe 开销
//           缺点：首次访问有 page fault 延迟；CPU/GPU 反复交替访问会导致 thrashing
//           适合：数据会被 GPU 大量重复访问，一次迁移开销可被摊销
//
//   (4) cudaMemAdviseUnsetPreferredLocation
//     撤销 SetPreferredLocation。
//
//   (5) cudaMemAdviseSetAccessedBy
//     声明指定 device 会访问此内存。
//
//     谁对谁写入有效 PTE：
//       SetAccessedBy(ptr, dev)
//         → UVM 在 dev 的 GMMU 中写入有效 PTE，指向 ptr 当前所在的物理地址
//         → "dev 的 GMMU" 是页表的归属，"ptr 的物理地址" 是 PTE 的内容
//       UnsetAccessedBy(ptr, dev)
//         → UVM 撤销 dev 的 GMMU 中指向 ptr 的有效 PTE（置为 invalid）
//         → dev 上的 kernel 再访问 ptr 时 GMMU miss，触发 page fault，走正常迁移流程
//
//     本质：驱动在声明设备的 GMMU 页表中写入有效 PTE（valid = 1），
//           指向数据当前所在的物理地址（CPU DRAM 或另一 GPU 的 VRAM）。
//           GPU 访问时 GMMU 直接命中有效 PTE，不触发 page fault，不迁移数据，
//           通过 PCIe / NVLink 远程访问原址。
//
//     "直接映射"的含义（取决于访问对象）：
//       情形 A：访问目标是另一 GPU 的 managed memory（显存）
//         → 驱动建立 BAR 映射（P2P peer access）
//         → 声明设备的 GPU MMU PTE 指向目标 GPU BAR1 的物理地址
//         → 本质上就是 peer access BAR 映射
//       情形 B：访问目标是 CPU 侧统一内存
//         → 驱动建立 GPU MMU PTE 指向 CPU 物理页
//         → GPU 通过 PCIe 根复合体访问 CPU DRAM（不是 BAR 映射）
//         → 不是 peer access，但同样是预建立映射、避免 page fault
//       两种情形共同点：预建立地址映射，将"按需迁移"变为"远程直接访问"
//
//     数据不迁移但可通过 PCIe / NVLink 直接访问，避免 page fault。
//     适合：多 GPU 共享同一份数据，不希望触发迁移。
//
//   (6) cudaMemAdviseUnsetAccessedBy
//     撤销 SetAccessedBy，移除直接映射。
//
static void demo_advise(int dev)
{
    printf("─── 示例 2：cudaMemAdvise ───\n");

    float *weights, *output;
    CUDA_CHECK(cudaMallocManaged(&weights, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&output,  N * sizeof(float)));

    for (int i = 0; i < N; i++) { weights[i] = 0.5f; output[i] = 0.0f;}

    // WSL2 不支持 concurrentManagedAccess，指定 GPU device 的 cudaMemAdvise
    // 和 cudaMemPrefetchAsync 均会报 cudaErrorInvalidDevice，需统一跳过。
    int concurrentAccess = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentAccess,
                                     cudaDevAttrConcurrentManagedAccess, dev));

    // 三个 hint 分别作用于不同 buffer、解决不同问题，互不矛盾：
    //   SetReadMostly        → weights：允许驱动建只读副本
    //   SetAccessedBy        → weights：预建 GPU GMMU PTE，prefetch 完成前也不 fault
    //   SetPreferredLocation → output ：首选驻留 GPU，CPU 访问走 peer mapping
    if (concurrentAccess) {
        // ── SetReadMostly ───────
        // weights 只读，驱动在 GPU 上创建只读副本，GPU 读取无需迁移
        CUDA_CHECK(cudaMemAdvise(weights, N * sizeof(float),
                                 cudaMemAdviseSetReadMostly, dev));

        // ── SetPreferredLocation ───────
        // output 主要由 GPU 写入，设置首选驻留在 GPU。
        //
        // CPU 访问 output 会触发迁移：
        //   SetPreferredLocation 对 CPU 访问方无保护效果。
        //   若下方代码中 CPU 直接读写 output（未先 prefetch 回 CPU），
        //   UVM 会将 output 从 GPU VRAM 迁移到 CPU DRAM，违反首选位置的意图。
        //   正确做法：CPU 访问前先 cudaMemPrefetchAsync(output, cudaCpuDeviceId)，
        //   主动把数据迁回 CPU（见下方 "结果读回 CPU" 处）。
        //
        // 另一张 GPU 访问 output 不需要显式 cudaDeviceEnablePeerAccess：
        //   cudaMallocManaged 统一内存由 UVM 驱动自动管理 GMMU PTE，
        //   包括跨卡远程映射（在访问方 GMMU 写入指向本卡 BAR1 的 PTE），
        //   不需要调用 cudaDeviceEnablePeerAccess，也不会报错。
        //   cudaDeviceEnablePeerAccess 仅对 cudaMalloc 普通设备指针是必须的；
        //   统一内存场景下调用它是多余的，UVM 内部已处理。

        // 统一内存下 UVM 自动处理的场景
        //
        //     SetPreferredLocation(data, GPU0) + GPU1 访问：
        //       → UVM 自动在 GPU1 GMMU 写入 PTE → GPU0 BAR1
        //       → 不需要 EnablePeerAccess
        //
        //     SetAccessedBy(data, GPU0) + data 在 CPU DRAM：
        //       → UVM 自动在 GPU0 GMMU 写入 PTE → CPU DRAM 物理地址
        //       → 不需要 EnablePeerAccess
        //
        //     page fault（无任何 hint）：
        //       → UVM 响应 fault，迁移数据，更新 GMMU PTE
        //       → 不需要 EnablePeerAccess
        CUDA_CHECK(cudaMemAdvise(output, N * sizeof(float),
                                 cudaMemAdviseSetPreferredLocation, dev));

        // ── SetAccessedBy ───────
        // 声明 GPU(dev) 会访问 weights，驱动在 GPU GMMU 中预建有效 PTE（valid=1）。
        // cudaMemPrefetchAsync 是异步的，prefetch 完成前若 kernel 已开始访问 weights：
        //   有 SetAccessedBy → GMMU 命中有效 PTE，GPU 经 PCIe 远程访问 CPU DRAM，不 fault
        //   无 SetAccessedBy → GMMU 无有效 PTE，触发 page fault，等待迁移后再访问
        // prefetch 完成后驱动更新 PTE 指向 GPU VRAM，后续访问走本地显存，无 PCIe 开销。
        // SetAccessedBy(weights) 与 SetPreferredLocation(output) 作用于不同指针，不矛盾。
        CUDA_CHECK(cudaMemAdvise(weights, N * sizeof(float),
                                 cudaMemAdviseSetAccessedBy, dev));

        // 预取到 GPU（结合 Advise 使用效果最佳）
        // 注意：PrefetchAsync 的目标设备必须与 SetPreferredLocation 一致，否则产生矛盾：
        //   SetPreferredLocation(output, GPU0) + PrefetchAsync(output, GPU1)
        //     → prefetch 是显式迁移命令，优先级高于 hint，output 被搬到 GPU1
        //     → 但 hint 仍登记为 GPU0，数据位置与 hint 矛盾
        //     → GPU0 kernel 访问时 GMMU miss → page fault → UVM 按 hint 迁回 GPU0
        //     → prefetch 的工作被白做，额外多一次迁移
        //   结论：两者目标应保持一致，prefetch 不报错但会制造多余迁移
        CUDA_CHECK(cudaMemPrefetchAsync(weights, N * sizeof(float), dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(output,  N * sizeof(float), dev, 0));
    }

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(output, weights, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 结果读回CPU
    if (concurrentAccess)
        CUDA_CHECK(cudaMemPrefetchAsync(output, N * sizeof(float), cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  output[0] = %.1f（期望 0.5）\n", output[0]);
    printf("  [PASS] cudaMemAdvise 示例完成\n\n");

    // 撤销建议（恢复默认行为，与 Set 对称，同样需要 concurrentAccess 保护）
    if (concurrentAccess) {
        CUDA_CHECK(cudaMemAdvise(weights, N * sizeof(float),
                                 cudaMemAdviseUnsetReadMostly, dev));
        CUDA_CHECK(cudaMemAdvise(output, N * sizeof(float),
                                 cudaMemAdviseUnsetPreferredLocation, dev));
        CUDA_CHECK(cudaMemAdvise(weights, N * sizeof(float),
                                 cudaMemAdviseUnsetAccessedBy, dev));
    }

    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(output));
}


// ═════════════════════════════
// 示例 3：组合使用（典型推理场景）
// ═════════════════════════════
//
// 场景：模型权重只读，输入/输出数据读写
//   · 权重：SetReadMostly + Prefetch 到 GPU → GPU 并发读取，无迁移开销
//   · 输入：Prefetch 到 GPU → kernel 前数据已就位
//   · 输出：SetPreferredLocation GPU + Prefetch 回 CPU → CPU 读取无 page fault
//
static void demo_combined(int dev)
{
    printf("─── 示例 3：组合使用（推理场景）───\n");

    float *model_weights, *input, *output;
    CUDA_CHECK(cudaMallocManaged(&model_weights, N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&input,         N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&output,        N * sizeof(float)));

    // CPU 准备数据
    for (int i = 0; i < N; i++) {
        model_weights[i] = 1.0f;
        input[i]         = 2.0f;
        output[i]        = 0.0f;
    }

    int concurrentAccess = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentAccess,
                                     cudaDevAttrConcurrentManagedAccess, dev));

    if (concurrentAccess) {
        // 权重只读，建立 GPU 只读副本
        CUDA_CHECK(cudaMemAdvise(model_weights, N * sizeof(float),
                                 cudaMemAdviseSetReadMostly, dev));
        // 输出首选驻留 GPU
        CUDA_CHECK(cudaMemAdvise(output, N * sizeof(float),
                                 cudaMemAdviseSetPreferredLocation, dev));
        // 预取所有数据到 GPU
        CUDA_CHECK(cudaMemPrefetchAsync(model_weights, N * sizeof(float), dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(input,         N * sizeof(float), dev, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(output,        N * sizeof(float), dev, 0));
    }

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // 模拟两步推理：output = input + weights，再 output += weights
    vecAdd<<<blocks, threads>>>(output, input,         N);
    vecAdd<<<blocks, threads>>>(output, model_weights, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 结果预取回 CPU
    if (concurrentAccess)
        CUDA_CHECK(cudaMemPrefetchAsync(output, N * sizeof(float), cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  output[0] = %.1f（期望 3.0）\n", output[0]);
    printf("  [PASS] 组合使用示例完成\n\n");

    CUDA_CHECK(cudaFree(model_weights));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
}


int main(void)
{
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // cudaGetDeviceProperties 只查询设备属性，不创建 CUDA context。
    // cudaMemPrefetchAsync 要求目标设备已有 context，否则报 cudaErrorInvalidDevice。
    // cudaSetDevice 显式初始化 context，确保后续所有操作均在 dev 上执行。
    CUDA_CHECK(cudaSetDevice(dev));

    demo_prefetch(dev);
    demo_advise(dev);
    demo_combined(dev);

    printf("全部示例完成。\n");
    return 0;
}
