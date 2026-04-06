/**
 * peer_access.cu
 *
 * 详解 cudaDeviceCanAccessPeer 与 cudaDeviceEnablePeerAccess
 * 两者均用于 GPU↔GPU 之间的 peer access（P2P 直接显存互访）。
 *
 * 编译：
 *   nvcc -arch=native -std=c++17 peer_access.cu -o peer_access
 *
 * 运行：
 *   ./peer_access
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ──────
// 错误检查宏
// ──────
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


// ══════════════════════════
// 背景：什么是 peer access
// ══════════════════════════
//
// peer access（P2P，Peer-to-Peer）= GPU↔GPU 直接显存互访，不经过 CPU DRAM。
//
// 无 peer access 时的 GPU 间拷贝（两次 DMA，CPU 居中）：
//   GPU0 Copy Engine → CPU DRAM（bounce buffer）→ GPU1 Copy Engine
//
// 有 peer access 时的 GPU 间拷贝（一次直接传输）：
//   GPU0 Copy Engine ──PCIe/NVLink──→ GPU1 显存
//
// 底层机制（BAR 映射）：
//   1. GPU 通过 PCIe BAR1 将自己的显存暴露到系统物理地址空间
//   2. 驱动调用 cudaDeviceEnablePeerAccess 后，
//      在 GPU0 的 GMMU 页表中写入 PTE，将虚拟地址映射到 GPU1 BAR1 的物理地址
//   3. GPU0 发出 load/store，GMMU 将虚拟地址翻译为 GPU1 BAR1 物理地址，
//      硬件自动生成指向 GPU1 的 PCIe 事务
//
// 注意：peer access 是有方向的。
//   GPU0 能访问 GPU1 ≠ GPU1 能访问 GPU0
//   需要分别在 GPU0 上 enable GPU1，以及在 GPU1 上 enable GPU0。
//
// 硬件要求：
//   · 两卡须在同一 PCIe Root Complex 下（或通过 NVLink 连接）
//   · 某些平台（如 WSL2）不支持 peer access，canAccessPeer 返回 0


// ══════════════════════════
// 函数一：cudaDeviceCanAccessPeer
// ══════════════════════════
//
// 函数签名：
//   cudaError_t cudaDeviceCanAccessPeer(int        *canAccessPeer,
//                                       int         device,
//                                       int         peerDevice)
//
// 参数：
//   canAccessPeer — 输出参数
//                   1：硬件支持 device 直接访问 peerDevice 的显存
//                   0：不支持（需走 CPU bounce buffer）
//   device        — 发起访问的 GPU 编号（访问方）
//   peerDevice    — 被访问的 GPU 编号（目标方）
//
// 返回值：
//   cudaSuccess           — 查询成功（canAccessPeer 填入结果）
//   cudaErrorInvalidDevice — device 或 peerDevice 编号无效
//
// 特性：
//   · 只查询，不修改任何状态，无副作用
//   · 结果是有方向的：
//       canAccessPeer(0→1) == 1  不代表  canAccessPeer(1→0) == 1
//   · 调用前无需 cudaSetDevice，device 参数显式指定访问方
//   · 单卡机器（device == peerDevice）返回 0


// ══════════════════════════
// 函数二：cudaDeviceEnablePeerAccess
// ══════════════════════════
//
// 函数签名：
//   cudaError_t cudaDeviceEnablePeerAccess(int          peerDevice,
//                                          unsigned int flags)
//
// 参数：
//   peerDevice — 被访问的 GPU 编号（目标方，即"要打开通往哪张卡的通道"）
//   flags      — 保留字段，当前必须传 0（传其他值返回 cudaErrorInvalidValue）
//
// 返回值：
//   cudaSuccess                — 成功启用
//   cudaErrorInvalidDevice     — peerDevice 编号无效
//   cudaErrorInvalidValue      — flags != 0
//   cudaErrorPeerAccessUnsupported — 硬件不支持（应先用 CanAccessPeer 检查）
//   cudaErrorPeerAccessAlreadyEnabled — 已经启用过，重复调用
//
// 关键：作用于"当前设备"（current device）
//   调用前必须先 cudaSetDevice(device) 设置当前设备。
//   效果：在当前设备（device）的 GMMU 中建立指向 peerDevice BAR1 的映射能力，
//         使 device 上运行的 kernel / Copy Engine 可以直接寻址 peerDevice 的显存。
//
// 对称性：
//   cudaDeviceEnablePeerAccess 只开启单向通道（当前设备 → peerDevice）。
//   若需要双向互访，必须在两张卡上各调用一次：
//     cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);  // GPU0 → GPU1
//     cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);  // GPU1 → GPU0
//
// 适用范围：仅针对 cudaMalloc 分配的普通设备指针
//   cudaMalloc        — GPU 间访问必须显式调用 EnablePeerAccess，否则越界崩溃
//   cudaMallocManaged — UVM 驱动自动管理 GMMU PTE（包括跨卡远程映射），
//                       无需调用 EnablePeerAccess，调用也不会报错但属多余操作


// ══════════════════════════
// 函数三：cudaDeviceDisablePeerAccess（配套关闭）
// ══════════════════════════
//
// 函数签名：
//   cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
//
// 参数：
//   peerDevice — 要关闭通道的目标 GPU 编号
//
// 作用：撤销当前设备对 peerDevice 的 GMMU 映射，与 EnablePeerAccess 对称。
// 返回值：
//   cudaErrorPeerAccessNotEnabled — 尚未启用，不能关闭


// ═════════════════════
// cudaMemcpyPeerAsync 与 cudaMemcpyAsync 对比
// ═════════════════════
//
// ── cudaMemcpyPeerAsync ─────────
//
// 函数签名：
//   cudaError_t cudaMemcpyPeerAsync(void        *dst,    int dstDevice,
//                                   const void  *src,    int srcDevice,
//                                   size_t       count,
//                                   cudaStream_t stream)
//
// 专为 GPU↔GPU 直接拷贝设计，显式传入源/目标设备编号。
// 前提：已调用 cudaDeviceEnablePeerAccess 开启双向通道。
// 底层：srcDevice 的 Copy Engine 通过 peer access BAR 映射
//       直接 DMA 写入 dstDevice 显存，全程不经过 CPU DRAM。
//
// ── cudaMemcpyAsync ─────────────
//
// 函数签名：
//   cudaError_t cudaMemcpyAsync(void        *dst,
//                               const void  *src,
//                               size_t       count,
//                               cudaMemcpyKind kind,
//                               cudaStream_t   stream)
//
// 通用拷贝接口，通过 kind 指定方向：
//   cudaMemcpyHostToDevice   — CPU DRAM → GPU 显存
//   cudaMemcpyDeviceToHost   — GPU 显存 → CPU DRAM
//   cudaMemcpyDeviceToDevice — GPU 显存 → GPU 显存（同卡或跨卡）
//   cudaMemcpyDefault        — 由指针属性自动推断方向
//
// 跨卡时（cudaMemcpyDeviceToDevice / cudaMemcpyDefault）的行为：
//   · 若已启用 peer access：驱动自动走 P2P 直传，不经过 CPU DRAM
//                           行为与 cudaMemcpyPeerAsync 等价
//   · 若未启用 peer access：驱动经 CPU DRAM 中转（bounce buffer），
//                           需两次 DMA，带宽减半
//
// ── 两者对比 ───────────────────
//
//   │ 维度               │ cudaMemcpyPeerAsync       │ cudaMemcpyAsync            │
//   ├────────────────────┼───────────────────────────┼────────────────────────────┤
//   │ 适用场景           │ 仅 GPU↔GPU                │ 任意方向（通用）           │
//   │ 设备参数           │ 显式 srcDevice / dstDevice│ 无，靠 kind 或指针属性推断 │
//   │ peer access 依赖   │ 必须预先 Enable           │ 已 Enable 时自动 P2P       │
//   │                    │                           │ 未 Enable 时走 CPU 中转    │
//   │ 无 peer access退路 │ 无（直接失败或行为未定义）│ 有（自动降级走 bounce buf）│
//   │ 语义清晰度         │ 明确表达 P2P 意图         │ 隐式，需查 kind 才知方向   │
//
// 建议：
//   · 明确需要 P2P 直传且已验证 peer access 支持 → 用 cudaMemcpyPeerAsync，意图清晰
//   · 通用代码或需要自动降级 → 用 cudaMemcpyAsync，兼容性更好


static const int N = 1 << 20;   // 1M 个 float，4 MB


// ══════════════════════════
// 示例 1：查询所有 GPU 对之间的 peer access 支持情况
// ══════════════════════════
static void demo_can_access_peer(int gpuCount)
{
    printf("─── 示例 1：cudaDeviceCanAccessPeer 查询矩阵 ───\n");

    // 打印表头
    printf("  src \\ dst |");
    for (int dst = 0; dst < gpuCount; dst++)
        printf("  GPU%d", dst);
    printf("\n  ----------+");
    for (int dst = 0; dst < gpuCount; dst++)
        printf("------");
    printf("\n");

    // 逐对查询
    for (int src = 0; src < gpuCount; src++) {
        printf("  GPU%-6d |", src);
        for (int dst = 0; dst < gpuCount; dst++) {
            if (src == dst) {
                printf("   -  ");   // 同一张卡，无意义
                continue;
            }
            int canAccess = 0;
            // cudaDeviceCanAccessPeer 不修改当前设备，无需 cudaSetDevice
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, src, dst));
            printf("   %d  ", canAccess);
        }
        printf("\n");
    }
    printf("\n");
}


// ══════════════════════════
// 示例 2：启用双向 peer access 并做 GPU 间直接拷贝
//
// 单卡机器跳过此示例（peer access 至少需要两张卡）。
// ══════════════════════════
static void demo_enable_peer_access(int gpuCount)
{
    printf("─── 示例 2：cudaDeviceEnablePeerAccess + cudaMemcpyPeerAsync ───\n");

    if (gpuCount < 2) {
        printf("  单卡机器，跳过 peer access 示例（至少需要两张 GPU）。\n\n");
        return;
    }

    const int src = 0, dst = 1;

    // ── 步骤 1：检查硬件是否支持 ────────────
    int canAccess_0to1 = 0, canAccess_1to0 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess_0to1, src, dst));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess_1to0, dst, src));

    printf("  GPU%d → GPU%d peer access 支持：%s\n", src, dst, canAccess_0to1 ? "是" : "否");
    printf("  GPU%d → GPU%d peer access 支持：%s\n", dst, src, canAccess_1to0 ? "是" : "否");

    if (!canAccess_0to1 || !canAccess_1to0) {
        printf("  硬件不支持双向 peer access，跳过示例。\n\n");
        return;
    }

    // ── 步骤 2：启用双向 peer access ───────
    //
    // EnablePeerAccess 作用于"当前设备"，必须先 cudaSetDevice。
    //
    // GPU0 → GPU1：在 GPU0 的 GMMU 中建立指向 GPU1 BAR1 的映射
    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dst, 0));
    printf("  已启用：GPU%d → GPU%d\n", src, dst);

    // GPU1 → GPU0：在 GPU1 的 GMMU 中建立指向 GPU0 BAR1 的映射
    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(src, 0));
    printf("  已启用：GPU%d → GPU%d\n", dst, src);

    // ── 步骤 3：在两张卡上分别分配显存 ─────
    float *d_src = nullptr, *d_dst = nullptr;

    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));

    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));

    // ── 步骤 4：GPU 间直接拷贝（peer-to-peer，不经过 CPU DRAM）──
    //
    // cudaMemcpyPeerAsync 函数签名：
    //   cudaError_t cudaMemcpyPeerAsync(void       *dst,    int dstDevice,
    //                                   const void *src,    int srcDevice,
    //                                   size_t      count,
    //                                   cudaStream_t stream)
    //
    // 底层：在 srcDevice 的 Copy Engine 上发起 DMA，
    //       通过 peer access 的 BAR 映射直接写入 dstDevice 显存，
    //       全程不经过 CPU DRAM。
    CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dst, d_src, src,
                                   N * sizeof(float), 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  cudaMemcpyPeerAsync 完成：GPU%d → GPU%d，%zu MB\n",
           src, dst, N * sizeof(float) / (1 << 20));

    // ── 步骤 5：清理 ──
    //
    // 关闭 peer access（与 Enable 对称，作用于当前设备）
    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(dst));

    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(src));

    CUDA_CHECK(cudaSetDevice(src)); CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst)); CUDA_CHECK(cudaFree(d_dst));

    printf("  peer access 已关闭，显存已释放。\n\n");
}


// ══════════════════════════
// 示例 3：重复 Enable 的错误处理
//
// cudaDeviceEnablePeerAccess 对同一对 (currentDevice, peerDevice)
// 重复调用会返回 cudaErrorPeerAccessAlreadyEnabled，
// 实际代码中通常用 cudaGetLastError 清除，或先 Disable 再 Enable。
// ══════════════════════════
static void demo_already_enabled(int gpuCount)
{
    printf("─── 示例 3：重复 Enable 的错误处理 ───\n");

    if (gpuCount < 2) {
        printf("  单卡机器，跳过。\n\n");
        return;
    }

    const int src = 0, dst = 1;

    int canAccess = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, src, dst));
    if (!canAccess) {
        printf("  硬件不支持，跳过。\n\n");
        return;
    }

    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dst, 0));
    printf("  第一次 Enable：成功\n");

    // 第二次 Enable 同一对，预期返回 cudaErrorPeerAccessAlreadyEnabled
    cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // 清除错误状态，避免影响后续 CUDA_CHECK
        cudaGetLastError();
        printf("  第二次 Enable：cudaErrorPeerAccessAlreadyEnabled（符合预期）\n");
    } else if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] 意外错误：%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaDeviceDisablePeerAccess(dst));
    printf("  Disable 完成。\n\n");
}


int main(void)
{
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    printf("检测到 %d 张 GPU\n\n", gpuCount);

    // 打印各卡名称
    for (int i = 0; i < gpuCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU%d：%s  (SM %d.%d)\n", i, prop.name, prop.major, prop.minor);
    }
    printf("\n");

    demo_can_access_peer(gpuCount);
    demo_enable_peer_access(gpuCount);
    demo_already_enabled(gpuCount);

    printf("全部示例完成。\n");
    return 0;
}
