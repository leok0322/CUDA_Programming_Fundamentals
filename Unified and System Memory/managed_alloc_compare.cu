/**
 * managed_alloc_compare.cu
 *
 * 统一内存三种分配方式对比：
 *   方式一：cudaMallocManaged（cudaMemAttachGlobal / cudaMemAttachHost）
 *   方式二：cudaMallocFromPoolAsync（Managed 内存池，CUDA 11.2+）
 *   方式三：__managed__ 全局变量
 *
 * 程序先检测 GPU 能力，判断系统属于以下哪种模式：
 *   ┌──────────────────────┬────────────────────────────────────────────┐
 *   │ 模式                 │ 条件                                       │
 *   ├──────────────────────┼────────────────────────────────────────────┤
 *   │ HMM / ATS            │ pageableMemoryAccess = 1                   │
 *   │ Full UM（软件 UVM）  │ managed=1, concurrent=1, pageable=0        │
 *   │ Limited UM           │ managed=1, concurrent=0, pageable=0        │
 *   │ 不支持统一内存        │ managed=0                                  │
 *   └──────────────────────┴────────────────────────────────────────────┘
 * 然后在对应路径下演示三种 managed 分配方式，并对比分配开销与访问行为。
 *
 * 编译：
 *   nvcc -arch=native -std=c++20 managed_alloc_compare.cu -o managed_alloc_compare
 *
 * 运行：
 *   ./managed_alloc_compare
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ──────────────────────────────────
// 错误检查宏
//
// 为什么用宏而不用函数？三个函数无法替代的能力：
//
// 1. __FILE__ / __LINE__ 指向调用处，而非定义处
//      宏在预处理阶段展开，__LINE__ 替换为宏被调用那一行的行号。
//      函数版本的 __LINE__ 只能拿到函数定义处的行号，错误信息无意义。
//        CUDA_CHECK(cudaMalloc(...));  ← 假设在第 200 行
//        → stderr 输出 "file.cu:200"  ✓ 精确定位到出错的调用处
//
// 2. call 在宏体内才执行（延迟求值）
//      函数传参时 call 已经执行完毕，函数内部无法包裹 call 本身。
//      宏展开后 cudaError_t err = (call) 在宏体内执行，
//      可以在 call 前后插入任意逻辑（计时、日志、错误处理）。
//
// 3. 直接修改调用方变量（CUDA_TRY 的 ok 参数）
//      宏的参数是文本替换，展开后直接操作调用处的变量，
//      无需指针或引用。
// ──────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",                  \
                    __FILE__, __LINE__,  /* 调用处的文件名和行号 */           \
                    cudaGetErrorName(err), cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 非致命版：失败时返回 false，调用方决定如何处理。
// ok 是文本替换（非引用/指针），展开后直接赋值给调用处的变量，
// 因此宏执行后外部可直接读取 ok 对应的变量（见 pool_ok 的使用）。
#define CUDA_TRY(call, ok)                                                    \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        (ok) = (err == cudaSuccess);                                          \
        if (!ok) cudaGetLastError(); /* 见下方说明 */                         \
    } while (0)
// CUDA_TRY 失败时调用 cudaGetLastError() 的原因：
//
// CUDA Runtime 为每个 host 线程维护一个"粘性错误槽（sticky error）"。
// 任何 Runtime API 调用失败都会把错误码写入这个槽，成功调用不清除它。
//   cudaGetLastError()      → 读取并清除错误槽（归零为 cudaSuccess）
//   cudaPeekAtLastError()   → 只读，不清除
//
// 若不清除，已处理的非致命错误会污染后续的 CUDA_CHECK：
//   CUDA_TRY(cudaMemPoolCreate(...), ok)  // 失败，sticky = cudaErrorNotSupported
//   // 我们判断 ok=false，决定跳过，继续执行
//   CUDA_CHECK(cudaMallocManaged(...))    // 本身成功，但…
//   // CUDA_CHECK 内部读 cudaGetLastError()
//   // → 读到残留的 cudaErrorNotSupported，误判为当前调用失败 → exit()
//
// 清除后：
//   CUDA_TRY(cudaMemPoolCreate(...), ok)  // 失败
//   cudaGetLastError()                    // 清除 sticky，归零
//   CUDA_CHECK(cudaMallocManaged(...))    // 正常读到 cudaSuccess ✓


// ──────────────────────────────────
// 方式三：__managed__ 全局变量
//
// 必须在文件作用域（全局/命名空间）声明，不能在函数内部。
// 生命周期 = 程序运行期。CPU 和 GPU 均可直接访问同一份数据。
// ──────────────────────────────────
// __managed__ 全局变量位于静态存储区（BSS 段，无显式初始化值时），
// 生命周期 = 整个程序，与普通 C++ 全局变量完全一致。
// 遵循 C++ 静态存储期的零初始化规则：
//   无显式初始化值时，所有字节置 0（float → 0.0f，int → 0），
//   在 main() 执行前由 host 端完成，CUDA 运行时随后将初始值同步到 managed 内存。
// CUDA 在普通全局变量的基础上额外将这块静态存储区注册为 managed 内存，
// 使 CPU 和 GPU 共享同一虚拟地址，但存储区归属和生命周期规则不变。
// 对比：函数内 float arr[1024]; 是栈上垃圾值；malloc 是堆上垃圾值。
//
// g_data 是二进制中的符号（symbol），__managed__ 与 __device__ 的符号解析方式不同：
//
//   __device__ float d_data[1024]：
//     CPU 不能直接使用符号名，必须通过 cudaGetSymbolAddress 显式解析为设备指针：
//       float* ptr;
//       cudaGetSymbolAddress((void**)&ptr, d_data);  // 符号 → 设备指针
//     注意：ptr 是合法的 UVA 地址，可传给 cudaMemcpy，
//           但 CPU 不能解引用（*ptr = 1.0f → segfault），因为 CPU 页表无此映射。
//
//   __managed__ float g_data[1024]：
//     符号地址本身就是合法的 UVM 指针，CPU/GPU 直接用符号名访问，无需解析：
//       g_data[0] = 1.0f;            // CPU 直接访问 ✓
//       float* ptr = g_data;         // 直接取地址，ptr 是合法 UVM 指针
//       cudaMemAdvise(g_data, ...);  // 直接传给 CUDA API
//
// 为什么 __device__ 有 UVA 地址却不能被 CPU 解引用？
//   UVA（Unified Virtual Addressing）只统一了地址空间的命名，
//   保证地址值唯一、不冲突，但不保证 CPU 能访问该地址。
//   __device__ 物理页在 GPU VRAM，CPU 页表中无映射 → CPU 解引用 segfault。
//   CPU 访问 GPU VRAM 需经过 PCIe BAR1 窗口，BAR1 通常只有 256MB，
//   不足以映射全部显存，因此 __device__ 不建立 CPU 侧映射。
//
//   UVM（__managed__）在 UVA 基础上额外建立双侧映射：
//     CPU 页表：有映射 ✓ → CPU 可解引用，缺页时 UVM 迁移物理页
//     GPU 页表：有映射 ✓ → GPU 可解引用，缺页时 UVM 迁移物理页
//
//   UVA：解决"地址唯一性"（谁在哪里）
//   UVM：解决"双向可访问性"（谁都能读写）
//
// 底层机制：
//   编译期：nvcc 给 __managed__ 符号打特殊标记，存入二进制元数据
//       ↓
//   程序启动（main() 之前）：CUDA Runtime 扫描这些 managed symbols
//       ↓
//   驱动注册：在 CPU 页表和 GPU 页表中均建立映射，注册为 UVM managed 内存
//       ↓
//   之后：&g_data == 合法 UVM 指针，CPU/GPU 均可直接用符号名访问
__managed__ float g_data[1024];   // 静态存储区，无显式初始化 → 零初始化，所有元素 0.0f
__managed__ int   g_counter = 0;  // 静态存储区，显式初始化为 0（与零初始化结果相同，但意图更明确）

// ──────────────────────────────────
// GPU 能力描述
// ──────────────────────────────────
struct UMCap {
    int managed;    // cudaDevAttrManagedMemory
    int concurrent; // cudaDevAttrConcurrentManagedAccess
    int pageable;   // cudaDevAttrPageableMemoryAccess
};

// ──────────────────────────────────
// 工具：毫秒级 CPU 计时
//
// 这里选用 CPU 时钟（steady_clock）而非 CUDA Event，原因：
//
//   CUDA Event 的工作原理：
//     cudaEventRecord(e, stream) 把一个"打戳"操作插入 stream 队列，
//     GPU 执行到该位置时记录硬件时间戳；
//     cudaEventElapsedTime 读取两个戳之间的 GPU 时间。
//     → 只能测量 GPU 时间线上的操作（kernel、memcpy、stream 内分配）。
//     → 无法感知 CPU 上发生的事情。
//
//   本文件中被计时的操作：
//     ① cudaMallocManaged 分配耗时
//          纯 CPU 路径：驱动调用 → 虚拟地址申请 → UVM 注册
//          GPU 完全未介入，无法在 stream 中插入 Event → 必须用 CPU 时钟。
//
//     ② cudaMallocFromPoolAsync 入队耗时
//          测的是"把分配操作插入 stream 队列"这一 CPU 动作本身的开销，
//          不是 GPU 上实际执行分配的时间 → 必须用 CPU 时钟。
//
//     ③ 高频分配对比循环（端到端吞吐量）
//          包含：分配入队 + kernel 入队 + cudaStreamSynchronize 等待
//          目的是衡量实际应用中的总耗时，而不是单纯的 GPU 执行时间。
//          用 CPU 时钟测端到端墙上时间更贴近真实场景。
//          （若只想测 GPU kernel 纯执行时间，应改用 CUDA Event。）
//
//   适合用 CUDA Event 的场景：
//     · 精确测量 kernel 执行时间（排除 OS 调度抖动）
//     · 测 cudaMemcpy / cudaMemPrefetchAsync 的 GPU 传输时间
//     · 在 GPU 时间线上对齐多个操作的相对耗时
// ──────────────────────────────────
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double cpu_ms_since(Clock::time_point t0)
{
    return Ms(Clock::now() - t0).count();
}

// ──────────────────────────────────
// Kernel：用 GPU 填充数组
// ──────────────────────────────────
__global__ void fill_kernel(float* ptr, int n, float val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = val + (float)i;
}

// Kernel for __managed__ global：写入 g_data，递增 g_counter
__global__ void fill_global_kernel(int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g_data[i] = 100.0f + (float)i;
    if (i == 0) atomicAdd(&g_counter, 1);   // 只让线程 0 计一次
}

// ──────────────────────────────────
// 打印节分隔
// ──────────────────────────────────
static void section(const char* title)
{
    printf("\n══════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("══════════════════════════════════════════\n");
}


// ═════════════════════════════
// 方式一：cudaMallocManaged
//
// cudaMemAttachGlobal（默认）：分配后 CPU + GPU 均可立即访问。
// cudaMemAttachHost：初始仅 CPU 可访问，显式关联到 stream 后 GPU 可访问。
//   用途：延迟到真正需要时才让 GPU 建立映射，减少无关 kernel 的开销。
//
// 底层流程：
//   1. Driver 在 UVA 地址空间分配虚拟地址范围
//   2. 标记为 managed，加入 UVM 追踪表
//   3. 物理页初始驻留在 Host DRAM（延迟实际分配，首次访问触发 demand paging）
//   4. 返回统一指针（CPU/GPU 用同一指针值）
// ═════════════════════════════
static void demo_malloc_managed(int dev, cudaStream_t stream, const UMCap& cap)
{
    section("方式一：cudaMallocManaged");

    constexpr int N    = 1 << 20;        // 1 M 个 float = 4 MB
    const size_t  size = N * sizeof(float);

    // ── 1a. cudaMemAttachGlobal（默认模式） ──────────────────────
    printf("\n[1a] cudaMallocManaged（cudaMemAttachGlobal，默认）\n");

    // cudaMallocManaged 是同步系统调用，整个分配过程在 CPU 驱动层完成，
    // GPU 未参与 → CUDA Event 无法计时，用 CPU 时钟测驱动调用开销。
    auto t0 = Clock::now();
    float* ptr_global = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr_global, size));   // 等价于带 cudaMemAttachGlobal
    printf("  分配耗时（CPU）: %.3f ms\n", cpu_ms_since(t0));

    // Full UM 底层以 4KB 页为粒度按需迁移（GPU 访问缺页时触发 page fault，
    // UVM 驱动只迁移那一页）。PrefetchAsync 是对这一机制的主动优化：
    // 提前把指定范围的页异步迁移到 GPU，避免 kernel 运行时逐页 fault 的开销。
    //
    // 与 Limited UM"整体迁移"的对比：
    //   Limited UM：kernel 启动时运行时自动把整块分配一次性迁移到 GPU（同步阻塞）。
    //   Full UM  ：不 Prefetch 时靠按需 page fault（4KB 粒度，只迁移实际访问的页）；
    //              Prefetch 后虽然也迁移了整块 size，但是异步插入 stream，
    //              CPU 不阻塞，且内部仍是按页操作。
    //
    // cudaMemAdviseSetPreferredLocation(ptr, size, dev) 的迁移策略：
    //
    //   preferred device（dev）发生缺页：
    //     → page fault → UVM 将该页迁移到 dev VRAM              （数据移到本地）
    //
    //   其他 GPU 发生缺页：
    //     → page fault → UVM 发现 preferred = dev → 不迁移
    //     → 在访问方 GPU 的 GMMU 建立 Peer PTE，指向 dev 的物理页（数据留在 dev）
    //     → 访问方通过 PCIe P2P 或 NVLink 远程读写 dev VRAM
    //
    //   CPU 发生缺页：
    //     → 迁移到 CPU DRAM
    //     → 在 PCIe 系统上 CPU 无法建立到 GPU VRAM 的远程映射，只能迁移回来
    //     （NVLink C2C / ATS 硬件一致性系统例外，但普通 PCIe GPU 均如此）
    //
    // 对比 SetPreferredLocation = -1（CPU 为 preferred）：
    //   CPU 缺页 → 迁移到 CPU DRAM（preferred）
    //   GPU 缺页 → 不迁移，GPU 在自身页表建立 host 内存映射，
    //              通过 PCIe 远程读写 CPU DRAM（零拷贝风格，带宽受限于 PCIe）
    //   此方向的远程映射才称为 BAR 映射：GPU 访问 CPU 内存，不是 CPU 访问 GPU 内存。
    if (cap.managed && cap.concurrent) {
        CUDA_CHECK(cudaMemAdvise(ptr_global, size,
                                 cudaMemAdviseSetPreferredLocation, dev));
        CUDA_CHECK(cudaMemPrefetchAsync(ptr_global, size, dev, stream));
        printf("  [Full UM] 已设置 PreferredLocation 并发起 PrefetchAsync → GPU\n");
    } else {
        // Limited UM：不支持按需 page fault 迁移，Advise/Prefetch API 不可用。
        // 运行时在 kernel 启动时同步地将全部 managed 页整体迁移到 GPU。
        printf("  [Limited UM] 跳过 Advise/Prefetch（不支持按需页迁移，迁移由运行时在 kernel 启动时整体完成）\n");
    }

    // GPU 填充
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    fill_kernel<<<blocks, threads, 0, stream>>>(ptr_global, N, 1.0f);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU 验证
    bool ok = true;
    for (int i = 0; i < 10 && ok; ++i)
        if (ptr_global[i] != 1.0f + (float)i) ok = false;
    printf("  CPU 验证：ptr_global[0..9] = [%.1f ... %.1f]  %s\n",
           ptr_global[0], ptr_global[9], ok ? "✓" : "✗");

    CUDA_CHECK(cudaFree(ptr_global));

    // ── 1b. cudaMemAttachHost ────────────────────────────────────
    printf("\n[1b] cudaMallocManaged（cudaMemAttachHost + cudaStreamAttachMemAsync）\n");

    //
    // cudaMemAttachHost 的用途：
    //   假设你有多个 stream 并行运行不同任务，某块 managed 内存只属于 stream S。
    //   使用 AttachHost 分配后，其他 stream 的 kernel 不会意外访问它，
    //   只有显式 cudaStreamAttachMemAsync(stream, ptr) 之后，stream 上的 kernel 才可访问。
    //   这减少了 UVM 对无关 stream 建立映射的开销。
    //
    t0 = Clock::now();
    float* ptr_host = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr_host, size, cudaMemAttachHost));
    printf("  分配耗时（CPU）: %.3f ms（初始仅 CPU 可访问）\n", cpu_ms_since(t0));

    // CPU 初始化（此时 GPU 还看不到这块内存）
    for (int i = 0; i < N; ++i) ptr_host[i] = 0.0f;

    // 关联到 stream 后，GPU 可以访问
    CUDA_CHECK(cudaStreamAttachMemAsync(stream, ptr_host));

    fill_kernel<<<blocks, threads, 0, stream>>>(ptr_host, N, 2.0f);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 同步后 CPU 可再次访问（DetachHost 是隐式的，sync 之后 CPU 可见）
    // 若要让 CPU 在 kernel 运行后立即再次独占，再调一次 AttachHost：
    //   cudaStreamAttachMemAsync(stream, ptr_host, 0, cudaMemAttachHost);
    ok = true;
    for (int i = 0; i < 10 && ok; ++i)
        if (ptr_host[i] != 2.0f + (float)i) ok = false;
    printf("  CPU 验证：ptr_host[0..9]  = [%.1f ... %.1f]  %s\n",
           ptr_host[0], ptr_host[9], ok ? "✓" : "✗");

    CUDA_CHECK(cudaFree(ptr_host));
}


// ═════════════════════════════
// 方式二：cudaMallocFromPoolAsync（流有序设备内存池）
//
// CUDA 11.2 引入的流有序内存分配器（stream-ordered memory allocator）。
// 通过 cudaMemPoolCreate 创建设备内存池，再用 cudaMallocFromPoolAsync
// 异步分配，分配操作插入 stream 队列，分配的内存仅 GPU 可访问。
//
// ── 关于"Managed 内存池"的说明 ──────────────────────────────────
//
//   cudaMemPoolProps.allocType 只有两个有效值（截至 CUDA 12.x）：
//     cudaMemAllocationTypeInvalid = 0  （无效）
//     cudaMemAllocationTypePinned  = 1  （设备内存，GPU 可访问）
//
//   并不存在 cudaMemAllocationTypeManaged。
//   cudaMallocFromPoolAsync 分配的始终是 GPU 设备内存（不可从 CPU 直接访问）。
//   若需 CPU 访问，仍须 cudaMemcpy，或改用 cudaMallocManaged。
//
// ── 与 cudaMallocManaged 的关键区别 ─────────────────────────────
//   ┌────────────────────────────┬──────────────────────────────┐
//   │ cudaMallocManaged          │ cudaMallocFromPoolAsync       │
//   ├────────────────────────────┼──────────────────────────────┤
//   │ 同步（阻塞 CPU 直到完成）  │ 异步（插入 stream，不阻塞）  │
//   │ 每次有系统调用开销          │ 池复用，减少系统调用          │
//   │ CPU + GPU 均可访问          │ 仅 GPU 可访问                 │
//   │ UVM 按需迁移               │ 无迁移，物理页固定在显存      │
//   │ 适合大块、低频分配          │ 适合高频、动态大小分配        │
//   │ CUDA 所有版本              │ CUDA 11.2+                    │
//   └────────────────────────────┴──────────────────────────────┘
// ═════════════════════════════
static void demo_pool_managed(int dev, cudaStream_t stream)
{
    section("方式二：cudaMallocFromPoolAsync（流有序设备内存池）");

    // ── 创建设备内存池 ────────────────────────────────────────────
    //
    // cudaMemPoolCreate 函数签名：
    //   cudaError_t cudaMemPoolCreate(
    //       cudaMemPool_t          *memPool,    // [out] 返回的池句柄（不透明指针）
    //       const cudaMemPoolProps *poolProps   // [in]  池属性
    //   );
    //
    // cudaMemPoolProps 关键字段：
    //   allocType  : 分配类型。目前唯一合法值是 cudaMemAllocationTypePinned，
    //                含义是"页锁定在 GPU VRAM（设备驻留）"，
    //                注意：与 Host Pinned Memory（cudaHostAlloc，页锁定在 CPU DRAM）
    //                完全不同，两者只是共用"Pinned"这个词。
    //   location   : 池所在设备（type=Device，id=GPU 编号）。
    //   maxSize    : 池的物理内存上限，0 = 系统默认（通常为 GPU 显存总量）。
    //
    // cudaMemPoolCreate 本身不分配任何物理内存，池初始大小为 0。
    // 它只创建一个池描述符（句柄），物理页在 cudaMallocFromPoolAsync 调用时
    // 才按需从 OS 申请；cudaFreeAsync 后物理页归还给池（不还给 OS）供复用；
    // cudaMemPoolDestroy 时池中所有物理页才最终归还 OS。
    //
    // 池的生命周期：
    //   cudaMemPoolCreate        → 创建句柄，物理内存：0 字节
    //   cudaMallocFromPoolAsync  → 按需向 OS 申请物理页，填入池，返回指针
    //   cudaFreeAsync            → 物理页归还给池（不还 OS），供下次复用
    //   cudaMemPoolTrimTo        → 将池中空闲页手动归还 OS
    //   cudaMemPoolDestroy       → 销毁句柄，所有物理页归还 OS
    //
    cudaMemPoolProps poolProps = {};
    poolProps.allocType         = cudaMemAllocationTypePinned;
    // location 是一个 cudaMemLocation 结构体：{ cudaMemLocationType type; int id; }
    // 这是"带标签的联合体（tagged type）"模式：type 是标签，id 的含义由 type 决定。
    //   cudaMemLocationTypeDevice          → id = GPU 设备编号（ordinal，即 cudaSetDevice 用的那个）
    //   cudaMemLocationTypeHost            → id 忽略
    //   cudaMemLocationTypeHostNuma        → id = NUMA 节点编号
    //   cudaMemLocationTypeHostNumaCurrent → id 忽略（自动选当前线程最近的 NUMA 节点）
    //
    // "id 是 device ordinal"是 API 约定，写在枚举注释里，调用方和实现方共同遵守。
    // switch 逻辑由两层共同完成，职责不同：
    //
    //   libcudart.so（CUDA Runtime Library，属于 CUDA Toolkit）：
    //     · 校验参数合法性（type 是否有效，id 是否越界）
    //     · 将 cudaMemPoolProps 翻译成 Driver API 的 CUmemPoolProps
    //     · 概念上的 switch 在这里：把 cudaMemLocation 解析为 Driver 能理解的参数
    //     · 调用 Driver API：cuMemPoolCreate(...)
    //   概念上等价于：
    //   switch (loc.type) {
    //     case cudaMemLocationTypeDevice:
    //         use_device(loc.id);
    //         // id = GPU 设备编号，在该 GPU 的 VRAM 上操作
    //         // 例：loc.id = 1 → 在 GPU 1 的显存上分配内存
    //         break;
    //
    //     case cudaMemLocationTypeHost:
    //         use_host();
    //         // id 被忽略，操作对象是 CPU DRAM
    //         // Host 只有一个逻辑单元，不需要编号
    //         break;
    //
    //     case cudaMemLocationTypeHostNuma:
    //         use_numa(loc.id);
    //         // id = NUMA 节点编号
    //         // 多路服务器上每个 CPU socket 有自己的本地 DRAM（NUMA 节点），
    //         // 跨节点访问需经过 QPI/UPI 互连，延迟更高。
    //         // 例：双路服务器，Socket 0 → NUMA 0，Socket 1 → NUMA 1
    //         //     loc.id = 0 → 优先在 NUMA 0 的 DRAM 上分配（Socket 0 本地）
    //         //     loc.id = 1 → 优先在 NUMA 1 的 DRAM 上分配（Socket 1 本地）
    //         break;
    //
    //     case cudaMemLocationTypeHostNumaCurrent:
    //         use_nearest_numa();
    //         // id 被忽略，运行时自动选择当前线程所在 CPU 最近的 NUMA 节点，
    //         // 避免手动查询 NUMA 拓扑，让运行时自动决定最优本地内存
    //         break;
    //   }
    //
    //   libcuda.so（CUDA Driver，随 GPU 驱动安装，不属于 Toolkit）：
    //     · 真正与硬件 / OS 内核交互
    //     · 实际分配 GPU 虚拟地址范围，建立 VRAM 管理结构
    //     · 处理 NUMA 亲和性、PCIe 拓扑等硬件细节
    //
    //   两者均闭源，上述为概念示意。
    //   libcudart.so ≈ 翻译官（高层 API → 底层 API）
    //   libcuda.so   ≈ 实际执行者（底层硬件操作）
    //
    // 相同模式的经典例子：POSIX sockaddr
    //   struct sockaddr { sa_family_t sa_family; char sa_data[]; };
    //   sa_family = AF_INET  → sa_data 解释为 IPv4 地址+端口
    //   sa_family = AF_INET6 → sa_data 解释为 IPv6 地址
    //
    // type 和 id 职责不同，缺一不可：
    //   type → 分类标签，唯一作用是声明"id 的含义是什么"
    //          cudaMemLocationTypeDevice 表示"id 是 GPU 设备编号"
    //          若只有 Device 一种 location，根本不需要 type 字段，直接用 id 即可；
    //          正因为有多种 location（Device/Host/HostNuma/HostNumaCurrent），
    //          才需要 type 作为区分标签。
    //   id   → 实际选择，指定具体是哪个 GPU（0, 1, 2...）
    //          type 不选择任何设备，id 才是真正的选择。
    poolProps.location.type     = cudaMemLocationTypeDevice;
    poolProps.location.id       = dev;
    // poolProps.maxSize        = 0;  // 默认，无需显式设置

    // cudaMemPool_t 是不透明句柄（opaque handle）：
    //
    // 定义（driver_types.h）：
    //   typedef struct CUmemPoolHandle_st *cudaMemPool_t;
    //
    // CUmemPoolHandle_st 只有前向声明，没有结构体定义体：
    //   // cuda.h：
    //   typedef struct CUmemPoolHandle_st *CUmemoryPool;  // 只有这一行，无 struct 定义
    //
    // 因此 cudaMemPool_t 本质是一个指向不完整类型的指针（pointer to incomplete type）：
    //   · 用户代码拿到的只是一个指针值（地址）
    //   · 无法解引用（*pool）、无法访问任何成员（pool->xxx）
    //   · 编译器不知道结构体内部有什么，用户也不需要知道
    //   · 真正的内存池状态（物理页列表、引用计数等）保存在 libcuda.so 内部
    //
    // 不透明句柄的设计意图：
    //   · 封装：隐藏内部实现细节，驱动可以自由修改内部结构而不破坏 ABI
    //   · 安全：用户无法直接篡改内部状态，只能通过 API 操作
    //   · 轻量：变量本身只占一个指针大小（8 字节），传参高效
    //
    // 相同模式的例子：
    //   FILE*        → 标准库文件句柄，内部结构用户不可见
    //   cudaStream_t → typedef struct CUstream_st* cudaStream_t，同为不透明句柄
    //   cudaEvent_t  → typedef struct CUevent_st*  cudaEvent_t，同为不透明句柄
    cudaMemPool_t pool;
    // CUDA_TRY 是宏，不是函数。pool_ok 不是通过引用/指针传入，
    // 而是预处理器将宏体内的 ok 直接文本替换为 pool_ok，
    // 展开后等价于在此处内联写：
    //   do {
    //       cudaError_t err = (cudaMemPoolCreate(&pool, &poolProps));
    //       (pool_ok) = (err == cudaSuccess);   // 直接赋值给 pool_ok
    //       if (!pool_ok) cudaGetLastError();
    //   } while (0);
    // 因此宏执行后 pool_ok 在外部可直接读取，无需任何间接寻址。
    bool pool_ok = false;
    CUDA_TRY(cudaMemPoolCreate(&pool, &poolProps), pool_ok);
    if (!pool_ok) {
        printf("  [跳过] cudaMemPoolCreate 失败，当前驱动/平台不支持流有序分配器。\n");
        return;
    }
    printf("  设备内存池创建成功（device %d，初始物理内存 0 字节，按需增长）\n", dev);

    // 可选：设置 ReleaseThreshold，控制池保留多少空闲内存不还给 OS。
    // 默认为 0，即 cudaFreeAsync 后空闲页立即可被 OS 回收（但仍在池中复用）。
    // cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_thresh);

    constexpr int N    = 1 << 20;
    const size_t  size = N * sizeof(float);

    // ── 异步分配（插入 stream） ───────
    //
    // 函数签名：
    //   cudaError_t cudaMallocFromPoolAsync(
    //       void          **ptr,     // [out] 二级指针：函数通过 *ptr 写回分配到的设备地址。
    //                                //       用 void** 而非 float** 是为了类型无关（与 malloc 一致）。
    //                                //       调用处取地址：&ptr_pool，让函数能写回指针值。
    //       size_t          size,    // [in]  请求分配的字节数
    //       cudaMemPool_t   memPool, // [in]  从哪个池分配（不透明句柄，由 cudaMemPoolCreate 创建）
    //       cudaStream_t    stream   // [in]  将分配操作插入哪个 stream
    //   );
    //
    //
    // 执行语义（流有序）：
    //   · 函数本身是 CPU 调用，立即返回（不等待 GPU）。
    //   · 分配操作作为一个节点插入 stream 队列，在 stream 中前序操作完成后才执行。
    //   · *ptr 在函数返回时已写入有效地址（可传给后续 stream 操作），
    //     但该地址的物理页直到 stream 执行到此节点时才真正分配。
    //   · 因此不能在 stream 执行前从 CPU 访问 *ptr 指向的内存。
    //
    // 与 cudaMallocAsync 的区别：
    //   cudaMallocAsync(ptr, size, stream)
    //     → 等价于从当前设备默认池分配：
    //       cudaMallocFromPoolAsync(ptr, size, defaultPool, stream)
    //       其中 defaultPool = cudaDeviceGetDefaultMemPool()，每个设备自动创建。
    //     → 同样使用池机制，同样避免重复 OS 调用，高频分配性能与 FromPoolAsync 一致。
    //     → 更简洁，不需要手动 cudaMemPoolCreate / cudaMemPoolDestroy。
    //   cudaMallocFromPoolAsync(ptr, size, pool, stream)
    //     → 从用户自定义池分配，适合需要精细控制池行为的场景：
    //         maxSize / ReleaseThreshold / IPC 跨进程共享 / 跨设备指定池
    //   结论：高频分配场景两者均可，cudaMallocAsync 更简洁；
    //         只有需要定制池属性时才用 cudaMallocFromPoolAsync。
    //
    // cudaMallocFromPoolAsync 只是把"分配"操作插入 stream 队列（CPU 动作），
    // 真正的分配在 GPU stream 执行时发生。
    // 这里测的是"入队"这一 CPU 动作的开销 → 用 CPU 时钟，而非 CUDA Event。
    // （若要测 GPU 上实际分配+执行的时间，应在 stream 前后各插入一个 Event。）
    auto t0 = Clock::now();
    float* ptr_pool = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&ptr_pool, size, pool, stream));
    printf("  cudaMallocFromPoolAsync 入队耗时（CPU）: %.3f ms\n",
           cpu_ms_since(t0));   // 入队本身极快；GPU 实际分配时间含在后续 kernel 的 sync 中

    // GPU 填充（在同一 stream，保证分配在填充之前完成）
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    fill_kernel<<<blocks, threads, 0, stream>>>(ptr_pool, N, 3.0f);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ptr_pool 不是 Unified Memory，不能从 CPU 直接访问，必须 cudaMemcpy。
    //
    // cudaMallocFromPoolAsync 分配的是普通设备内存（Device Memory），
    // 物理页固定在 GPU VRAM，与 cudaMalloc 性质完全相同：
    //
    //   cudaMallocManaged       → UVM 管理，CPU + GPU 共享同一虚拟地址  ← Unified Memory
    //   cudaMalloc              → GPU VRAM，CPU 不可直接访问，需 cudaMemcpy
    //   cudaMallocFromPoolAsync → GPU VRAM（池化），CPU 不可直接访问，需 cudaMemcpy
    //
    // 方式二放在本文件中对比的原因：
    //   它是高频分配场景的替代方案——只需 GPU 计算、不需 CPU 直接读写时，
    //   池化设备内存比 cudaMallocManaged 快得多（实测 16x），
    //   代价是失去 CPU 直接访问能力，需显式 cudaMemcpy 才能读回结果。
    float sample[10];
    CUDA_CHECK(cudaMemcpy(sample, ptr_pool, 10 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < 10 && ok; ++i)
        if (sample[i] != 3.0f + (float)i) ok = false;
    printf("  CPU 验证（通过 cudaMemcpy）：ptr_pool[0..9] = [%.1f ... %.1f]  %s\n",
           sample[0], sample[9], ok ? "✓" : "✗");
    printf("  注：池内存是设备内存（非 Unified Memory），CPU 不可直接访问\n");

    // ── 异步释放 ─────────────────────────────────────────────────
    //
    // 函数签名：
    //   cudaError_t cudaFreeAsync(
    //       void         *devPtr,  // [in] 要释放的设备指针（一级指针，非二级）
    //                              //      与 cudaMallocFromPoolAsync 返回的 *ptr 对应。
    //                              //      注意：cudaFree 传的也是一级指针，两者一致。
    //       cudaStream_t  hStream  // [in] 将释放操作插入的 stream
    //   );
    //
    // 必须成对使用（cudaFreeAsync 只能释放流有序分配器分配的内存）：
    //   cudaMallocAsync         → cudaFreeAsync    ✓
    //   cudaMallocFromPoolAsync → cudaFreeAsync    ✓
    //   cudaMalloc              → cudaFree         ✓  （不能用 cudaFreeAsync）
    //   cudaMallocManaged       → cudaFree         ✓  （不能用 cudaFreeAsync）
    //
    // 原因：cudaMallocAsync / cudaMallocFromPoolAsync 分配时在内部记录了
    // 该指针的流有序元数据（所属池、stream 依赖关系等），cudaFreeAsync
    // 依赖这些元数据执行异步归还。cudaMalloc / cudaMallocManaged 分配的
    // 内存没有这套元数据，传给 cudaFreeAsync 会返回 cudaErrorInvalidValue。
    //
    // 与 cudaFree 的对比：
    //   cudaFree(devPtr)
    //     · 同步：CPU 阻塞直到释放完成
    //     · 物理页立即归还 OS（系统调用）
    //     · 函数返回后指针立即失效
    //
    //   cudaFreeAsync(devPtr, stream)
    //     · 异步：把"释放"操作插入 stream 队列，CPU 立即返回
    //     · 物理页在 stream 执行到此节点后归还给池（不归还 OS）
    //     · 函数返回后 CPU 不得再访问 devPtr（未定义行为），
    //       但同一 stream 中在此节点之前入队的 kernel 仍可访问
    //
    // 安全语义（流有序保证）：
    //   stream 中的顺序：[cudaMallocFromPoolAsync] → [kernel] → [cudaFreeAsync]
    //   GPU 按序执行，kernel 结束后才执行释放，不会出现 use-after-free。
    //   跨 stream 访问需用 cudaEvent 建立依赖，否则行为未定义。
    //
    // cudaFreeAsync 返回后 devPtr 不可再用（即使 GPU 尚未执行到释放节点）：
    //   CUDA_CHECK(cudaFreeAsync(ptr_pool, stream));
    //   ptr_pool[0] = 1.0f;  // ← 未定义行为，API 返回后 CPU 侧指针即视为无效
    CUDA_CHECK(cudaFreeAsync(ptr_pool, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 模拟高频分配：10 次循环 ─────────
    //
    // 为什么不直接用 cudaMalloc 而用 cudaMallocFromPoolAsync？
    //
    // 单次分配时两者差别不大，但高频反复分配时差距显著：
    //
    //   cudaMalloc / cudaFree（每次都走 OS）：
    //     · cudaMalloc：同步，CPU 阻塞，向 OS 申请物理页（系统调用）
    //     · cudaFree  ：同步，CPU 阻塞，物理页归还 OS（系统调用）
    //     · 下次再 cudaMalloc 时重新向 OS 申请
    //     → 每次迭代都有完整的系统调用开销
    //
    //   cudaMallocFromPoolAsync / cudaFreeAsync（池复用）：
    //     · 分配：异步入队，CPU 不阻塞；池中有空闲页则直接复用，无系统调用
    //     · 释放：异步入队，物理页还给池（不还 OS），供下次直接复用
    //     · 10 次循环中只有第一次可能向 OS 申请，后续全部复用池中物理页
    //     → 系统调用次数趋近于 0
    //
    // 典型场景：神经网络推理，每个 batch 的中间结果大小不同，
    // 每次迭代都要分配和释放临时 buffer，池化可大幅降低分配开销。
    //
    // 端到端吞吐量对比：用 CPU 时钟测总墙上时间（含分配、kernel、sync）。
    // 注：若只想精确测 GPU kernel 执行时间，应用 cudaEventRecord/ElapsedTime。

    // Pool 版：10 次全部入队，最后一次性 sync
    auto t_pool = Clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        float* p = nullptr;
        CUDA_CHECK(cudaMallocFromPoolAsync(&p, size, pool, stream));
        fill_kernel<<<blocks, threads, 0, stream>>>(p, N, (float)iter);
        CUDA_CHECK(cudaFreeAsync(p, stream));  // 还给池，不还 OS
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));  // 一次性等待所有 10 次迭代完成
    double ms_pool = cpu_ms_since(t_pool);

    // cudaMallocManaged 对照版：同步调用，每次必须在循环内 sync
    // （无法批量入队，因为 cudaMallocManaged 本身会阻塞 CPU）
    auto t_managed = Clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        float* p = nullptr;
        CUDA_CHECK(cudaMallocManaged(&p, size));         // 同步，阻塞，系统调用
        fill_kernel<<<blocks, threads, 0, stream>>>(p, N, (float)iter);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(p));                         // 同步，阻塞，物理页还 OS
    }
    double ms_managed = cpu_ms_since(t_managed);

    printf("  Pool  (cudaMallocFromPoolAsync × 10): %.2f ms\n", ms_pool);
    printf("  Naive (cudaMallocManaged       × 10): %.2f ms\n", ms_managed);
    printf("  加速比: %.2fx\n", ms_managed / ms_pool);

    CUDA_CHECK(cudaMemPoolDestroy(pool));
}


// ═════════════════════════════
// 方式三：__managed__ 全局变量
//
// 在文件顶部已声明：
//   __managed__ float g_data[1024];
//   __managed__ int   g_counter = 0;
//
// 特点：
//   ✓ 生命周期 = 整个程序，无需显式分配/释放
//   ✓ 可带 C++ 静态初始化值
//   ✓ CPU/GPU 均可直接用变量名访问（编译器自动处理指针）
//   ✗ 大小必须是编译期常量（不能动态大小）
//   ✗ 不适合大数组（污染 BSS/data 段，且大小固定）
//   ✓ 最适合：全局计数器、配置标志、小型共享状态
// ═════════════════════════════
static void demo_managed_global(cudaStream_t stream)
{
    section("方式三：__managed__ 全局变量");

    constexpr int N = 1024;   // 必须与声明大小一致

    printf("  [初始] g_counter = %d，g_data[0] = %.1f（静态初始化值）\n",
           g_counter, g_data[0]);

    // GPU 写入
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    fill_global_kernel<<<blocks, threads, 0, stream>>>(N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU 直接读取同一变量——无需任何 memcpy
    printf("  [GPU 写后，CPU 读] g_counter = %d\n", g_counter);
    printf("  g_data[0..4] = %.1f  %.1f  %.1f  %.1f  %.1f\n",
           g_data[0], g_data[1], g_data[2], g_data[3], g_data[4]);

    bool ok = true;
    for (int i = 0; i < N && ok; ++i)
        if (g_data[i] != 100.0f + (float)i) ok = false;
    printf("  CPU 验证 g_data[0..1023]：%s\n", ok ? "✓" : "✗");

    // CPU 写，GPU 再读
    g_counter = 42;
    g_data[0] = -1.0f;
    fill_global_kernel<<<1, 1, 0, stream>>>(1);   // 只让线程 0 改 g_data[0]
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("  [CPU 写 g_data[0]=-1，GPU 重写后] g_data[0] = %.1f  g_counter = %d\n",
           g_data[0], g_counter);

    // 注意：不需要也不能 cudaFree(__managed__ 全局变量)。
    // __managed__ 是静态存储期，生命周期 = 整个程序，由运行时自动回收：
    //   程序退出 → CUDA Runtime 析构 Context → 驱动注销 UVM 映射 → OS 回收物理页
    // 与普通全局变量一致：int g_arr[1024] 也不能 free(g_arr)，否则未定义行为。
    //
    // 三种分配方式的释放对比：
    //   cudaMallocManaged       → 必须显式 cudaFree(ptr)
    //   cudaMallocFromPoolAsync → 必须显式 cudaFreeAsync(ptr, stream)
    //   __managed__ 全局变量    → 程序退出时自动释放，不需要也不能手动释放
}


// ═════════════════════════════
// 三种方式横向对比表
// ═════════════════════════════
static void print_comparison_table()
{
    section("三种分配方式横向对比");
    printf(
        "  ╔══════════════════════╦══════════════════╦══════════════════════╦══════════════════╗\n"
        "  ║ 维度                 ║ cudaMallocManaged ║ cudaMallocFromPool    ║ __managed__ 全局 ║\n"
        "  ╠══════════════════════╬══════════════════╬══════════════════════╬══════════════════╣\n"
        "  ║ 分配时机             ║ 同步（阻塞 CPU）  ║ 异步（插入 stream）  ║ 程序启动时       ║\n"
        "  ║ 动态大小             ║ ✓                ║ ✓                    ║ ✗（编译期常量）  ║\n"
        "  ║ 内存位置             ║ UVM（可迁移）     ║ 设备显存（固定）     ║ UVM（可迁移）    ║\n"
        "  ║ CPU 直接访问         ║ ✓                ║ ✗（须 cudaMemcpy）   ║ ✓               ║\n"
        "  ║ 显式 free            ║ cudaFree          ║ cudaFreeAsync        ║ 不需要（自动）   ║\n"
        "  ║ Advise/Prefetch      ║ ✓（Full UM）      ║ N/A（设备内存）      ║ ✓（Full UM）    ║\n"
        "  ║ 高频小分配性能       ║ 差（系统调用）    ║ 好（池复用）         ║ N/A（静态）      ║\n"
        "  ║ Limited UM 可用      ║ ✓                ║ ✓（设备内存无限制）  ║ ✓               ║\n"
        "  ║ CUDA 版本要求        ║ 全部版本          ║ 11.2+                ║ 全部版本         ║\n"
        "  ║ 适用场景             ║ CPU+GPU 共享大块  ║ GPU 高频动态分配     ║ 全局配置/计数器  ║\n"
        "  ╚══════════════════════╩══════════════════╩══════════════════════╩══════════════════╝\n"
    );
}


// ═════════════════════════════
// 系统能力检测 + 路由
// ═════════════════════════════
static void run_capability_demo(int dev)
{
    // ── 检测三个关键属性 ──────
    // UMCap cap{}：列表初始化触发值初始化，所有 int 成员归零。
    // 对比：UMCap cap; （默认初始化）→ 成员值不确定（垃圾值）。
    // 防御性写法：保证 cudaDeviceGetAttribute 覆写之前成员不含随机值。
    UMCap cap{};
    CUDA_CHECK(cudaDeviceGetAttribute(&cap.managed,
                                      cudaDevAttrManagedMemory,           dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&cap.concurrent,
                                      cudaDevAttrConcurrentManagedAccess, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&cap.pageable,
                                      cudaDevAttrPageableMemoryAccess,    dev));

    // ── 打印检测结果 ────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU %d：%s  (SM %d.%d)\n", dev, prop.name, prop.major, prop.minor);
    printf("  cudaDevAttrManagedMemory           = %d\n", cap.managed);
    printf("  cudaDevAttrConcurrentManagedAccess = %d\n", cap.concurrent);
    printf("  cudaDevAttrPageableMemoryAccess    = %d\n", cap.pageable);

    // ── 判断模式并路由 ────────
    //
    // 判断顺序：
    //   1. pageableMemoryAccess = 1 → HMM/ATS：普通 malloc 对 GPU 透明可见，
    //      不需要 cudaMallocManaged，但三种 managed 方式仍然有效。
    //   2. managed=1, concurrent=1  → Full UM：按需缺页迁移，Advise/Prefetch 有效。
    //   3. managed=1, concurrent=0  → Limited UM：GPU kernel 期间 CPU 不可访问，
    //      整体迁移，无 Advise/Prefetch。
    //   4. managed=0                → 不支持统一内存，需要显式 cudaMemcpy。
    //
    if (cap.pageable) {
        printf("  → 模式：HMM / ATS 系统（GPU 可直接访问 malloc 内存）\n\n");
        printf("  在 HMM 系统上，普通 malloc 即可供 GPU 使用，\n"
               "  但三种 managed 分配方式仍然有效（且可用 Advise/Prefetch 优化）。\n");

    } else if (cap.managed && cap.concurrent) {
        printf("  → 模式：Full UM（软件 UVM，按需缺页迁移，Advise/Prefetch 有效）\n\n");

    } else if (cap.managed) {
        printf("  → 模式：Limited UM（整体迁移，GPU 独占访问，无 Advise/Prefetch）\n\n");
        printf("  注意：kernel 运行期间 CPU 不得访问 managed 内存。\n"
               "        cudaMemAdvise / cudaMemPrefetchAsync 调用将返回错误，已跳过。\n\n");

    } else {
        printf("  → 模式：不支持统一内存（cudaMallocManaged 不可用）\n\n");
        printf("  回退方案：显式分配两份内存 + 手动 cudaMemcpy。\n");

        // 展示无 UM 时的回退路径
        constexpr int N    = 1 << 20;
        const size_t  size = N * sizeof(float);
        // malloc 在堆上分配，不初始化 → 垃圾值，需手动写入后才能使用。
        // 对比：calloc 分配并零初始化；new float[N]{} 值初始化为 0.0f。
        float* h_ptr = (float*)malloc(size);
        float* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, size));

        for (int i = 0; i < N; ++i) h_ptr[i] = (float)i;
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

        printf("  h_ptr[0..2] = %.1f  %.1f  %.1f（CPU 初始化后 memcpy 到 GPU）\n",
               h_ptr[0], h_ptr[1], h_ptr[2]);

        CUDA_CHECK(cudaFree(d_ptr));
        free(h_ptr);
        return;
    }

    // ── 创建 stream，演示三种分配方式 ────────
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    demo_malloc_managed(dev, stream, cap);
    demo_pool_managed(dev, stream);
    demo_managed_global(stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ──────────────────────────────────
// main
// ──────────────────────────────────
int main(void)
{
    int gpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&gpuCount));
    printf("检测到 %d 张 GPU\n\n", gpuCount);

    for (int dev = 0; dev < gpuCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        run_capability_demo(dev);
    }

    print_comparison_table();
    return 0;
}


// ──────────────────────────────────
// 实测输出（RTX 3060 Laptop / WSL2，Limited UM）：
// 检测到 1 张 GPU
//
// GPU 0：NVIDIA GeForce RTX 3060 Laptop GPU  (SM 8.6)
//   cudaDevAttrManagedMemory           = 1
//   cudaDevAttrConcurrentManagedAccess = 0
//   cudaDevAttrPageableMemoryAccess    = 0
//   → 模式：Limited UM（整体迁移，GPU 独占访问，无 Advise/Prefetch）
//
//   注意：kernel 运行期间 CPU 不得访问 managed 内存。
//         cudaMemAdvise / cudaMemPrefetchAsync 调用将返回错误，已跳过。
//
//
// ══════════════════════════════════════════
//   方式一：cudaMallocManaged
// ══════════════════════════════════════════
//
// [1a] cudaMallocManaged（cudaMemAttachGlobal，默认）
//   分配耗时（CPU）: 11.271 ms
//   [Limited UM] 跳过 Advise/Prefetch（不支持按需页迁移，迁移由运行时在 kernel 启动时整体完成）
//   CPU 验证：ptr_global[0..9] = [1.0 ... 10.0]  ✓
//
// [1b] cudaMallocManaged（cudaMemAttachHost + cudaStreamAttachMemAsync）
//   分配耗时（CPU）: 2.161 ms（初始仅 CPU 可访问）
//   CPU 验证：ptr_host[0..9]  = [2.0 ... 11.0]  ✓
//
// ══════════════════════════════════════════
//   方式二：cudaMallocFromPoolAsync（流有序设备内存池）
// ══════════════════════════════════════════
//   设备内存池创建成功（device 0，初始物理内存 0 字节，按需增长）
//   cudaMallocFromPoolAsync 入队耗时（CPU）: 6.120 ms
//   CPU 验证（通过 cudaMemcpy）：ptr_pool[0..9] = [3.0 ... 12.0]  ✓
//   注：池内存是设备内存（非 Unified Memory），CPU 不可直接访问
//   Pool  (cudaMallocFromPoolAsync × 10): 2.30 ms
//   Naive (cudaMallocManaged       × 10): 50.04 ms
//   加速比: 21.71x
//
// ══════════════════════════════════════════
//   方式三：__managed__ 全局变量
// ══════════════════════════════════════════
//   [初始] g_counter = 0，g_data[0] = 0.0（静态初始化值）
//   [GPU 写后，CPU 读] g_counter = 1
//   g_data[0..4] = 100.0  101.0  102.0  103.0  104.0
//   CPU 验证 g_data[0..1023]：✓
//   [CPU 写 g_data[0]=-1，GPU 重写后] g_data[0] = 100.0  g_counter = 43