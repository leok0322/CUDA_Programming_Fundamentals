/**
 * cuda_graph.cu
 *
 * CUDA Graph 详解与示例
 *
 * 编译：
 *   nvcc -arch=native -std=c++17 cuda_graph.cu -o cg
 *
 * 运行：
 *   ./cg
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ───────────────
// 错误检查宏
// ───────────────────
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


// ═════════════
// 背景：为什么需要 CUDA Graph
//
// 普通 stream 启动方式的问题：
//   每次 kernel<<<>>> 都是一次 CPU→GPU 的 API 调用：
//     CPU 准备参数 → 写入命令缓冲区 → GPU 驱动解析 → 调度执行
//   单次开销约 5~10 μs（CPU 侧），对于执行时间极短的 kernel（<10 μs），
//   启动开销本身就占总时间的主要部分，GPU 大量时间处于空闲等待状态。
//
//   迭代场景（同一组 kernel 重复执行 N 次）：
//     每次迭代都要重新走一遍 API 调用流程，CPU 成为瓶颈。
//
// CUDA Graph 的解决思路：
//   将一组操作（kernel、memcpy、memset 等）及其依赖关系
//   描述为一个 DAG（有向无环图），一次性提交给驱动。
//   后续重复执行时，只需一次 cudaGraphLaunch，驱动直接按图执行，
//   跳过重复的参数准备和命令解析，大幅降低 CPU 开销。
//
// 两种构建 Graph 的方式：
//   ① Stream Capture（流捕获）：在 stream 上"录制"操作序列，自动生成 DAG
//   ② Explicit API（显式构建）：手动添加节点和边，精确控制依赖关系
//
// 核心数据结构：
//   cudaGraph_t      → 图定义（DAG，未编译，不可直接执行）
//   cudaGraphExec_t  → 可执行图实例（编译后的图，可反复 Launch）
// ═════════════════════════


static const int N       = 1 << 20;   // 1M 元素
static const int THREADS = 256;
static const int BLOCKS  = (N + THREADS - 1) / THREADS;
static const int NKERNEL = 10;        // 每次迭代执行的 kernel 数
static const int NSTEP   = 100;       // 迭代次数


// kernel①：out[i] = in[i] + val（用于示例3，读 in 写 out）
__global__ void addKernel(float *out, const float *in, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + val;
}

// kernel②：buf[i] += val（in-place，用于示例1/2的串行累加）
// 示例1/2 中 NKERNEL 个 kernel 串行，每次读写同一块显存：
//   buf 初始为 0，NKERNEL 次 += 1.0f 后 = NKERNEL × 1.0f
// 若用 addKernel（读 in 写 out），每次 kernel 都从固定的 in 读，
// 结果始终是 0 + 1 = 1.0f，无法体现 NKERNEL 个串行 kernel 的累积效果。
__global__ void addInPlace(float *buf, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] += val;
}


// ════════════
// 示例 1：普通 stream 启动（对照组）
//
// 每次迭代重复提交 NKERNEL 次 kernel API 调用。
// NSTEP × NKERNEL 次 API 调用，CPU 开销随迭代次数线性增长。
// ═══════════
void demo_normal_launch(float *d_out, const float *d_in, cudaStream_t stream)
{
    printf("\n─── 示例 1：普通 stream 启动（对照组）───\n");
    printf("  每次迭代提交 %d 次 kernel API 调用，共 %d 次迭代\n",
           NKERNEL, NSTEP);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int istep = 0; istep < NSTEP; istep++) {
        // 每次迭代前重置为 0，使 NKERNEL 次串行累加从 0 开始
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, N * sizeof(float), stream));
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
            // 每次都是独立的 API 调用：CPU 准备参数 → 写命令缓冲区 → 驱动解析
            // in-place 累加：+1 共 NKERNEL 次，最终 = NKERNEL × 1.0f
            addInPlace<<<BLOCKS, THREADS, 0, stream>>>(d_out, 1.0f, N);
            CUDA_CHECK(cudaGetLastError());
        }
        // CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("  耗时：%.3f ms（%d 次迭代 × %d 次 kernel API 调用）\n",
           ms, NSTEP, NKERNEL);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}


// ═══════════
// 示例 2：CUDA Graph（Stream Capture 方式）
//
// 流捕获（Stream Capture）工作原理：
//
//   cudaStreamBeginCapture(stream, mode)
//     · 将 stream 切换为"捕获模式"
//     · 此后提交到该 stream 的所有操作不会真正执行，
//       而是被记录为 DAG 节点
//     · mode = cudaStreamCaptureModeGlobal：
//         捕获期间，任何其他线程对该 context 的操作
//         若与被捕获 stream 有依赖（通过 event），也会被捕获进图中
//
//   cudaStreamEndCapture(stream, &graph)
//     · 结束捕获，stream 恢复正常模式
//     · 捕获的操作序列写入 graph（cudaGraph_t）
//     · graph 是 DAG 定义，不可直接执行
//
//   cudaGraphInstantiate(&instance, graph, NULL, NULL, 0)
//     · 编译 graph，生成可执行图实例（cudaGraphExec_t）
//     · 驱动在此阶段做优化：合并操作、预分配资源、优化调度顺序
//     · 编译有一定开销，但只需做一次
//
//   cudaGraphLaunch(instance, stream)
//     · 单次 API 调用，将整张图的所有节点提交给 GPU 执行
//     · 驱动跳过重复的参数解析，直接按预编译的执行计划调度
//     · 无论图中有多少个节点，CPU 开销恒定（一次 API 调用）
//
// 适用场景：
//   · 同一组操作需要重复执行多次（迭代计算、推理循环等）
//   · kernel 执行时间短，启动开销占比大
//   · 操作依赖关系固定，不需要每次迭代动态调整
// ══════════
void demo_graph_launch(float *d_out, const float *d_in, cudaStream_t stream)
{
    printf("\n─── 示例 2：CUDA Graph（Stream Capture）───\n");
    printf("  第 1 次迭代：捕获图 + 编译，后续 %d 次迭代只需 cudaGraphLaunch\n",
           NSTEP - 1);

    // cudaGraph_t 和 cudaGraphExec_t 都是不透明句柄（opaque handle）：
    //
    //   不透明句柄模式：
    //     用户代码持有的是指向内部结构体的指针，
    //     内部结构体定义隐藏在 CUDA 驱动内部，用户无法直接访问其成员。
    //     只能通过 CUDA API 操作它，不能解引用或读写内部字段。
    //
    //   在 driver_types.h 中的实际定义：
    //     typedef struct CUgraph_st*     cudaGraph_t;
    //     typedef struct CUgraphExec_st* cudaGraphExec_t;
    //     typedef struct CUgraphNode_st* cudaGraphNode_t;
    //
    //     struct CUgraph_st    的定义在驱动内部，头文件中只有前向声明（forward declaration），
    //     用户代码看不到成员，sizeof(cudaGraph_t) = sizeof(指针) = 8 字节（64位系统）。
    //
    //   同类型的其他不透明句柄：
    //     cudaStream_t  = struct CUstream_st*
    //     cudaEvent_t   = struct CUevent_st*
    //
    //   为什么用不透明句柄：
    //     · 驱动可以在不改变 API 的情况下修改内部实现
    //     · 防止用户直接操作内部状态导致驱动崩溃
    //     · 跨平台：不同 GPU 架构的内部结构可以不同，API 保持一致
    //
    // cudaGraph_t 和 cudaGraphExec_t 是指针类型（内置类型），不是类类型：
    //     指针与 int、float 同属内置类型，没有构造函数。
    //     C++ 类型分为内置类型（int/float/指针）和类类型（class/struct，有构造函数）。
    //     类类型局部变量声明时调用默认构造函数自动初始化（如 std::string → 空字符串）；
    //     内置类型局部变量声明时不初始化，值是栈上残留的垃圾值。
    //     注意：struct* 是指针（内置类型），struct 本身才是类类型，两者不同。

    //     全局/静态变量例外：无论内置类型还是类类型，全局变量都零初始化
    //     全局 cudaGraphExec_t g_inst;  → nullptr（零初始化）
    //     局部 cudaGraphExec_t inst;    → 垃圾值（不初始化）

    //     本文件中不加 {} 安全的原因：声明后立刻被 API 写入，垃圾值没有被读取的机会。
    //     若声明和赋值之间有条件分支可能跳过赋值，则应加 {} 初始化为 nullptr。

    // cudaGraph_t：图定义（DAG），记录节点和依赖关系，不可直接执行
    cudaGraph_t graph;

    // cudaGraphExec_t：编译后的可执行图实例，可反复 Launch
    cudaGraphExec_t instance;

    bool graphCreated = false;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int istep = 0; istep < NSTEP; istep++) {

        if (!graphCreated) {
            // ── 第一次迭代：捕获图并编译 ───────────
            //
            // cudaStreamBeginCapture：将 stream 切换为捕获模式
            //
            // 函数签名：
            //   cudaError_t cudaStreamBeginCapture(
            //     cudaStream_t             stream,  // 要开始捕获的 stream
            //     cudaStreamCaptureMode    mode     // 捕获模式
            //   );
            //
            // 捕获是针对特定 stream 的：
            //   BeginCapture 只将 stream（此处为 stream）切换为录制状态，
            //   其他 stream 不受影响，照常在 GPU 上执行。
            //   提交到被捕获 stream 的操作不会真正执行，而是被记录为 DAG 节点。
            //
            //   Global 模式下其他 stream 被"拉入捕获"的含义：
            //     并非其他 stream 也变成录制模式，而是它们通过 event 依赖
            //     与被捕获 stream 关联，驱动将其操作纳入图的 DAG 作为节点。
            //     其他 stream 本身仍是普通 stream，捕获主体始终只有这一个 stream。
            //
            //   EndCapture 也只针对同一个 stream：
            //     cudaStreamEndCapture(stream, &graph) 让 stream 退出录制模式，
            //     其他被"拉入"的 stream 从未进入录制模式，无需 EndCapture。
            //
            // mode 的核心问题：
            //   捕获期间，其他线程的操作若通过 event 与被捕获 stream 产生依赖，
            //   驱动应该怎么处理？三种模式给出不同答案：
            //
            //   cudaStreamCaptureModeGlobal（最严格，本例使用）：
            //
            //     场景举例（多线程协同捕获）：
            //       线程A（主线程）：
            //         cudaStreamBeginCapture(streamA, ModeGlobal);
            //         kernel1<<<..., streamA>>>();          // 节点1 入图
            //         cudaEventRecord(ev, streamA);         // ev 标记 streamA 当前位置
            //         // 通知线程B
            //
            //       线程B（工作线程）：
            //         cudaStreamWaitEvent(streamB, ev, 0);  // streamB 等待 ev
            //         kernel2<<<..., streamB>>>();          // Global 模式：
            //                                               // streamB 被自动拉入捕获，
            //                                               // kernel2 成为图的节点2
            //         cudaEventRecord(ev2, streamB);
            //
            //       线程A：
            //         cudaStreamWaitEvent(streamA, ev2, 0); // 等待线程B的节点
            //         kernel3<<<..., streamA>>>();          // 节点3，依赖节点2
            //         cudaStreamEndCapture(streamA, &graph);
            //         // 最终 graph：node1 → node2(线程B) → node3，跨线程依赖完整
            //
            //     规则：Global 模式下，其他线程若对 context 做"语义上矛盾"的操作
            //           会报错 cudaErrorStreamCaptureUnsupported。
            //
            //       为什么 cudaStreamSynchronize 不能被捕获为图节点，而是直接报错：
            //
            //         图的节点 = GPU 侧操作（kernel、memcpy、event record/wait 等），
            //         每个节点都有对应的 GPU 硬件指令。
            //         cudaStreamSynchronize 是纯 CPU 侧阻塞调用，
            //         没有任何对应的 GPU 硬件指令，无法表达为图节点。
            //
            //         若强行捕获为节点，图在 GPU 上执行到该节点时：
            //           GPU：等待 CPU 确认完成
            //           CPU：等待 GPU 执行完图（GraphLaunch 后的 Sync）
            //           → GPU 等 CPU，CPU 等 GPU → 死锁
            //
            //         图内 GPU 操作的顺序依赖已由 DAG 的有向边表达，
            //         不需要在节点间插入"CPU 确认节点"。
            //         CPU 等图全部完成，只需在 GraphLaunch 之后调用一次
            //         cudaStreamSynchronize，不应出现在图内部。
            //         因此捕获期间遇到 cudaStreamSynchronize 直接报错。
            //
            //       即使对非捕获 stream 也可能报错（Global 模式的保守策略）：
            //
            //         场景：捕获前，线程B 已向 streamB 提交了操作，
            //               并通过 event 与 streamA 建立了依赖：
            //
            //           捕获前：
            //             线程B：kernelX<<<..., streamB>>>()
            //                    cudaEventRecord(ev, streamB)   // ev 标记 streamB
            //
            //             线程A：cudaStreamWaitEvent(streamA, ev, 0)
            //                    // streamA 等待 ev，即等待 kernelX 完成
            //                    BeginCapture(streamA, Global)
            //                    kernel1<<<..., streamA>>>()    // 开始录制
            //
            //           此时 streamA 虽然进入录制模式，但它仍"记得"捕获前
            //           有一个对 ev 的等待依赖（kernelX → ev → streamA）。
            //
            //           cudaEvent 状态机（理解后续分析的基础）：
            //             cudaEventCreate(ev)
            //               → 初始状态（未记录）
            //             cudaEventRecord(ev, stream)          ← CPU API 调用瞬间
            //               → ev 立即重置为 pending            ← CPU 侧同步完成
            //               → GPU stream 中插入标记            ← GPU 侧异步执行
            //             GPU 执行到该标记
            //               → ev 变为 complete
            //             再次 cudaEventRecord(ev, stream)     ← CPU 调用瞬间
            //               → ev 再次重置为 pending（状态重置，不只是覆盖时间戳）
            //
            //           为什么循环执行图不会有 ev 状态残留问题：
            //             每次循环迭代开始前，CPU 重新调用 cudaEventRecord(ev, streamB)
            //             → ev 立即变为 pending（CPU 侧重置）
            //             → GPU 执行 kernelX 完成后，ev 变为 complete
            //             → 图的 WaitEvent(ev) 看到 complete → 放行
            //             每次迭代都有 Record → 每次都从 pending 开始，依赖始终有效。
            //
            //           捕获中：
            //             线程B：cudaStreamSynchronize(streamB)
            //                    // ← CUDA runtime 在此 API 调用时立即报错
            //                    //
            //                    // 报错原因（与 ev 状态无关）：
            //                    //
            //                    // Global 模式规则：
            //                    //   捕获开始后，驱动追踪所有通过 event 与被捕获 stream
            //                    //   有依赖关系的 stream，将它们纳入"捕获范围"。
            //                    //
            //                    //   streamA 依赖 ev，ev 来自 streamB
            //                    //   → streamB 被拉入捕获范围
            //                    //   → 捕获期间任何线程对 streamB 的操作均非法
            //                    //   → 线程B 调用 Sync(streamB) → 驱动检测到 → 报错
            //                    //
            //                    // 错误类型：cudaErrorStreamCaptureUnsupported
            //                    // 报错时机：API 调用时（捕获期间），不是图执行时
            //
            //         驱动的报错充要条件：
            //           其他线程操作的 stream 与被捕获 stream 之间
            //           存在可追踪的依赖链（通过 event 建立），
            //           且该操作可能影响这条依赖链的状态，才报错。
            //
            //           若 streamB 与 streamA 之间从未有过任何 event 关联：
            //             线程A：BeginCapture(streamA, Global)
            //             线程B：cudaStreamSynchronize(streamB)
            //                    // 驱动检查：streamB 与 streamA 有依赖链吗？
            //                    // → 没有 → 不报错，正常执行
            //
            //           Global 模式不是"捕获期间禁止一切其他线程操作"，
            //           而是"禁止操作与捕获有依赖关联的 stream"。
            //           强制要求：所有跨线程依赖必须在捕获期间通过显式 event 声明，
            //           驱动才能将相关 stream 正确拉入捕获，保证图的完整性。
            //
            //     适用：需要多线程协同构建同一张图，依赖关系跨线程的场景。
            //
            //   cudaStreamCaptureModeThreadLocal：
            //
            //     场景举例（单线程捕获，后台线程独立工作）：
            //       线程A（主线程）：
            //         cudaStreamBeginCapture(streamA, ModeThreadLocal);
            //         kernel1<<<..., streamA>>>();          // 节点1 入图
            //         cudaEventRecord(ev, streamA);
            //         // 通知线程B
            //
            //       线程B（后台线程，与捕获无关的独立工作）：
            //         cudaStreamWaitEvent(streamB, ev, 0);  // ThreadLocal 模式：
            //         kernel2<<<..., streamB>>>();          // streamB 不会被拉入捕获，
            //                                               // 此操作正常执行（不入图）
            //                                               // 线程B 对捕获完全透明
            //
            //       线程A：
            //         kernel3<<<..., streamA>>>();          // 节点2（不依赖线程B）
            //         cudaStreamEndCapture(streamA, &graph);
            //         // 最终 graph：node1 → node2，只含线程A的操作
            //         // 线程B 的 kernel2 在图外正常执行，互不干扰
            //
            //     规则：
            //       · 线程B 操作非捕获 stream（如 cudaStreamSynchronize(streamB)）
            //         → 不报错，正常执行；驱动只监视调用 BeginCapture 的线程A
            //       · 线程B 通过 event 与 streamA 建立依赖
            //         → 忽略，不被捕获，不报错；线程B 的操作在图外独立执行
            //       · 线程B 直接操作被捕获的 streamA 本身
            //         → 报错（任何模式下直接操作被捕获 stream 本身都报错）
            //
            //     与 Global 的关键区别：
            //       Global 模式：线程B 的 cudaStreamSynchronize(streamB)
            //                    · 驱动检测到 streamB 与 streamA 之间存在依赖链
            //                      → streamB 被拉入捕获范围
            //                      → Sync(streamB) 是对捕获范围内 stream 的非法操作
            //                      → 报错（cudaErrorStreamCaptureUnsupported）
            //                    · 驱动未检测到任何依赖链
            //                      → streamB 不在捕获范围内 → 不报错，正常执行
            //                    （"也可能报错"的前提是依赖链存在且被驱动检测到）
            //       ThreadLocal：线程B 的 cudaStreamSynchronize(streamB) 不报错
            //                    （驱动完全不关心线程B，由程序员自己保证隔离性）
            //
            //     代价：若线程B 的操作实际上是图的输入/输出，
            //           依赖关系不完整，图执行结果未定义；驱动不检查，后果自负。
            //     适用：单线程捕获，后台线程有独立 GPU 工作且不需要进入图的场景。
            //
            //   cudaStreamCaptureModeRelaxed（最宽松）：
            //     "非同步操作"指的是：其他线程对捕获相关资源（如 event）
            //     的操作不会被限制，即使这些操作在语义上是"不安全"的也不报错。
            //
            //     具体场景举例：
            //       线程B 调用 cudaEventRecord(ev, streamB)：
            //         ev 曾被 streamA 通过 WaitEvent 依赖，但此操作是对 streamB 的，
            //         不是直接操作被捕获的 streamA 本身。
            //
            //         · Global 模式    → 报错
            //             驱动检测到依赖链 streamB→ev→streamA（捕获中），
            //             streamB 被拉入捕获范围；
            //             线程B 对 streamB 的 EventRecord 是对捕获范围内 stream 的非法操作
            //             → 报错（cudaErrorStreamCaptureUnsupported）
            //         · ThreadLocal 模式 → 不报错，且不入图
            //             驱动只捕获调用 BeginCapture 的线程A 的操作，
            //             线程B 的 EventRecord(ev, streamB) 不会被录入图，
            //             图里只有线程A 的节点，执行时线程B 的操作对图无影响。
            //             风险：程序员需自行保证线程B 的工作不影响图的数据正确性，
            //             驱动不检查，后果自负。
            //         · Relaxed 模式  → 不报错（与 ThreadLocal 相同，驱动不检查）
            //
            //     Relaxed 与 ThreadLocal 的区别：
            //       "直接操作被捕获 stream 本身"指的是：
            //       线程B 将 streamA 作为参数传给会改变其状态的 API，例如：
            //         cudaStreamSynchronize(streamA)  // 等待 streamA
            //         cudaMemcpyAsync(..., streamA)   // 向 streamA 提交操作
            //         cudaEventRecord(ev, streamA)    // 在 streamA 上记录 event
            //
            //       ThreadLocal 模式下，线程B 这样做会报错：
            //         线程A：BeginCapture(streamA, ThreadLocal)
            //                kernel1<<<..., streamA>>>()   // 录制中
            //
            //         线程B：cudaMemcpyAsync(dst, src, n, streamA)
            //                → 报错：streamA 正在被线程A 捕获，
            //                  线程B 试图向其提交真实操作，破坏捕获状态
            //                  （streamA 处于录制模式，不接受真实提交）
            //
            //       Relaxed 模式下，线程B 同样的操作不报错：
            //         线程A：BeginCapture(streamA, Relaxed)
            //                kernel1<<<..., streamA>>>()   // 录制中
            //
            //         线程B：cudaMemcpyAsync(dst, src, n, streamA)
            //                → 不报错，驱动允许此操作
            //                  但 memcpy 提交到了正在录制的 streamA，
            //                  它究竟被录制进图还是真实执行，行为未定义
            //                  图的节点顺序可能被破坏，结果不可预期
            //
            //     三种模式报错边界总结：
            //       操作类型                          Global  ThreadLocal  Relaxed
            //       线程B Sync(非捕获 streamB)         报错    不报错       不报错
            //       线程B EventRecord(捕获关联 event)  报错    不报错       不报错
            //       线程B 直接操作被捕获 streamA       报错    报错         不报错
            //
            //     代价：驱动完全不保护捕获完整性，程序员需自行确保安全，慎用。
            //     适用：明确知道跨线程操作不影响图语义的特殊场景。
            CUDA_CHECK(cudaStreamBeginCapture(stream,
                                              cudaStreamCaptureModeGlobal));

            // memset 也被捕获进图，作为第一个节点，每次 Launch 都会重置
            CUDA_CHECK(cudaMemsetAsync(d_out, 0, N * sizeof(float), stream));

            for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
                // 这些 kernel 启动不会真正执行，被记录为 DAG 节点
                // 同一 stream 内的操作自动形成串行依赖链：
                //   memset → node0 → node1 → ... → node(NKERNEL-1)
                addInPlace<<<BLOCKS, THREADS, 0, stream>>>(d_out, 1.0f, N);
                CUDA_CHECK(cudaGetLastError());
            }

            // cudaStreamEndCapture：结束捕获，将录制结果写入 graph
            //
            // 函数签名：
            //   cudaError_t cudaStreamEndCapture(
            //     cudaStream_t   stream,   // 要结束捕获的 stream（必须与 Begin 一致）
            //     cudaGraph_t   *pGraph    // 输出：生成的图定义
            //   );
            //
            // 调用后：
            //   · stream 恢复正常模式，后续操作正常执行
            //   · *pGraph 包含所有被捕获操作构成的 DAG
            //     同一 stream 内的操作自动形成串行依赖链
            //   · graph 是未编译的图定义，不可直接执行，
            //     必须经过 cudaGraphInstantiate 编译后才能 Launch
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

            // cudaGraphInstantiate：编译 graph，生成可执行图实例
            //
            // 函数签名：
            //   cudaError_t cudaGraphInstantiate(
            //     cudaGraphExec_t  *pGraphExec,  // 输出：编译后的可执行实例
            //     cudaGraph_t       graph,        // 输入：图定义（DAG）
            //     cudaGraphNode_t  *pErrorNode,  // 输出：出错的节点（可传 NULL）
            //     char             *pLogBuffer,  // 输出：错误日志缓冲区（可传 NULL）
            //     size_t            bufferSize   // 日志缓冲区大小（传 NULL 时填 0）
            //   );
            //
            // 编译阶段驱动做的事：
            //   · 验证 DAG 中无环（保证可执行性）
            //   · 优化节点调度顺序
            //   · 预分配执行所需资源
            //   · 生成可直接提交给 GPU 的执行计划
            //
            // instance 与 graph 独立：
            //   · instance 创建后，graph 可以安全销毁
            //   · 修改 graph 不影响已有 instance
            //   · 若需要用新 graph 更新 instance，调用 cudaGraphExecUpdate
            CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

            // graph 定义在 instance 创建后即可销毁，instance 独立存在
            // cudaGraphDestroy：释放 graph 定义占用的内存，不影响 instance
            CUDA_CHECK(cudaGraphDestroy(graph));

            graphCreated = true;
            printf("  [istep=0] 图已捕获并编译（%d 个节点）\n", NKERNEL);
        }

        // ── 每次迭代：单次API调用执行整张图 ───────
        //
        // cudaGraphLaunch：将可执行图实例提交到 stream 执行
        //
        // 函数签名：
        //   cudaError_t cudaGraphLaunch(
        //     cudaGraphExec_t   graphExec,  // 要执行的图实例
        //     cudaStream_t      stream      // 提交到哪个 stream
        //   );
        //
        // 与普通 kernel 启动的区别：
        //   普通启动：每个 kernel 一次 API 调用，驱动每次重新解析参数
        //   图启动  ：一次 API 调用提交图中所有节点，驱动直接按预编译计划执行
        //   无论图中有多少节点，CPU 开销恒定为一次 API 调用的代价
        //
        // stream 的作用：
        //   图本身不属于任何 stream，Launch 时指定 stream，
        //   图的执行在该 stream 中排队，遵循 stream 的顺序语义
        CUDA_CHECK(cudaGraphLaunch(instance, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("  耗时：%.3f ms（%d 次迭代，每次 1 次 cudaGraphLaunch）\n",
           ms, NSTEP);

    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}


// ═════════════════
// 示例 3：显式构建 Graph（不用 Stream Capture）
//
// Stream Capture 自动从 stream 操作序列推导依赖关系；
// 显式构建则手动添加节点和边，适合依赖关系复杂或需要精确控制的场景。
//
// 本示例构建如下 DAG：
//
//   [kernelA] ──┐
//               ├──→ [kernelC]
//   [kernelB] ──┘
//
//   kernelA 和 kernelB 并行执行，kernelC 等两者都完成后执行。
//   用普通 stream 难以简洁表达此结构，Graph 显式构建更直观。
// ══════════════════════
void demo_explicit_graph(float *d_outA, float *d_outB, float *d_outC,
                         const float *d_in, cudaStream_t stream)
{
    printf("\n─── 示例 3：显式构建 Graph（A‖B → C）───\n");

    // cudaGraphCreate：创建空图（无节点、无边）
    //
    // 函数签名：
    //   cudaError_t cudaGraphCreate(
    //     cudaGraph_t  *pGraph,  // 输出：新创建的空图句柄
    //     unsigned int  flags   // 保留字段，必须传 0
    //   );
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // ── kernelA 节点 ────────
    //
    // cudaKernelNodeParams：描述 kernel 节点所有参数的结构体
    //
    //   struct cudaKernelNodeParams {
    //     void                  *func;           // kernel 函数指针（void* 形式）
    //     dim3                   gridDim;        // grid 维度
    //     dim3                   blockDim;       // block 维度
    //     unsigned int           sharedMemBytes; // 动态 shared memory 大小（字节）
    //     void                 **kernelParams;   // kernel 参数地址数组
    //                                            // 每个元素是对应参数的指针
    //                                            // 与 C 版 cudaLaunchKernel 的 args 相同
    //     void                 **extra;          // 额外参数，通常 NULL
    //   };
    float valA = 1.0f;
    int   n    = N;

    // argsA：kernel 参数的地址数组（void* 数组），每个元素是对应参数的地址。
    //   驱动不需要在运行时还原 C++ 类型，类型信息在编译期已编码进 kernel 元数据：
    //     nvcc 编译 addKernel 时，将参数布局写入 PTX/SASS 元数据：
    //       参数0：float*，大小 8 字节
    //       参数1：const float*，大小 8 字节
    //       参数2：float，大小 4 字节
    //       参数3：int，大小 4 字节
    //   驱动启动 kernel 时，按元数据逐参数从 argsA[i] 地址读取对应字节数，
    //   填入 GPU 参数寄存器，不使用也不需要 C++ 类型信息：
    //     argsA[0]=&d_outA → 读 8 字节 → GPU 寄存器0（float*）
    //     argsA[1]=&d_in   → 读 8 字节 → GPU 寄存器1（const float*）
    //     argsA[2]=&valA   → 读 4 字节 → GPU 寄存器2（float）
    //     argsA[3]=&n      → 读 4 字节 → GPU 寄存器3（int）
    //   代价：传错类型不报错，驱动按元数据大小读字节，只产生错误结果。
    //   这是 C 版接口的固有缺陷，cudaLaunchKernelEx 的 C++ 模板版本
    //   （ExpTypes/ActTypes）通过编译期类型检查弥补了这一问题。
    void *argsA[] = { &d_outA, &d_in, &valA, &n };

    // = {} 并非必须（下方已逐一赋值所有已知字段），但属于防御性编程惯用法：
    //   cudaKernelNodeParams 的字段可能随 CUDA 版本增加，
    //   = {} 确保所有未显式赋值的字段（包括未来新增字段）都是安全的零值，
    //   避免驱动读到垃圾值导致未定义行为。代价为零（编译器优化掉多余清零）。
    cudaKernelNodeParams paramA = {};

    // (void*)addKernel：将函数指针强转为 void*
    //   func 字段类型是 void*（C 接口无法表达具体函数签名）。
    //   驱动不通过 func 指针直接调用函数，而是将其作为 key
    //   在已注册的 kernel 列表中查找对应的 PTX/SASS 二进制及参数布局元数据，
    //   再结合 argsA 按元数据读取参数字节完成 kernel 启动。
    paramA.func            = (void *)addKernel;
    paramA.gridDim         = dim3(BLOCKS);
    paramA.blockDim        = dim3(THREADS);
    paramA.sharedMemBytes  = 0;
    paramA.kernelParams    = argsA;
    // extra 字段类型是 void**（指向 void* 数组的指针）：
    //   extra 是传递 kernel 参数的另一种机制，与 kernelParams 互斥，只能用其一。
    //   extra 指向一个键值对数组，每个元素交替存放预定义 key 常量和对应 value 指针：
    //     void *extra[] = {
    //         CU_LAUNCH_PARAM_BUFFER_POINTER, args_buffer,  // key → 参数缓冲区地址
    //         CU_LAUNCH_PARAM_BUFFER_SIZE,    &buf_size,    // key → 缓冲区大小
    //         CU_LAUNCH_PARAM_END                           // 终止标记
    //     };
    //   数组每个元素是 void*（key 常量或 value 指针），故 extra 本身是 void**。
    //   用 kernelParams 时 extra 必须为 NULL，用 extra 时 kernelParams 必须为 NULL。
    //
    // NULL vs nullptr：
    //   两者在此处效果相同（都表示空指针）。
    //   NULL 是 C/C++ 通用写法（定义为整数 0 或 (void*)0），沿用 C API 惯例。
    //   nullptr 是 C++11 专属，类型为 std::nullptr_t，只能转为指针类型，
    //   避免 NULL 作为整数 0 导致的函数重载歧义，现代 C++ 中更安全。
    paramA.extra           = NULL;

    // cudaGraphAddKernelNode：向图中添加一个 kernel 节点
    //
    // 函数签名：
    //   cudaError_t cudaGraphAddKernelNode(
    //     cudaGraphNode_t            *pGraphNode,   // 输出：新节点句柄
    //     cudaGraph_t                 graph,         // 目标图
    //     const cudaGraphNode_t      *pDependencies, // 前驱节点数组（NULL = 无前驱）
    //     size_t                      numDependencies,// 前驱节点数量
    //     const cudaKernelNodeParams *pNodeParams    // kernel 参数
    //   );
    //
    // nodeA 无前驱（pDependencies=NULL, numDependencies=0）→ 入口节点，可立即执行
    // pDependencies 类型是 const cudaGraphNode_t*（指针类型），
    // NULL 和 nullptr 效果完全相同，均表示空指针。
    // NULL 是沿用 C API 惯例；现代 C++ 中 nullptr 更类型安全，两者可互换。
    cudaGraphNode_t nodeA;
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeA, graph, NULL, 0, &paramA));

    // ── kernelB 节点（与 A 并行，同样无前驱）──────────
    float valB = 2.0f;
    void *argsB[] = { &d_outB, &d_in, &valB, &n };

    cudaKernelNodeParams paramB {};
    paramB.func           = (void *)addKernel;
    paramB.gridDim        = dim3(BLOCKS);
    paramB.blockDim       = dim3(THREADS);
    paramB.sharedMemBytes = 0;
    paramB.kernelParams   = argsB;
    paramB.extra          = NULL;

    cudaGraphNode_t nodeB;
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeB, graph, NULL, 0, &paramB));

    // ── kernelC 节点（依赖 A 和 B，等两者完成后执行）──────
    float valC = 3.0f;
    void *argsC[] = { &d_outC, &d_in, &valC, &n };

    cudaKernelNodeParams paramC = {};
    paramC.func           = (void *)addKernel;
    paramC.gridDim        = dim3(BLOCKS);
    paramC.blockDim       = dim3(THREADS);
    paramC.sharedMemBytes = 0;
    paramC.kernelParams   = argsC;
    paramC.extra          = NULL;

    // nodeC 的前驱 = {nodeA, nodeB}：
    //   cudaGraphAddKernelNode 根据 deps 数组自动插入 A→C 和 B→C 两条依赖边，
    //   运行时保证 nodeA 和 nodeB 都完成后，nodeC 才开始执行。
    cudaGraphNode_t deps[] = { nodeA, nodeB };
    cudaGraphNode_t nodeC;
    CUDA_CHECK(cudaGraphAddKernelNode(&nodeC, graph, deps, 2, &paramC));

    // cudaGraphExecDestroy / cudaGraphDestroy：
    //   cudaGraphExecDestroy(instance)：释放编译后的可执行实例
    //   cudaGraphDestroy(graph)        ：释放图定义
    //   两者独立，销毁顺序无要求
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    CUDA_CHECK(cudaGraphLaunch(instance, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("  [PASS] A(+1.0) 和 B(+2.0) 并行，C(+3.0) 等待 A‖B 完成后执行\n");

    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
}


// ══════════
// 验证结果
// ═══════════
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

    float *d_in, *d_out, *d_outA, *d_outB, *d_outC;
    CUDA_CHECK(cudaMalloc(&d_in,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outC, N * sizeof(float)));

    // 初始化输入为 0.0f
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 示例 1：普通 stream 启动
    demo_normal_launch(d_out, d_in, stream);
    // NKERNEL 次累加，每次 +1.0f，最终 = NKERNEL × 1.0f = 10.0f
    verify(d_out, (float)NKERNEL, N, "normal launch out");

    // 重置输出
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    // 示例 2：CUDA Graph（Stream Capture）
    demo_graph_launch(d_out, d_in, stream);
    verify(d_out, (float)NKERNEL, N, "graph launch out");

    // 示例 3：显式构建 Graph
    demo_explicit_graph(d_outA, d_outB, d_outC, d_in, stream);
    verify(d_outA, 1.0f, N, "explicit graph outA");
    verify(d_outB, 2.0f, N, "explicit graph outB");
    verify(d_outC, 3.0f, N, "explicit graph outC");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_outA));
    CUDA_CHECK(cudaFree(d_outB));
    CUDA_CHECK(cudaFree(d_outC));

    printf("\n全部示例完成。\n");
    return 0;
}
