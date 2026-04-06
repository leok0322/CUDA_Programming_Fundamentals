/**
 * cuda_host_func_data.cu
 *
 * 详细解析 cudaLaunchHostFunc 的 void *data 参数（type-erased pointer）：
 *
 *   void *data 是类型擦除指针（type-erased pointer）：
 *     - 传入时：任意结构体的地址被隐式转换为 void*，类型信息丢失
 *     - 传出时：在 hostFunc 内用 static_cast<T*> 还原类型
 *     - 驱动只负责原样传递，运行时不知道它指向什么
 *
 * 示例：
 *   A. 基础 type-erased pointer 用法（结构体定义、传入、还原）
 *   B. 多字段结构体（results、count、ready flag）
 *   C. 生命周期管理（stack vs heap，何时会出错）
 *   D. 多个 stream、多个 callback，各自携带独立数据
 *   E. 结合 semaphore 的完整流程（不在 hostFunc 内调用 CUDA API）
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 -std=c++20 cuda_host_func_data.cu -o cuda_host_func_data
 *
 * 运行：
 *   ./cuda_host_func_data
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <semaphore>   // C++20
#include <thread>
#include <cassert>

// ─────────────────
// 错误检查宏
// ────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",                 \
                    __FILE__, __LINE__,                                     \
                    cudaGetErrorName(err), cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void fillKernel(float *out, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}

static const int    N    = 1 << 18;
static const size_t SIZE = N * sizeof(float);


// ══════════════════════
// void* 类型擦除机制解析
// ═════════════════════
//
//   cudaLaunchHostFunc 签名：
//
//     cudaError_t cudaLaunchHostFunc(
//         cudaStream_t  stream,    // 目标 stream
//         void (*func)(void *),    // 函数指针，签名固定为 void(void*)
//         void         *data       // 传给 func 的任意数据
//     );
//
//   void *data 的含义：
//
//     void* 是 C/C++ 的通用指针类型，可以指向任意类型的数据，
//     但编译器和运行时不记录它"原来是什么类型"——类型信息被擦除了。
//
//     擦除过程（传入时）：
//       CallbackData *cbData = new CallbackData{...};
//       cudaLaunchHostFunc(stream, myFunc, cbData);
//       //                                 ↑
//       //   CallbackData* 隐式转换为 void*
//       //   编译器：地址值不变，但类型标签丢失
//       //   驱动：只看到一个地址，不知道它指向什么
//
//     还原过程（在 hostFunc 内）：
//       void myFunc(void *data) {
//           auto *d = static_cast<CallbackData *>(data);
//           //         ↑
//           //   static_cast：告诉编译器"我知道这个地址指向 CallbackData"
//           //   运行时：只是把地址重新解释为 CallbackData* 类型，
//           //           不做任何验证（程序员自己负责类型正确）
//       }
//
//   为什么用 void* 而不是直接传结构体？
//
//     函数指针签名必须固定为 void (*)(void*)：
//       驱动不可能知道每个用户的回调函数需要什么参数类型，
//       所以用 void* 作为统一的"任意数据"接口。
//       这是 C 时代的泛型模式，等价于 C++ 的模板或 std::function，
//       但更底层、更通用（跨语言接口也可用）。
//
//   void* vs C++ 类型安全：
//
//     static_cast<T*>(void*) 不做运行时类型检查：
//       如果实际传入的是 TypeA*，却 cast 成 TypeB*，行为未定义。
//       程序员必须保证传入和 cast 时的类型一致。
//       dynamic_cast 不能用于 void*（需要多态类型）。
//
//   生命周期（最重要的陷阱）：
//
//     cudaLaunchHostFunc 是异步的：函数调用立刻返回，hostFunc 稍后才执行。
//     data 指向的内存必须在 hostFunc 执行期间依然有效。
//
//     ✅ 安全：heap 分配（new），在 hostFunc 内或之后 delete
//     ✅ 安全：全局变量、static 变量
//     ✅ 安全：生命周期明确超过 hostFunc 执行时间的栈变量
//     ❌ 危险：局部栈变量，函数返回后 hostFunc 还未执行（悬空指针）


// ══════════════════════════════════
// 示例 A：基础 type-erased pointer 用法
// ══════════════════════════════════

// ── 步骤 1：定义自己的数据结构 ──────────
struct BasicData {
    int   id;       // 标识这次回调的编号
    float value;    // 附带的数值
};

// ── 步骤 2：定义 hostFunc，参数固定为 void*，内部 cast 还原 ──────────
void basicHostFunc(void *data)
{
    // static_cast<T*>：将 void* 还原为原始类型
    // 这是合法的：传入时是 BasicData*，cast 回 BasicData* 类型正确
    auto *d = static_cast<BasicData *>(data);

    // 现在可以正常访问结构体成员
    printf("  [basicHostFunc] id=%d, value=%.2f\n", d->id, d->value);

    // ❌ 不能在这里调用任何 CUDA API（死锁，见 cuda_launch_host_func.cu）
}

void demo_basic(void)
{
    printf("\n─── 示例 A：基础 type-erased pointer ───\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));

    int threads = 256, blocks = (N + threads - 1) / threads;
    fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N);

    // ── 步骤 3：分配数据，确保生命周期足够长 ─────────────────────────
    //
    //   用 new 分配到 heap：hostFunc 执行时此内存依然有效。
    //   传入时：BasicData* → void*（隐式转换，地址不变，类型擦除）
    auto *cbData = new BasicData{42, 3.14f};
    CUDA_CHECK(cudaLaunchHostFunc(stream, basicHostFunc, cbData));
    //                                                   ↑
    //   cbData 是 BasicData*，传给 void* 参数：隐式类型擦除
    //   驱动保存这个地址，kernel 完成后调用 basicHostFunc(cbData)

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── 步骤 4：hostFunc 执行完后再释放 ──────────────────────────────
    //   StreamSynchronize 保证 hostFunc 已执行完，此时 delete 安全
    delete cbData;

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════════════════════════
// 示例 B：多字段结构体（results、count、ready flag）
// ════════════════════════════════

struct CallbackData {
    float              *results;    // 指向已完成的 GPU 结果（device 指针）
    size_t              count;      // 元素个数
    std::atomic<bool>  *ready;      // 通知主线程：hostFunc 已执行
};

// hostFunc：
//   此时 stream 中 hostFunc 之前的所有操作（kernel A 等）已完成。
//   stream 中 hostFunc 之后的操作（kernel B 等）可能正在执行。
void myHostFunc(void *data)
{
    // 类型还原：void* → CallbackData*
    auto *d = static_cast<CallbackData *>(data);

    // ✅ 可以：访问结构体字段，做纯 CPU 端的工作
    printf("  [myHostFunc] GPU 结果已就绪，count=%zu，results ptr=%p\n",
           d->count, (void *)d->results);

    // ✅ 可以：使用 OS 同步原语通知其他线程
    //   memory_order_release：确保之前的 CPU 写操作对其他线程可见
    d->ready->store(true, std::memory_order_release);

    // ❌ 禁止：任何 CUDA API
    // cudaMemcpy(h_buf, d->results, ...);   // → 死锁
    // cudaMalloc(...);                       // → 死锁
}

void demo_callback_data(void)
{
    printf("\n─── 示例 B：多字段结构体 + atomic ready flag ───\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_results, *h_results;
    CUDA_CHECK(cudaMalloc((void **)&d_results, SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_results, SIZE));  // pinned

    int threads = 256, blocks = (N + threads - 1) / threads;

    // kernel A：产生结果
    fillKernel<<<blocks, threads, 0, stream>>>(d_results, 9.9f, N);

    // 准备回调数据（heap 分配，生命周期由主线程管理）
    std::atomic<bool> readyFlag{false};
    auto *cbData = new CallbackData{d_results, (size_t)N, &readyFlag};

    // 插入 hostFunc：kernel A 完成后触发
    CUDA_CHECK(cudaLaunchHostFunc(stream, myHostFunc, cbData));

    // 主线程：轮询等待 hostFunc 已执行（也可用 semaphore/cv 替代）
    while (!readyFlag.load(std::memory_order_acquire)) {
        // CPU 做其他工作，或 yield 让出时间片
        std::this_thread::yield();
    }
    printf("  主线程：readyFlag 已置位，开始 D2H 拷贝\n");

    // 此时 hostFunc 已返回，CUDA mutex 已释放，可以安全调用 CUDA API
    CUDA_CHECK(cudaMemcpy(h_results, d_results, SIZE, cudaMemcpyDeviceToHost));
    printf("  主线程：D2H 完成，h_results[0] = %.2f\n", h_results[0]);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    delete cbData;

    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFreeHost(h_results));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ════════════
// 示例 C：生命周期管理——stack vs heap
// ════════════

struct TimingData {
    const char *label;
    long long   submit_ns;   // 提交时的 CPU 时间戳（仅示意）
};

void timingHostFunc(void *data)
{
    auto *d = static_cast<TimingData *>(data);
    printf("  [timingHostFunc] label=%s\n", d->label);
}

void demo_lifetime(void)
{
    printf("\n─── 示例 C：生命周期管理 ───\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *d_buf;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));
    int threads = 256, blocks = (N + threads - 1) / threads;

    // ── ✅ 正确：heap 分配，生命周期明确 ───────
    {
        auto *heapData = new TimingData{"heap-kernel", 0};
        fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 1.0f, N);
        CUDA_CHECK(cudaLaunchHostFunc(stream, timingHostFunc, heapData));
        // heapData 在 hostFunc 执行期间仍然有效（heap 不会自动释放）
        CUDA_CHECK(cudaStreamSynchronize(stream));
        delete heapData;  // 同步后再 delete，安全
        printf("  ✅ heap 分配：正确，hostFunc 执行时内存有效\n");
    }

    // ── ✅ 正确：static 局部变量，生命周期=程序运行期 ─────────────────
    {
        // static：存储在静态存储区，不随函数返回而销毁
        static TimingData staticData{"static-kernel", 0};
        fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 2.0f, N);
        CUDA_CHECK(cudaLaunchHostFunc(stream, timingHostFunc, &staticData));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("  ✅ static 变量：正确，生命周期 = 程序运行期\n");
    }

    // ── ✅ 正确：栈变量 + StreamSynchronize 确保 hostFunc 先执行 ──────
    {
        // 栈变量本身有风险，但只要在函数返回前 Synchronize，就能保证
        // hostFunc 在栈帧销毁前已执行完
        TimingData stackData{"stack-kernel-safe", 0};
        fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 3.0f, N);
        CUDA_CHECK(cudaLaunchHostFunc(stream, timingHostFunc, &stackData));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Synchronize 返回时 hostFunc 已执行完，stackData 仍在栈上，安全
        printf("  ✅ 栈变量（Synchronize 保护）：正确\n");
    }   // stackData 在此处销毁，但 hostFunc 已执行完

    // ── ❌ 危险：栈变量，函数返回后可能访问悬空指针（注释，不运行）──
    //
    //   void bad_lifetime_example(cudaStream_t stream) {
    //       TimingData localData{"dangling", 0};    // 栈变量
    //       fillKernel<<<...>>>(d_buf, 1.0f, N, stream);
    //       cudaLaunchHostFunc(stream, timingHostFunc, &localData);
    //       // 函数返回，localData 被销毁
    //       // hostFunc 稍后执行，访问已销毁的内存 → 未定义行为
    //   }   // ← localData 在此销毁，但 hostFunc 还没执行！

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ══════════════
// 示例 D：多个 stream，各自携带独立数据
// ═════════════

struct StreamData {
    int   stream_id;
    float val;
    std::counting_semaphore<1> *sem;
};

void streamHostFunc(void *data)
{
    auto *d = static_cast<StreamData *>(data);
    printf("  [streamHostFunc] stream_id=%d, val=%.1f\n", d->stream_id, d->val);
    d->sem->release();   // 通知对应线程
}

void demo_multi_stream(void)
{
    printf("\n─── 示例 D：多 stream，各自独立的 void* data ───\n");

    const int NSTREAMS = 3;
    cudaStream_t streams[NSTREAMS];
    float *d_bufs[NSTREAMS];

    // 每个 stream 有独立的 semaphore 和数据结构
    std::counting_semaphore<1> sems[NSTREAMS] = {
        std::counting_semaphore<1>(0),
        std::counting_semaphore<1>(0),
        std::counting_semaphore<1>(0)
    };

    StreamData cbDatas[NSTREAMS];
    int threads = 256, blocks = (N + threads - 1) / threads;

    for (int i = 0; i < NSTREAMS; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc((void **)&d_bufs[i], SIZE));

        // 每个 stream 的 data 指向不同的 cbDatas[i]
        cbDatas[i] = {i, (float)(i * 10), &sems[i]};

        fillKernel<<<blocks, threads, 0, streams[i]>>>(d_bufs[i], (float)i, N);

        // 传入各自的 &cbDatas[i]：void* 指向不同地址，互不干扰
        CUDA_CHECK(cudaLaunchHostFunc(streams[i], streamHostFunc, &cbDatas[i]));
    }

    // 等待所有 stream 的 hostFunc 完成
    for (int i = 0; i < NSTREAMS; i++) {
        sems[i].acquire();
    }
    printf("  所有 stream 的 hostFunc 均已执行\n");

    for (int i = 0; i < NSTREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaFree(d_bufs[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}


// ═════════════
// 示例 E：完整流程——void* 传递 + semaphore 通知 + 另一线程调 CUDA API
// ════════════

struct FullData {
    float                       *d_src;     // GPU 结果指针
    float                       *h_dst;     // Host 目标指针（pinned）
    size_t                       bytes;
    std::counting_semaphore<1>  *sem;       // 通知消费线程
};

// hostFunc：只通知，立刻返回
void fullHostFunc(void *data)
{
    auto *d = static_cast<FullData *>(data);
    d->sem->release();
    // 不调用 cudaMemcpy，不调用任何 CUDA API
}

void demo_full(void)
{
    printf("\n─── 示例 E：完整流程 ───\n");

    std::counting_semaphore<1> sem(0);

    float *d_results, *h_results;
    CUDA_CHECK(cudaMalloc((void **)&d_results, SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_results, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 分配 heap 数据，void* 将指向此结构体
    auto *fd = new FullData{d_results, h_results, SIZE, &sem};

    int threads = 256, blocks = (N + threads - 1) / threads;
    fillKernel<<<blocks, threads, 0, stream>>>(d_results, 7.77f, N);

    // cudaLaunchHostFunc 的 void* 传递：
    //   fd（FullData*）→ 隐式转换为 void* → 驱动原样保存
    //   kernel 完成后，驱动调用 fullHostFunc(fd)
    //   fullHostFunc 内：void* → static_cast<FullData*> 还原
    CUDA_CHECK(cudaLaunchHostFunc(stream, fullHostFunc, fd));

    // 消费线程：等通知后做 D2H
    std::thread consumer([&]() {
        sem.acquire();   // 等 fullHostFunc 的 release

        // hostFunc 已返回，CUDA mutex 已释放
        CUDA_CHECK(cudaMemcpy(fd->h_dst, fd->d_src, fd->bytes,
                              cudaMemcpyDeviceToHost));
        printf("  consumer：D2H 完成，h_results[0] = %.2f\n", fd->h_dst[0]);
    });

    consumer.join();  // consumer.join() 阻塞主线程，直到 consumer 线程的函数体（lambda）执行完毕返回。  
    CUDA_CHECK(cudaStreamSynchronize(stream));
    delete fd;

    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFreeHost(h_results));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


int main(void)
{
    demo_basic();
    demo_callback_data();
    demo_lifetime();
    demo_multi_stream();
    demo_full();

    // ── 速查：void* 类型擦除三步走 ───────
    //
    //   步骤       代码                             说明
    //   ───────    ─────────────────────────   ────────────────────────
    //   定义       struct MyData { ... };           自定义数据结构
    //   传入       cudaLaunchHostFunc(..., &data)   &data: T* → void*（擦除）
    //   还原       static_cast<MyData*>(data)       void* → T*（还原，无验证）
    //
    // ── 速查：生命周期规则 ───────────
    //
    //   方式                  安全条件
    //   ─────────────────     ──────────────────────
    //   new（heap）           在 hostFunc 执行后 delete
    //   static 变量           始终安全（程序生命周期）
    //   栈变量                必须在函数返回前 StreamSynchronize
    //   局部变量（无同步）    ❌ 危险，悬空指针

    printf("\n全部示例完成。\n");
    return 0;
}
