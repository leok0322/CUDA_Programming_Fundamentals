/**
 * cuda_launch_host_func.cu
 *
 * 详细解析 cudaLaunchHostFunc 及其死锁陷阱与正确用法。
 *
 * 内容：
 *   1. cudaLaunchHostFunc 签名与语义
 *   2. 死锁原理：为何在 hostFunc 内调用 CUDA API 会死锁
 *   3. 解决方案：hostFunc 只通知，另一个线程调用 CUDA API
 *   4. 示例 A：错误写法（注释掉，不运行）
 *   5. 示例 B：正确写法——std::counting_semaphore
 *   6. 示例 C：正确写法——std::condition_variable（传递更多数据）
 *   7. 示例 D：完整流水线（GPU kernel → hostFunc 通知 → CPU 线程处理结果）
 *
 * 编译（需要 C++20 或 C++17）：
 *   nvcc -O2 -arch=sm_80 -std=c++20 cuda_launch_host_func.cu -o cuda_launch_host_func
 *
 * 运行：
 *   ./cuda_launch_host_func
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <thread>
#include <semaphore>       // C++20：std::counting_semaphore
#include <mutex>
#include <condition_variable>
#include <atomic>

// ────────────
// 错误检查宏
// ───────────
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

__global__ void fillKernel(float *d_out, float val, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_out[idx] = val;
}

static const int    N    = 1 << 20;
static const size_t SIZE = N * sizeof(float);





// ═════════════════════
// 死锁原理详解
// ═════════════════════
//
//   CUDA 驱动内部有一把（或多把）全局互斥锁，用于保护 stream 状态、
//   设备上下文等共享数据结构。几乎所有 CUDA API 调用都需要先获取这把锁。
//
//   cudaLaunchHostFunc 的执行路径（简化）：
//
//     ┌─────────────────────────────────────────────────────────────┐
//     │  CUDA 驱动 worker thread                                    │
//     │                                                             │
//     │  1. 持有内部锁 mutex（正在处理 stream 命令队列）            │
//     │  2. 发现队列里有 HostFunc 命令                              │
//     │  3. 调用 fn(userData)   ← 此时 mutex 仍被持有              │
//     │     fn 内部：                                               │
//     │       cudaMemcpy(...)   ← 尝试获取 mutex                   │
//     │                            mutex 已被 worker thread 持有！  │
//     │                            → 永远等待 → 死锁               │
//     └─────────────────────────────────────────────────────────────┘
//
//   图示：
//
//     worker thread（持有 mutex）:
//         → 调用 fn(userData)
//             fn 内部调用 cudaMemcpy
//                 cudaMemcpy 尝试加锁 mutex ──────┐
//                                                 │ 等待
//     worker thread 持有 mutex，等 fn 返回 ←──── ┘
//
//     两方互相等待 → 死锁（deadlock）


// ═══════════════════
// 解决方案：hostFunc 只做通知，另一个线程调用 CUDA API
// ═══════════════════
//
//   原则：hostFunc 内只做一件事——通过 OS 原语通知另一个线程，立刻返回。
//   由那个线程调用 CUDA API。那个线程调用 CUDA API 时走正常路径，
//   mutex 已经空闲（worker thread 已从 fn 返回并释放 mutex），可以正常获取。
//
//   正确的执行流：
//
//     worker thread（持有 mutex）:
//         → 调用 fn(userData)
//             fn 内部：
//                 sem.release()    ← 只做这一件事，立刻返回（不碰 CUDA API）
//         → fn 返回
//         → worker thread 释放 mutex    ← 锁释放了
//
//     用户线程（一直在等 sem）:
//         sem.acquire()            ← 被唤醒
//         cudaMemcpy(...)          ← mutex 已空闲，正常获取，没有死锁
//
//   可用的 OS 原语（只要能"通知"即可）：
//     std::counting_semaphore   最简单，推荐
//     std::condition_variable   可以携带更多状态信息
//     std::atomic + notify      轻量，C++20 atomic wait/notify
//     pipe / eventfd            POSIX，跨进程也适用


// ─────────────────
// 示例 A：错误写法（死锁，仅注释展示，不运行）
// ──────────────────
//
//   struct BadData { float *dst; float *src; size_t size; };
//
//   // 错误：在 hostFunc 内直接调用 cudaMemcpy
//   void badHostFunc(void *userData) {
//       auto *d = static_cast<BadData *>(userData);
//       // ↓ 死锁：worker thread 持有 mutex，cudaMemcpy 也要获取 mutex
//       cudaMemcpy(d->dst, d->src, d->size, cudaMemcpyDeviceToHost);
//   }
//
//   void bad_example(cudaStream_t stream) {
//       BadData data = { h_dst, d_src, SIZE };
//       cudaLaunchHostFunc(stream, badHostFunc, &data);
//       cudaStreamSynchronize(stream);   // 永远不会返回
//   }


// ─────────────
// 示例 B：正确写法——std::counting_semaphore
//
//   hostFunc 只 release semaphore，另一个线程等待后调用 CUDA API。
// ──────────────

// hostFunc：只 release，不碰任何 CUDA API，立刻返回
void hostFuncSem(void *userData)
{
    // userData 是调用方传入的 semaphore 指针
    auto *sem = static_cast<std::counting_semaphore<1> *>(userData);

    // 只做这一件事：通知等待方
    // counting_semaphore<1>：最大计数为 1，release 将计数从 0 变为 1
    sem->release();

    // 立刻返回，不做任何其他操作
    // 此时 worker thread 持有的 mutex 不受影响
}

void demo_semaphore(void)
{
    printf("\n─── 示例 B：counting_semaphore ───\n");

    // 初始计数为 0：用户线程 acquire 时会阻塞，等 hostFunc release 后才继续
    std::counting_semaphore<1> sem(0);

    float *d_buf, *h_result;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_result, SIZE));  // pinned，D2H 更快

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;

    // ① GPU kernel 填充数据
    fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 3.14f, N);

    // ② kernel 完成后，驱动 worker thread 调用 hostFuncSem
    //    hostFuncSem 只 release semaphore，立刻返回
    CUDA_CHECK(cudaLaunchHostFunc(stream, hostFuncSem, &sem));

    // ③ 启动 consumer 线程
    //
    //   std::thread consumer( [&]() { ... } );
    //   │            │         │
    //   │            │         └─ lambda 表达式作为线程函数体
    //   │            │              [&]：按引用捕获外层所有变量
    //   │            │                   sem、h_result、d_buf 不是拷贝，
    //   │            │                   是对原变量的引用，lambda 执行期间
    //   │            │                   这些变量必须保持有效（不能提前销毁）
    //   │            └─ 线程对象名
    //   └─ 构造 std::thread，传入 lambda 作为入口
    //      构造完成后 OS 立刻创建新线程并执行 lambda，
    //      主线程同时继续向下执行（真正并发）
    //
    //   为何用独立线程而不在主线程直接 acquire？
    //     主线程直接 acquire → 主线程阻塞 → 无法继续提交工作 → 串行（不死锁但失去并发）
    //     独立线程 acquire   → consumer 阻塞，主线程继续运行 → 真正并发
    //
    std::thread consumer([&]() {
        // sem.acquire()：
        //   sem 初始计数为 0（构造时传入 0），consumer 线程在此挂起阻塞。
        //   等 hostFuncSem 调用 sem.release()（计数 0→1）后被唤醒，
        //   acquire 将计数 1→0，然后继续执行。
        sem.acquire();

        // acquire 返回意味着：
        //   1. hostFuncSem 已执行 sem.release()
        //   2. hostFuncSem 已返回（release 后立刻结束）
        //   3. CUDA worker thread 随即释放了内部 mutex
        //   → CUDA 内部锁已空闲，可以安全调用任何 CUDA API
        CUDA_CHECK(cudaMemcpy(h_result, d_buf, SIZE, cudaMemcpyDeviceToHost));
        printf("  consumer 线程：D2H 完成，h_result[0] = %.2f\n", h_result[0]);
    });
    // consumer 线程已启动（阻塞在 acquire），主线程继续向下

    // 主线程继续做其他工作（stream 仍在执行）
    printf("  主线程：已提交 kernel 和 hostFunc，继续做其他工作...\n");

    // consumer.join()：
    //   主线程阻塞，等 consumer 线程执行完毕后才继续。
    //   必须 join（或 detach），否则 std::thread 析构时线程仍在运行
    //   会调用 std::terminate() 强制终止程序。
    consumer.join();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ───────────────
// 示例 C：正确写法——std::condition_variable（携带状态信息）
//
//   hostFunc 需要传递更多数据给等待线程时，用 condition_variable + 共享结构体。
// ─────────────────

// 共享的通知结构体：hostFunc 写入，用户线程读取
struct NotifyData {
    // std::mutex：互斥锁
    //   同一时刻只有一个线程能持有（lock），其他线程尝试 lock 时阻塞。
    //   用途：保护共享变量（ready、batchId）的读写，防止数据竞争。
    //   不能直接 lock/unlock，通常配合 lock_guard / unique_lock 使用
    //   （RAII：构造时加锁，析构时自动解锁，异常安全）。
    std::mutex              mtx;

    // std::condition_variable：条件变量
    //   必须与 std::mutex 配合使用。
    //   核心操作：
    //     wait(lk, pred)   ：释放 lk 持有的锁，挂起线程；
    //                        被唤醒后重新获取锁，检查 pred()，
    //                        若 pred() 为 false 则再次挂起（防止虚假唤醒）。
    //     notify_one()     ：唤醒一个正在 wait 的线程。
    //     notify_all()     ：唤醒所有正在 wait 的线程。
    //   与 semaphore 的区别：
    //     semaphore  只传"信号"（计数），不携带其他信息
    //     condition_variable 可以配合共享变量传递任意状态（如 batchId）
    std::condition_variable cv;

    bool                    ready  = false;  // hostFunc 是否已触发
    int                     batchId = -1;    // hostFunc 传递的附加信息
    float                  *d_src  = nullptr;
    float                  *h_dst  = nullptr;
    size_t                  bytes  = 0;
};

// hostFunc：只写共享状态 + notify，立刻返回
void hostFuncCV(void *userData)
{
    auto *nd = static_cast<NotifyData *>(userData);

    {
        // std::lock_guard<std::mutex> lk(nd->mtx)：
        //   RAII 加锁：构造时调用 nd->mtx.lock()，析构时自动调用 nd->mtx.unlock()。
        //   作用域结束（右花括号）时 lk 析构，锁自动释放。
        //   这把锁是用户自己的锁，与 CUDA 内部锁完全独立，不会死锁。
        //
        //   nd->mtx 与下方 consumer 线程里的 nd.mtx 是同一把锁：
        //     nd 是 NotifyData* 指针，nd->mtx 等价于 (*nd).mtx
        //     consumer 线程里 nd 是 NotifyData 对象引用，nd.mtx 直接访问同一成员
        //     → 两处操作的是 NotifyData 结构体里唯一的那个 std::mutex 对象
        //     → 同一把锁才能互斥地保护 ready / batchId 的读写
        std::lock_guard<std::mutex> lk(nd->mtx);
        nd->ready   = true;
        nd->batchId = 42;   // 告诉用户线程是哪一批数据完成了
    }
    // 此处 lk 析构，nd->mtx 已解锁
    // notify_one()：唤醒一个正在 nd.cv.wait() 的线程
    // 必须在解锁后（或解锁前）调用，解锁后调用可避免被唤醒的线程立刻又阻塞在锁上
    nd->cv.notify_one();

    // 立刻返回，不调用任何 CUDA API
}

void demo_condition_variable(void)
{
    printf("\n─── 示例 C：condition_variable（携带状态）───\n");

    NotifyData nd;  // 默认初始化
    float *d_buf, *h_result;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&h_result, SIZE));
    nd.d_src  = d_buf;
    nd.h_dst  = h_result;
    nd.bytes  = SIZE;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256, blocks = (N + threads - 1) / threads;
    fillKernel<<<blocks, threads, 0, stream>>>(d_buf, 2.71f, N);
    CUDA_CHECK(cudaLaunchHostFunc(stream, hostFuncCV, &nd));

    std::thread consumer([&]() {
        // std::unique_lock<std::mutex> lk(nd.mtx)：
        //   与 lock_guard 类似，构造时加锁，析构时解锁。
        //   区别：unique_lock 可以手动 unlock() / lock()，
        //         而 lock_guard 不行（只能在析构时解锁）。
        //   condition_variable::wait 要求传入 unique_lock（因为 wait 内部需要临时解锁）。
        //
        //   nd.mtx 与上方 hostFuncCV 里的 nd->mtx 是同一把锁：
        //     hostFuncCV 拿到的 nd 是 NotifyData*，访问方式是 nd->mtx
        //     consumer 线程里 nd 是 NotifyData 对象，访问方式是 nd.mtx
        //     → 语法不同，但指向同一个 std::mutex 对象（NotifyData 里唯一的 mtx）
        //
        //   两把"不同的锁"包裹同一个 mutex，时序协作：
        //     consumer:   unique_lock 加锁 nd.mtx → wait 内部释放锁 → 挂起
        //     hostFuncCV: lock_guard  加锁 nd->mtx（同一把）→ 写 ready=true → 解锁 → notify
        //     consumer:   被唤醒 → 重新加锁 nd.mtx → 确认 ready==true → wait 返回
        std::unique_lock<std::mutex> lk(nd.mtx);

        // nd.cv.wait(lk, predicate)：
        //   ① 检查 predicate（[&] { return nd.ready; }）：
        //      若已为 true（hostFuncCV 在 wait 前就执行完了）→ 直接继续，不挂起
        //      若为 false → 执行 ②
        //   ② 原子地释放 lk 持有的锁 + 挂起当前线程
        //      （释放锁是为了让 hostFuncCV 能获取锁写入 ready）
        //   ③ 被 notify_one() 唤醒后，重新获取锁，再次检查 predicate：
        //      若 true  → wait 返回，lk 重新持有锁
        //      若 false → 虚假唤醒（spurious wakeup），再次释放锁挂起
        //   predicate 的作用：防止虚假唤醒导致逻辑错误
        nd.cv.wait(lk, [&] { return nd.ready; });


        // ● 是的，两者构造时都会立刻加锁。但它们不会同时持有锁，因为 std::mutex 的本质就是"同一时刻只有一个线程能持有"：                                    
                                                                                                                                                        
        //   // 线程 A 先持有锁                                                                                                                              
        //   std::lock_guard lk(mtx);      // mtx.lock() 成功，A 持有                                                                                        
                                                                                                                                                        
        //   // 线程 B 同时尝试                                                                                                                              
        //   std::unique_lock lk(mtx);     // mtx.lock() → 阻塞，等 A 释放                                                                                   
                                                                                                                                                        
        //   关键在于 cv.wait() 内部会临时释放锁：                                                                                                           
        
        //   consumer 线程：                                                                                                                                 
        //     unique_lock lk(nd.mtx)      ← 加锁（持有）
        //     cv.wait(lk, pred)                                                                                                                             
        //       ├─ 内部：lk.unlock()      ← 释放锁！（让 hostFuncCV 能加锁）
        //       ├─ 挂起线程                                                                                                                                 
        //       ├─ 被 notify 唤醒
        //       └─ 内部：lk.lock()        ← 重新加锁，wait 返回                                                                                             
                                                                                                                                                        
        //   hostFuncCV 线程（wait 释放锁后）：                                                                                                              
        //     lock_guard lk(nd->mtx)      ← 此时锁空闲，加锁成功                                                                                            
        //     nd->ready = true                                                                                                                              
        //     // lock_guard 析构，解锁                                                                                                                      
        //     cv.notify_one()                                                                                                                               
                                                                                                                                                        
        //   时序图：                                                                                                                                        
        
        //   consumer:    [加锁]──[wait: 释放锁, 挂起]────────────[唤醒, 重新加锁]──[解锁]                                                                   
        //                                       ↓ 锁空闲                                                                                                    
        //   hostFuncCV:                    [加锁]──[写 ready]──[解锁]──[notify]                                                                             
                                                                                                                                                        
        //   所以两者从来不会同时持有锁——wait 释放锁是为了让 hostFuncCV 能获取锁写入数据，notify 之后 consumer 才重新加锁读取数据。这是 condition_variable   
        //   必须配合 unique_lock（而非 lock_guard）的原因：wait 需要能调用 lk.unlock()，而 lock_guard 不支持手动解锁。   

        // wait 返回时 lk 重新持有锁，nd.ready == true 已确认
        int bid = nd.batchId;  // 读取 hostFunc 写入的附加信息
        lk.unlock();           // 手动解锁：后面的 cudaMemcpy 不需要持有锁

        // 读到了 hostFunc 传来的 batchId，再调用 CUDA API（安全）
        CUDA_CHECK(cudaMemcpy(nd.h_dst, nd.d_src, nd.bytes, cudaMemcpyDeviceToHost));
        printf("  consumer 线程：batch %d 完成，h_result[0] = %.2f\n",
               bid, nd.h_dst[0]);
    });

    printf("  主线程：已提交，等待 consumer 线程...\n");
    consumer.join();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


// ─────────────────
// 示例 D：完整流水线
//
//   多轮循环：每轮 GPU kernel 完成后，hostFunc 通知 CPU 线程做 D2H 拷贝，
//   CPU 线程处理完后再触发下一轮。
//
//   时间轴（理想情况下 GPU 和 CPU 重叠）：
//
//     GPU stream: [kernel 0] [HostFunc 0] [kernel 1] [HostFunc 1] ...
//                                  ↓notify          ↓notify
//     CPU thread: ─────────── [D2H 0] ──────────── [D2H 1] ──────
//
//   注意：这里 D2H 在 hostFunc 之外，不会死锁。
//         如果 D2H 本身很慢，可能跟不上 GPU；实际生产中可用双缓冲优化。
// ──────────────────

struct PipelineData {
    std::counting_semaphore<1> sem{0};  // hostFunc → consumer 的信号
    float   *d_buf    = nullptr;
    float   *h_result = nullptr;
    int      round    = -1;             // 当前轮次（hostFunc 写入前由主线程设置）
};

void hostFuncPipeline(void *userData)
{
    auto *pd = static_cast<PipelineData *>(userData);
    pd->sem.release();   // 只通知，立刻返回，从0变为1
}

void demo_pipeline(void)
{
    printf("\n─── 示例 D：完整流水线（多轮 kernel → hostFunc → D2H）───\n");

    PipelineData pd;
    CUDA_CHECK(cudaMalloc((void **)&pd.d_buf,    SIZE));
    CUDA_CHECK(cudaMallocHost((void **)&pd.h_result, SIZE));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const int ROUNDS  = 4;
    int threads = 256, blocks = (N + threads - 1) / threads;

    // consumer 线程：每轮等通知，做 D2H，然后处理结果
    std::thread consumer([&]() {
        for (int r = 0; r < ROUNDS; r++) {
            pd.sem.acquire();   // 等 hostFuncPipeline 的信号，从1变为0

            // hostFunc 已返回，mutex 已释放，可以安全调用 CUDA API
            CUDA_CHECK(cudaMemcpy(pd.h_result, pd.d_buf, SIZE,
                                  cudaMemcpyDeviceToHost));
            printf("  consumer：第 %d 轮 D2H 完成，h_result[0] = %.2f\n",
                   r, pd.h_result[0]);
        }
    });

    // 主线程：连续提交多轮 kernel + hostFunc
    for (int r = 0; r < ROUNDS; r++) {
        pd.round = r;
        fillKernel<<<blocks, threads, 0, stream>>>(pd.d_buf, (float)r, N);
        CUDA_CHECK(cudaLaunchHostFunc(stream, hostFuncPipeline, &pd));
    }

    // consumer.join()：
    //   主线程阻塞，等 consumer 线程的 for 循环跑完（所有轮 D2H 完成）后才继续。
    //
    //   替代方案：用 std::atomic<bool> + 轮询（主线程不能阻塞时使用）
    //
    //     std::atomic<bool> done{false};
    //     // consumer 线程末尾：
    //     done.store(true, std::memory_order_release);
    //
    //     // 主线程轮询（不阻塞，可同时做其他工作）：
    //     while (!done.load(std::memory_order_acquire)) {
    //         // 做其他工作，或让出时间片
    //         std::this_thread::yield();
    //     }
    //
    //   join vs 轮询对比：
    //     join      主线程阻塞，简单，无法同时做其他事
    //     轮询      主线程不阻塞，可并行做其他工作，但需要手动管理 atomic 标志

    // 没有 yield（忙等）：                                                                                                                            
    // while (!done) { }        ← CPU 核心 100% 占用，一直空转检查                                                                                   
                                                                                                                                                    
    // 有 yield：                                                                                                                                      
    // while (!done) { yield(); }                                                                                                                    
    // ↓                                                                                                                                             
    // 检查 done → false → yield() → OS 调度器接管
    //                                 ├─ 运行其他线程（consumer、hostFunc 等）                                                                       
    //                                 └─ 稍后再切回来检查 done                                                                                       
                                                                                                                                                    
    // yield 不保证睡多久：                                                                                                                            
                                                                                                                                                    
    // yield()  → 让出时间片，OS 决定何时切回（可能立刻，可能几 ms）                                                                                   
    // sleep_for(1ms) → 至少睡 1ms，更可预期但响应更慢                                                                                                 
                                                                                                                                                    
    // 三种等待方式对比：                                                                                                                              
                                                                                                                                                    
    // 方式                        CPU 占用    响应延迟    适用场景
    // ──────────────────────────  ──────────  ──────────  ──────────────────────                                                                      
    // while (!done) {}            100%        最低        等待极短（ns 级）
    // while (!done) { yield(); }  低          低          等待时间不确定                                                                              
    // while (!done) { sleep(1ms)} 极低        ≥1ms        等待较长，不在乎延迟                                                                        
    // join()                      0           OS 唤醒     不需要同时做其他事
    consumer.join();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(pd.d_buf));
    CUDA_CHECK(cudaFreeHost(pd.h_result));
    CUDA_CHECK(cudaStreamDestroy(stream));
}


int main(void)
{
    demo_semaphore();
    demo_condition_variable();
    demo_pipeline();

    // ── 速查：hostFunc 内允许 / 禁止的操作 ─────────────────────────
    //
    //   允许                              禁止
    //   ───────────────────────────────   ────────────────────────────────
    //   sem.release()                     cudaMemcpy / cudaMemcpyAsync
    //   cv.notify_one() / notify_all()    cudaLaunchKernel / kernel<<<>>>
    //   atomic.store() / notify()         cudaStreamSynchronize
    //   write to shared memory            cudaEventSynchronize
    //   printf / logging（短暂）          任何其他 CUDA Runtime API
    //
    // ── 速查：三种通知原语对比 ──────────────────────────────────────
    //
    //   原语                      优点                    适用场景
    //   ──────────────────────    ──────────────────────  ────────────────────
    //   counting_semaphore        最简单，无额外数据       只需通知"完成"
    //   condition_variable        可携带任意共享状态       需要传递批次ID等信息
    //   atomic + wait/notify      最轻量（C++20）          高频通知，低开销
    //
    // ── 核心原则 ────────────────────────────────────────────────────
    //
    //   hostFunc 内只做一件事：通知另一个线程。
    //   由那个线程去调用 CUDA API。
    //   那个线程调用 CUDA API 时，CUDA 内部 mutex 已空闲，不会死锁。

    printf("\n全部示例完成。\n");
    return 0;
}
