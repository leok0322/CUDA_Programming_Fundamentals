/**
 * cuda_host_register.cu
 *
 * 详细解析 cudaHostRegister 和 cudaHostUnregister 的用法。
 *
 * 核心问题：
 *   cudaMallocHost / cudaHostAlloc 是"先分配再锁定"，
 *   但有时内存已经由 malloc / new / 系统调用分配好了，
 *   cudaHostRegister 解决的就是"事后锁定"的问题：
 *   把一块已存在的 pageable 内存原地注册为 pinned memory。
 *
 * 编译：
 *   nvcc -O2 -arch=sm_80 cuda_host_register.cu -o host_register
 *
 * 运行：
 *   ./host_register
 */

#include <cuda_runtime.h>  // cudaHostRegister、cudaHostUnregister 等
#include <stdio.h>         // printf、fprintf
#include <stdlib.h>        // malloc、free、exit、EXIT_FAILURE
#include <string.h>        // memset

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
// kernel：向量加一
// ─────────────────────────────────────────────
__global__ void addOneKernel(float *ptr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        ptr[idx] += 1.0f;
}

static const int    N    = 1 << 20;        // 1M 个 float
static const size_t SIZE = N * sizeof(float);



// ────────────────────────────
// 示例 1：cudaHostRegisterDefault
//
//   最基本用法：把 malloc 分配的内存锁定为 pinned，
//   使 cudaMemcpy 可以走 DMA 直传，不再需要内部 staging buffer。
//   锁定后 cudaMemcpy 的 H2D / D2H 带宽与 cudaMallocHost 相当。
// ────────────────────────────
void demo_register_default(void)
{
    printf("\n─── 示例 1：cudaHostRegisterDefault ───\n");

    // 1. 用 malloc 分配 pageable 内存（常见于已有代码，不能改分配方式）
    float *h_data = (float *)malloc(SIZE);
    if (!h_data) { fprintf(stderr, "malloc 失败\n"); return; }

    for (int i = 0; i < N; i++)
        h_data[i] = (float)i;

    // 2. 原地注册为 pinned memory
    //    物理地址不变，OS 锁定这些页，DMA 可直接访问
    CUDA_CHECK(cudaHostRegister(
        h_data,                    // [in] 已存在的 pageable 地址
        SIZE,                      // [in] 锁定字节数
        cudaHostRegisterDefault    // [in] 仅锁定，无额外映射
    ));
    printf("  注册后 h_data 已是 pinned，地址不变：%p\n", (void *)h_data);

    // 3. 分配 GPU 显存
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, SIZE));

    // 4. H2D：现在走 DMA 直传，无 staging buffer
    CUDA_CHECK(cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice));

    // 5. GPU 计算
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. D2H
    CUDA_CHECK(cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_data[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    // 7. 先 Unregister（解锁），再 free（释放）
    //    顺序不能颠倒：free 后内存不合法，Unregister 会出错
    CUDA_CHECK(cudaHostUnregister(h_data));   // 解除 pinned，归还给 OS 正常管理
    free(h_data);                              // 释放虚拟地址
    CUDA_CHECK(cudaFree(d_data));

    printf("  cudaHostUnregister + free 完成\n");
}

// ──────────────────────────────
// 示例 2：cudaHostRegisterMapped（零拷贝）
//
//   注册时附加 Mapped flag，GPU 可通过 cudaHostGetDevicePointer
//   拿到 device pointer，直接经 PCIe 访问 host 内存，无需 cudaMemcpy。
//   与 cudaHostAllocMapped 的区别：
//     cudaHostAllocMapped  ── 分配 + 映射（新内存）
//     cudaHostRegisterMapped ─ 注册 + 映射（已有内存）
// ─────────────────────────────
void demo_register_mapped(void)
{
    printf("\n─── 示例 2：cudaHostRegisterMapped（零拷贝）───\n");

    // 检查设备是否支持 Mapped
    int canMap = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, 0));
    if (!canMap) {
        printf("  设备不支持 Mapped，跳过本示例\n");
        return;
    }

    float *h_data = (float *)malloc(SIZE);
    if (!h_data) { fprintf(stderr, "malloc 失败\n"); return; }

    for (int i = 0; i < N; i++)
        h_data[i] = (float)i;

    // 注册 + 映射到 GPU 地址空间
    CUDA_CHECK(cudaHostRegister(
        h_data,
        SIZE,
        cudaHostRegisterMapped    // 锁定 + 允许 GPU 直接访问
    ));

    // 查询 GPU 侧映射地址（与 cudaHostAllocMapped 的用法完全相同）
    float *d_ptr = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer(
        (void **)&d_ptr,
        (void  *) h_data,
        0
    ));

    printf("  h_data (host 地址) : %p\n", (void *)h_data);
    printf("  d_ptr  (GPU 映射)  : %p\n", (void *)d_ptr);

    // GPU 直接访问 host 内存，无需 cudaMemcpy
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    addOneKernel<<<blocks, threads>>>(d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int errors = 0;
    for (int i = 0; i < N; i++)
        if (h_data[i] != (float)i + 1.0f) errors++;
    printf("  验证：%s（错误数 %d）\n", errors == 0 ? "PASS" : "FAIL", errors);

    CUDA_CHECK(cudaHostUnregister(h_data));
    free(h_data);
}

// ───────────────────────────────────
// 示例 3：cudaHostRegisterPortable（多 GPU）
//
//   默认情况下，cudaHostRegister 注册的 pinned memory
//   只对当前 CUDA context（当前 GPU）有效。
//   加 Portable flag 后，所有 GPU 都能高效访问同一块 pinned memory。
// ──────────────────────────────────
void demo_register_portable(void)
{
    printf("\n─── 示例 3：cudaHostRegisterPortable（多 GPU 可见）───\n");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("  系统中 GPU 数量：%d\n", deviceCount);

    float *h_data = (float *)malloc(SIZE);
    if (!h_data) { fprintf(stderr, "malloc 失败\n"); return; }

    for (int i = 0; i < N; i++)
        h_data[i] = (float)i;

    // Portable：所有 GPU 的 CUDA context 均可直接使用此 pinned memory
    CUDA_CHECK(cudaHostRegister(
        h_data,
        SIZE,
        cudaHostRegisterPortable   // 多 GPU 可见
    ));

    // 对每个 GPU 分别做 H2D 传输
    for (int dev = 0; dev < deviceCount; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));

        float *d_data = NULL;
        CUDA_CHECK(cudaMalloc((void **)&d_data, SIZE));

        // 不需要重新注册，Portable pinned memory 对所有设备都有效
        CUDA_CHECK(cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks  = (N + threads - 1) / threads;
        addOneKernel<<<blocks, threads>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 只取第一个元素做快速验证
        float result = 0.0f;
        CUDA_CHECK(cudaMemcpy(&result, d_data, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  GPU %d：h_data[0]=%.1f → result=%.1f  %s\n",
               dev, h_data[0], result, result == 1.0f ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_data));
    }

    // Unregister 只需调用一次，对所有 GPU 同时解除
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaHostUnregister(h_data));
    free(h_data);
}

// ─────────────────────────────────────────────────────────────────────
// 示例 4：错误用法演示（不执行，仅注释说明）
//
//   以下操作会导致错误，列出以便对比理解：
// ─────────────────────────────────────────────────────────────────────
//
//   // ❌ 重复注册同一地址范围
//   cudaHostRegister(ptr, SIZE, 0);
//   cudaHostRegister(ptr, SIZE, 0);   // 返回 cudaErrorHostMemoryAlreadyRegistered
//
//   // ❌ 用偏移地址 Unregister（必须用原始指针）
//   cudaHostRegister(ptr, SIZE, 0);
//   cudaHostUnregister(ptr + 100);    // 返回 cudaErrorInvalidValue
//
//   // ❌ 先 free 再 Unregister（内存已不合法）
//   cudaHostRegister(ptr, SIZE, 0);
//   free(ptr);                        // ptr 指向的内存已释放
//   cudaHostUnregister(ptr);          // 未定义行为
//
//   // ❌ 注册 cudaMallocHost 分配的内存（已经是 pinned）
//   cudaMallocHost(&ptr, SIZE);
//   cudaHostRegister(ptr, SIZE, 0);   // 返回 cudaErrorHostMemoryAlreadyRegistered
//
//   // ❌ 注册后忘记 Unregister 直接 free（内存泄漏 + CUDA 状态损坏）
//   cudaHostRegister(ptr, SIZE, 0);
//   free(ptr);                        // 应先 Unregister 再 free

// ─────────────────────────────────────────────────────────────────────
// 示例 5：与 cudaMallocHost 的对比
//
//   两种方式最终都得到 pinned memory，差异在于使用场景：
//
//   cudaMallocHost / cudaHostAlloc          cudaHostRegister
//   ─────────────────────────────────       ─────────────────────────────
//   从零开始分配，CUDA 全程管理             已有 pageable 内存，事后锁定
//   生命周期完全由 CUDA 控制               生命周期仍由原分配方式控制
//   释放：cudaFreeHost()                   解锁：cudaHostUnregister()
//                                           释放：free() / delete / munmap
//   适合：新写的 CUDA 代码                 适合：集成旧代码、第三方库、
//                                                  系统分配的缓冲区
// ─────────────────────────────────────────────────────────────────────
void demo_compare_with_cudaMallocHost(void)
{
    printf("\n─── 示例 5：cudaHostRegister vs cudaMallocHost 内存状态对比 ───\n");

    // 方式 A：cudaMallocHost（分配即 pinned）
    float *h_pinned = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_pinned, SIZE));
    printf("  [A] cudaMallocHost    ptr=%p  分配后立即是 pinned\n", (void *)h_pinned);

    // 方式 B：malloc → cudaHostRegister（分配后锁定）
    float *h_registered = (float *)malloc(SIZE);
    printf("  [B] malloc            ptr=%p  此时是 pageable\n", (void *)h_registered);
    CUDA_CHECK(cudaHostRegister(h_registered, SIZE, cudaHostRegisterDefault));
    printf("  [B] 注册后            ptr=%p  现在是 pinned（地址不变）\n", (void *)h_registered);

    // 两者对 cudaMemcpy 的效果相同：均走 DMA 直传
    float *d_a = NULL, *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, SIZE));
    CUDA_CHECK(cudaMalloc((void **)&d_b, SIZE));

    CUDA_CHECK(cudaMemcpy(d_a, h_pinned,     SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_registered, SIZE, cudaMemcpyHostToDevice));
    printf("  两者 cudaMemcpy 均走 DMA 直传，带宽相当\n");

    // 释放方式不同
    CUDA_CHECK(cudaFreeHost(h_pinned));          // A：cudaFreeHost 一步完成
    CUDA_CHECK(cudaHostUnregister(h_registered));// B：先解锁
    free(h_registered);                           // B：再释放

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    printf("  释放完成\n");
}

int main(void)
{
    demo_register_default();
    demo_register_mapped();
    demo_register_portable();
    demo_compare_with_cudaMallocHost();

    // ── 速查：flags 含义 ───────────────────
    //
    //   flags                        作用
    //   ────────────────    ──────────────────────
    //   cudaHostRegisterDefault      仅锁定（基本 pinned）
    //   cudaHostRegisterPortable     多 GPU / 多 context 可见
    //   cudaHostRegisterMapped       映射到 GPU 地址空间（零拷贝）
    //   cudaHostRegisterIoMemory     注册 I/O / BAR 内存（极少用）
    //   cudaHostRegisterReadOnly     只读映射，降低 GPU TLB 压力
    //
    // ── 速查：生命周期管理 ────────────────────────
    //
    //   分配               锁定                 解锁                   释放
    //   ──────────────     ─────────────────    ──────────────────     ──────
    //   malloc(size)   →   cudaHostRegister →   cudaHostUnregister →   free()
    //   new float[N]   →   cudaHostRegister →   cudaHostUnregister →   delete[]
    //   mmap(...)      →   cudaHostRegister →   cudaHostUnregister →   munmap()

    printf("\n全部示例完成。\n");
    return 0;
}
