#include <cuda_runtime.h>
#include <stdio.h>

#define N 10

// constant memory: read-only, cached, broadcast to all threads
__constant__ float coeffs[4];

__global__ void compute(float *out) {
    int idx = threadIdx.x;
    // use all 4 coefficients: out[i] = a*i^3 + b*i^2 + c*i + d
    out[idx] = coeffs[0] * idx * idx * idx  // 是先从constant mem读到constant cache（同一 warp 内所有线程访问 coeffs[0] 是同一地址，constant cache 一次读取广播给全部 32个线程，无需多次访问，效率很高。），constant cache在sm上，在寄存器中计算，然后再从寄存器写到l1 cache再写到l2 cache，直接写到HBM
             + coeffs[1] * idx * idx
             + coeffs[2] * idx
             + coeffs[3]; 
}

int main(int argc, char** argv)
{
    float h_coeffs[4] {1.0f, 2.0f, 3.0f, 4.0f};
    float h_out[N] {0};
    float* device_out = nullptr;

    cudaMalloc(&device_out, N * sizeof(float));  // GPU 显存（device global memory），尚未初始化
//   cudaMalloc 有模板包装：                                                                                                                                                                          
//   // cuda_runtime.h 内部定义                                                                                                                                                                       
//   template<class T>                                                                                                                                                                                
//   cudaError_t cudaMalloc(T **devPtr, size_t size) {
//       return ::cudaMalloc((void**)(void*)devPtr, size);
//   }
//   所以传 float** 时，模板推导出 T=float，自动处理类型，无需强转。

    // copy coefficients to constant memory
    cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
    // 普通内存（pageable）→ cudaMemcpyToSymbol 时，CUDA runtime 内部会：                                                                                                                               
    // 1. 临时分配一块 pinned buffer
    // 2. 先把数据从 pageable → pinned buffer                                                                                                                                                           
    // 3. 再 DMA 传输 pinned buffer → GPU



    compute<<<1, N>>>(device_out);

    // copy result back to host
    cudaMemcpy(h_out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);  
    // 用 cudaMemcpyDefault 可以自动判断，但有条件：只在使用 Unified Memory（cudaMallocManaged） 的指针时可靠，普通 cudaMalloc 分配的指针用 cudaMemcpyDefault 行为未定义，可能出错
    // cudaMemcpy(h_out, device_out, N * sizeof(float), cudaMemcpyDefault);

    for (int i = 0; i < N; i++) {
        printf("out[%d] = %.1f\n", i, h_out[i]);
    }


    // 锁页内存写法
    float* hp_coeffs {};
    cudaMallocHost((void**)&hp_coeffs, 4 * sizeof(float));

    // cudaMallocHost 只有 C 风格声明：
    // cudaError_t cudaMallocHost(void **ptr, size_t size);
    // 没有模板包装，而 C++ 中 float** → void** 不是隐式转换（注意：float* → void* 可以隐式转，但多一层指针就不行），所以必须显式强转。


    hp_coeffs[0] = 1.0f; hp_coeffs[1] = 2.0f;
    hp_coeffs[2] = 3.0f; hp_coeffs[3] = 4.0f;
    cudaMemcpyToSymbol(coeffs, hp_coeffs, 4 * sizeof(float));
    
//       什么时候用哪种：

//   ┌─────────────────────────────────────┬──────────┬───────────────────────────────┐
//   │                场景                 │   选择   │             原因              │
//   ├─────────────────────────────────────┼──────────┼───────────────────────────────┤
//   │ 小数据、只传一次（如本例 16 bytes） │ 普通内存 │ 开销小，pinned 分配本身有成本 │
//   ├─────────────────────────────────────┼──────────┼───────────────────────────────┤
//   │ 大数据、反复传输                    │ 锁页内存 │ 省去每次的中间拷贝，带宽更高  │
//   ├─────────────────────────────────────┼──────────┼───────────────────────────────┤
//   │ 使用 cudaMemcpyAsync + Stream       │ 锁页内存 │ 必须用，否则 async 退化为同步 │
//   ├─────────────────────────────────────┼──────────┼───────────────────────────────┤
//   │ CPU/GPU 重叠计算（双缓冲）          │ 锁页内存 │ 实现真正的 overlap            │
//   └─────────────────────────────────────┴──────────┴───────────────────────────────┘
// 
//   本例 coeffs[4] 只有 16 字节且只传一次，用普通内存完全合理，改成锁页内存反而增加代码复杂度，得不偿失。

    compute<<<1, N>>>(device_out);

    // copy result back to host
    cudaMemcpy(h_out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);  
    // 用 cudaMemcpyDefault 可以自动判断，但有条件：只在使用 Unified Memory（cudaMallocManaged） 的指针时可靠，普通 cudaMalloc 分配的指针用 cudaMemcpyDefault 行为未定义，可能出错
    // cudaMemcpy(h_out, device_out, N * sizeof(float), cudaMemcpyDefault);

    for (int i = 0; i < N; i++) {
        printf("out[%d] = %.1f\n", i, h_out[i]); 
        printf("完成");
    }

    // 常用格式说明：  
                                                                                                                                                                
    // ┌──────┬───────────────────────┬──────────┐
    // │ 格式 │         含义          │ 示例输出 │
    // ├──────┼───────────────────────┼──────────┤
    // │ %d   │ 整数                  │ 42       │
    // ├──────┼───────────────────────┼──────────┤
    // │ %f   │ 浮点数（默认6位小数） │ 3.140000 │
    // ├──────┼───────────────────────┼──────────┤
    // │ %.1f │ 浮点数，1位小数       │ 3.1      │
    // ├──────┼───────────────────────┼──────────┤
    // │ %.3f │ 浮点数，3位小数       │ 3.140    │
    // ├──────┼───────────────────────┼──────────┤
    // │ %e   │ 科学计数法            │ 3.14e+00 │
    // ├──────┼───────────────────────┼──────────┤
    // │ %s   │ 字符串                │ hello    │
    // ├──────┼───────────────────────┼──────────┤
    // │ %zu  │ size_t 类型           │ 1024     │
    // └──────┴───────────────────────┴──────────┘

    cudaFree(device_out);
    cudaFreeHost(h_coeffs);  // 锁页内存用完需手动释放
    return 0;
}
