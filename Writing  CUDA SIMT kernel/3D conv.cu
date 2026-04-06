#include <cuda_runtime.h>
#include <stdio.h>

// 视频尺寸
#define WIDTH   128
#define HEIGHT  128
#define FRAMES  32

// 卷积核尺寸（3x3x3）
#define KW 3
#define KH 3
#define KF 3
#define K_RADIUS 1  // = KW/2 = KH/2 = KF/2

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t result = expr;                                      \
    if (result != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA Error: %s:%d = %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(result));    \
    }                                                               \
} while(0)

// 3x3x3 卷积核存放在 constant memory，所有线程广播读取
__constant__ float kernel3D[KF][KH][KW];

// 3D 卷积（视频处理）：每个线程处理 volume 中的一个体素
// output[frame][row][col] = sum over kernel3D * volume neighbors
__global__ void Conv3D(const float *volume, float *output,
                       int width, int height, int frames)
{
    int col   = blockIdx.x * blockDim.x + threadIdx.x;  // x = 列
    int row   = blockIdx.y * blockDim.y + threadIdx.y;  // y = 行
    int frame = blockIdx.z * blockDim.z + threadIdx.z;  // z = 帧

    // 边界检查
    if (col >= width || row >= height || frame >= frames) return;

    float sum = 0.0f;

    for (int kf = 0; kf < KF; kf++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int f = frame + kf - K_RADIUS; // K_RADIUS=1, kf∈{0,1,2} → 偏移 {-1, 0, +1}
                int r = row   + kh - K_RADIUS; // 同上  
                int c = col   + kw - K_RADIUS; // 同上
                //   以当前体素 (frame, row, col) 为中心，向周围扩展 ±1：

                // 边界处理：zero padding
                // if (f >= 0 && f < frames && r >= 0 && r < height && c >= 0 && c < width)
                // 边界外的邻居不累加（等价于视为 0），内部体素 27 个邻居全部有效，角落体素只有 8 个有效邻居。
                if (f >= 0 && f < frames &&
                    r >= 0 && r < height &&
                    c >= 0 && c < width)
                {
                    int vol_idx = f * height * width   // 第 f 帧的起始偏移
                                + r * width            // 第 r 行的偏移
                                + c;                   // 第 c 列
                    //  均值滤波时 kernel3D 全为 1/27，内部体素结果 = 27个1 × 1/27 = 1.0。
                    sum += kernel3D[kf][kh][kw] * volume[vol_idx];
                    // Thread(0,0,0): idx = frame*H*W + row*W + 0
                    // Thread(1,0,0): idx = frame*H*W + row*W + 1  ← 差1，连续
                    // Thread(2,0,0): idx = frame*H*W + row*W + 2
                    // ...
                    // → 32个线程的访问可以合并成1次内存事务
                }
            }
        }
    }
    // 完整卷积公式：
    // output[frame][row][col] = Σ kernel3D[kf][kh][kw] * volume[f][r][c]
    //                             kf,kh,kw ∈ {0,1,2}

    // 每个线程对应唯一的 (col, row, frame)，直接赋值，不依赖原始值
    int out_idx = frame * height * width + row * width + col;
    // Thread(0,0,0): idx = frame*H*W + row*W + 0
    // Thread(1,0,0): idx = frame*H*W + row*W + 1  ← 差1，连续
    // Thread(2,0,0): idx = frame*H*W + row*W + 2
    // ...
    // → 32个线程的访问可以合并成1次内存事务
    output[out_idx] = sum;
}

int main(int argc, char **argv)
{
    const int total = WIDTH * HEIGHT * FRAMES;

    // Host 数据
    float *h_volume = new float[total];  // 不初始化，因为后面立刻赋值为 1.0f
    float *h_output = new float[total] {};  // 先零初始化，再调用隐式默认构造函数 


    // 因为 total = WIDTH * HEIGHT * FRAMES = 128 * 128 * 32 = 524288 个 float，共 2MB。                                                                                                                                                                                                                                                                                                                                                                                   
    // 栈空间通常只有 1~8MB，直接在栈上声明会栈溢出：                                                                                                                                                                                        
    // float h_volume[524288];  // 2MB，直接放栈上 → stack overflow 风险                                                                                                                                                                     
    // 堆没有这个限制，受系统物理内存约束，放 GB 级数据都没问题：
    // float *h_volume = new float[total];  // 2MB 放堆上，安全
    // 初始化输入：全 1
    for (int i = 0; i < total; i++) h_volume[i] = 1.0f;

    // 初始化卷积核：均值滤波（所有权重 = 1/27）
    float h_kernel[KF][KH][KW] {};
    for (int i = 0; i < KF * KH * KW; i++)
        ((float*)h_kernel)[i] = 1.0f / (KF * KH * KW);
    // h_kernel 是三维数组，但在内存中是连续排列的：                                                                                                                                                                                         
                                                                                                                                                                                                                                        
    // h_kernel[0][0][0], h_kernel[0][0][1], h_kernel[0][0][2],                                                                                                                                                                              
    // h_kernel[0][1][0], h_kernel[0][1][1], ...
    // h_kernel[2][2][2]                                                                                                                                                                                                                     
                    
    // 共 KF*KH*KW = 27 个 float，物理上就是一块连续的内存。

    // (float*)h_kernel 的作用：

    // (float*)h_kernel   // 把 float[3][3][3]* 强转为 float*
    //                     // 指向同一块内存的首地址

    // 强转之后就变成了一维 float*，用 [i] 线性索引就能遍历全部 27 个元素，等价于：

    // // 等价的三重循环写法
    // for (int f = 0; f < KF; f++)
    //     for (int h = 0; h < KH; h++)
    //         for (int w = 0; w < KW; w++)
    //             h_kernel[f][h][w] = 1.0f / (KF * KH * KW);

    // C/C++ 中多维数组保证内存连续，所以这种强转是合法的。


    // 多维数组的退化只退化一层：                                                                                                                                                                                                      
                                                                                                                                                                                                                                        
    // float h_kernel[3][3][3];                                                                                                                                                                                                              
                                                                                                                                                                                                                                            
    // // 退化为：                                                                                                                                                                                                                           
    // float (*)[3][3]   // 指向二维数组的指针，不是 float*

    // 所以 h_kernel[i] 拿到的是第 i 个 float[3][3]，而不是第 i 个 float，无法线性索引所有 27 个元素。

    // 必须显式强转才能变成 float*：

    // ((float*)h_kernel)[i]  // 才能线性访问所有 27 个 float

    // 对比：
    // h_kernel[i]          // float[3][3]，第 i 帧的二维切片
    // ((float*)h_kernel)[i] // float，第 i 个元素

    // 一维数组 float arr[N] 才会退化为 float*，多维不会一路退化到底。
    

    // 拷贝卷积核到 constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(kernel3D, h_kernel, sizeof(h_kernel)));

    // 分配 device global memory
    float *d_volume {};
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_volume, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_volume, h_volume, total * sizeof(float), cudaMemcpyHostToDevice));

    // 3D 卷积（视频处理）：x=列, y=行, z=帧
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (WIDTH  + BLOCK_X - 1) / BLOCK_X,   // 16
        (HEIGHT + BLOCK_Y - 1) / BLOCK_Y,   // 16
        (FRAMES + BLOCK_Z - 1) / BLOCK_Z    //  8
    );

    //   两者维度数相同（都是3D）只是因为问题本身是3D的，需要三个维度来覆盖整个 volume。

        // grid 和 block 的关系：

        // ┌───────┬───────────────────────┬─────────────────────────────┐
        // │       │         含义          │            约束             │
        // ├───────┼───────────────────────┼─────────────────────────────┤
        // │ block │ 每个 block 内的线程数 │ 三维乘积 ≤ 1024（硬件限制） │
        // ├───────┼───────────────────────┼─────────────────────────────┤
        // │ grid  │ block 的排列数量      │ 覆盖整个数据空间即可        │
        // └───────┴───────────────────────┴─────────────────────────────┘

        // 两者维度可以完全不同，例如完全合法的写法：

        // dim3 block(256);        // 1D block
        // dim3 grid(16, 16, 8);  // 3D grid

        // 唯一的要求是：grid × block 覆盖的线程总数能处理所有数据，不够就漏算，超出就靠边界检查过滤：
    
    Conv3D<<<grid, block>>>(d_volume, d_output, WIDTH, HEIGHT, FRAMES);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证：内部体素（无边界效应）均值滤波结果应为 1.0
    printf("Center voxel output: %.4f (expected ~1.0000)\n",
           h_output[16 * HEIGHT * WIDTH + 64 * WIDTH + 64]);

    // 打印前 4 帧第 0 行第 0 列（边界体素，zero padding 影响结果）
    printf("Corner voxels (frame 0~3, row=0, col=0):\n");
    for (int f = 0; f < 4; f++)
        printf("  [frame=%d] = %.4f\n", f, h_output[f * HEIGHT * WIDTH]);

    cudaFree(d_volume);
    cudaFree(d_output);
    delete[] h_volume;
    delete[] h_output;

    return 0;
}
