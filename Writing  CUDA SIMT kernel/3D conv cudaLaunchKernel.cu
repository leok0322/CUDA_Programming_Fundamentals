#include <cuda_runtime.h>
#include <stdio.h>

#define WIDTH   128
#define HEIGHT  128
#define FRAMES  32

#define KW 3
#define KH 3
#define KF 3
#define K_RADIUS 1

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

__constant__ float kernel3D[KF][KH][KW];

__global__ void Conv3D(const float *volume, float *output,
                       int width, int height, int frames)
{
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z * blockDim.z + threadIdx.z;

    if (col >= width || row >= height || frame >= frames) return;

    float sum = 0.0f;
    for (int kf = 0; kf < KF; kf++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int f = frame + kf - K_RADIUS;
                int r = row   + kh - K_RADIUS;
                int c = col   + kw - K_RADIUS;
                if (f >= 0 && f < frames &&
                    r >= 0 && r < height &&
                    c >= 0 && c < width)
                {
                    int vol_idx = f * height * width + r * width + c;
                    sum += kernel3D[kf][kh][kw] * volume[vol_idx];
                }
            }
        }
    }

    int out_idx = frame * height * width + row * width + col;
    output[out_idx] = sum;
}

int main(int argc, char **argv)
{
    const int total = WIDTH * HEIGHT * FRAMES;

    float *h_volume = new float[total];
    float *h_output = new float[total]{};

    for (int i = 0; i < total; i++) h_volume[i] = 1.0f;

    float h_kernel[KF][KH][KW]{};
    for (int i = 0; i < KF * KH * KW; i++)
        ((float*)h_kernel)[i] = 1.0f / (KF * KH * KW);

    CUDA_CHECK(cudaMemcpyToSymbol(kernel3D, h_kernel, sizeof(h_kernel)));

    float *d_volume = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_volume, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_volume, h_volume, total * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 grid(
        (WIDTH  + BLOCK_X - 1) / BLOCK_X,
        (HEIGHT + BLOCK_Y - 1) / BLOCK_Y,
        (FRAMES + BLOCK_Z - 1) / BLOCK_Z
    );

    // cudaLaunchKernel 替代 <<<>>> 语法：
    // 参数通过 void* 数组传递，每个元素是对应参数的地址
    int width = WIDTH, height = HEIGHT, frames = FRAMES;
    void* args[] { &d_volume, &d_output, &width, &height, &frames };  // 表示元素是指针的数组
    // 每个元素的类型：                                                                                                                                                                                                                      
                                                                                                                                                                                                                                        
    //     &d_volume  // float** （d_volume 是 float*，取地址得 float**）                                                                                                                                                                        
    //     &d_output  // float**
    //     &width     // int*
    //     &height    // int*
    //     &frames    // int*
    //   这些不同类型的指针存入 void*[] 时，隐式转换为 void*（C++ 中任意指针类型都可以隐式转为 void*）。

    // 编译期：nvcc 为每个 kernel 生成参数描述符                                                                                                                                                                                             
    // Conv3D 的参数表：
    //     param[0]: size=8 bytes  (float* 指针，64位)                                                                                                                                                                                         
    //     param[1]: size=8 bytes  (float* 指针，64位)
    //     param[2]: size=4 bytes  (int)
    //     param[3]: size=4 bytes  (int)
    //     param[4]: size=4 bytes  (int)

    // 运行期：cudaLaunchKernel 只做内存拷贝
    // // 伪代码，内部实际行为：
    // for (int i = 0; i < num_params; i++) {
    //     memcpy(kernel_param_buffer + offset[i],
    //             args[i],               // void* 指向参数地址
    //             param_size[i]);        // 从元数据得知拷贝多少字节
    // }

    // 具体到每个参数：
    // args[0] = &d_volume → 从该地址读 8 字节 → 得到 d_volume 的值（float*）
    // args[2] = &width    → 从该地址读 4 字节 → 得到 width 的值（int）



    CUDA_CHECK(cudaLaunchKernel(
        (const void *)Conv3D,  // kernel 函数指针
        grid,                  // grid 维度
        block,                 // block 维度
        args,                  // 参数数组
        0,                     // 动态 shared memory 大小（字节）
        nullptr                // stream（nullptr = 默认流）
    ));

    // cudaLaunchKernel 的签名：
    // cudaError_t cudaLaunchKernel(const void *func, ...);
    //                             ↑ 接收 void*，需要显式强转函数指针
    // float *p;                                                                                                                                                                                                                             
    // void *vp = p;          // ✓ 对象指针可以隐式转为 void*

    // void (*fp)(int);
    // void *vp2 = fp;        // ✗ 函数指针不能隐式转为 void*，编译报错
    // void *vp3 = (void*)fp; // ✓ 必须显式强转

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Center voxel output: %.4f (expected ~1.0000)\n",
           h_output[16 * HEIGHT * WIDTH + 64 * WIDTH + 64]);

    printf("Corner voxels (frame 0~3, row=0, col=0):\n");
    for (int f = 0; f < 4; f++)
        printf("  [frame=%d] = %.4f\n", f, h_output[f * HEIGHT * WIDTH]);

    cudaFree(d_volume);
    cudaFree(d_output);
    delete[] h_volume;
    delete[] h_output;

    return 0;
}
