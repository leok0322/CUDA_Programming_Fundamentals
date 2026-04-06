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

    int width = WIDTH, height = HEIGHT, frames = FRAMES;

    // cudaLaunchKernelEx：通过 cudaLaunchConfig_t 支持扩展属性
    // 属性：设置 launch 优先级（高优先级先被调度，所有 GPU 均支持）
    // ClusterDimension 属性仅 Hopper H100+ 支持，此处不使用以保证兼容性
    cudaLaunchAttribute attrs[1] {};
    attrs[0].id           = cudaLaunchAttributePriority;
    attrs[0].val.priority = 1;  // 0=普通, 正值=高优先级

    cudaLaunchConfig_t config  {};
    config.gridDim          = grid;
    config.blockDim         = block;
    config.dynamicSmemBytes = 0;       // 无动态 shared memory
    config.stream           = nullptr; // 默认流
    config.attrs            = attrs;
    config.numAttrs         = 1;  // attrs 数组的元素个数

    // 模板版本：直接传实际参数值，编译器自动推导类型，无需 void*[] 数组
    CUDA_CHECK(cudaLaunchKernelEx(
        &config,                        // 扩展配置
        Conv3D,                         // kernel 函数指针，类型：void(*)(const float*, float*, int, int, int)
        d_volume, d_output,             // float* 参数
        width, height, frames           // int 参数
    ));

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
