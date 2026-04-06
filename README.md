# Programming GPUs in CUDA

CUDA 编程指南配套示例项目，覆盖统一内存、异步执行、SIMT kernel、内存模型等核心主题。

## 环境

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU（SM 8.6，Limited UM） |
| CUDA | 12.x |
| 编译器 | nvcc `/usr/local/cuda/bin/nvcc`，g++ 11.4.0 |
| CMake | 3.24+（需支持 `CMAKE_CUDA_ARCHITECTURES native`） |
| OS | Ubuntu / WSL2（Linux 6.6） |
| IDE | CLion |

## 构建

```bash
# CLion：打开项目后自动 configure，点击 Build 即可

# 命令行：
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug --target <目标名> -j$(nproc)
```

GPU 架构由 `nvidia-smi` 在 configure 阶段自动探测（`8.6` → `86`），
无需手动填写 SM 版本号。若 `nvidia-smi` 不可用则回退到 `native`。

## 项目结构

```
Programming GPUs in CUDA/
├── CMakeLists.txt
├── README.md
├── clion_external_libraries.md          CLion External Libraries 路径说明
│
├── Into to CUDA C++/                    基础入门
│   ├── vecAdd_explicitMemory.cu         向量加法（显式内存管理）
│   └── vecAdd_unifiedMemory.cu          向量加法（统一内存）
│
├── Writing CUDA SIMT kernel/            SIMT kernel 编写
│   ├── 3D conv.cu                       3D 卷积
│   ├── 3D conv cudaLaunchKernel.cu      cudaLaunchKernel 启动方式
│   ├── 3D conv cudaLaunchKernelEx.cu    cudaLaunchKernelEx 扩展启动
│   ├── Distributed Shared Memory.cu     分布式共享内存（cluster 级别）
│   ├── Dynamic Allocation of Shared Memory.cu  动态共享内存
│   ├── Global Memory.cu                 全局内存访问
│   ├── Global Memory device.cu          __device__ 全局内存
│   ├── Global Memory unified.cu         统一内存下的全局内存
│   ├── Shared Memory.cu                 共享内存基础
│   ├── atomatic.cu                      原子操作
│   ├── atomic_scope.cu                  原子操作作用域
│   ├── cluster.cu                       Thread Block Cluster
│   ├── constant memory.cu               常量内存
│   ├── cross_sector_and_bank.cu         bank conflict 与跨 sector 访问
│   ├── global_memory_coalescing.cu      全局内存合并访问
│   ├── matrix_transpose_with_GM.cu      矩阵转置（全局内存版）
│   ├── matrix_transpose_with_sm.cu      矩阵转置（共享内存优化版）
│   ├── multi_cluster.cu                 多 cluster 协作
│   └── shared_memory_bank_conflict.cu   共享内存 bank conflict 演示
│
├── Asynchronous Execution/              异步执行
│   ├── cuda_stream.cu                   CUDA Stream 基础
│   ├── cuda_event.cu                    CUDA Event 计时与同步
│   ├── cuda_graph.cu                    CUDA Graph 捕获与回放
│   ├── cuda_memcpy_async.cu             异步内存拷贝
│   ├── cuda_default_stream.cu           默认 stream 行为
│   ├── cuda_free_comparison.cu          cudaFree vs cudaFreeAsync 对比
│   ├── cuda_host_alloc_examples.cu      cudaHostAlloc 各 flag 演示
│   ├── cuda_host_func_data.cu           cudaLaunchHostFunc 数据传递
│   ├── cuda_host_get_device_pointer.cu  cudaHostGetDevicePointer
│   ├── cuda_host_register.cu            cudaHostRegister 注册已有内存
│   ├── cuda_launch_host_func.cu         在 stream 中插入 host 回调
│   ├── cuda_pdl.cu                      程序化依赖启动（PDL）
│   ├── cuda_pipeline_hostfunc.cu        pipeline + host func 组合
│   ├── hostfunc_unified_mem.cu          host func 与统一内存结合
│   └── ptr_init_test.cpp                指针初始化行为测试（纯 C++）
│
└── Unified and System Memory/           统一内存与系统内存
    ├── device_properties_um.cu          GPU UM 能力检测与属性查询
    ├── managed_alloc_compare.cu         三种 managed 内存分配方式对比
    │                                    cudaMallocManaged / cudaMallocFromPoolAsync / __managed__
    ├── peer_access.cu                   多 GPU Peer Access
    ├── pinned_host_kernel.cu            Pinned 内存直接传入 kernel 验证
    │                                    cudaMallocHost / cudaHostAlloc+Mapped
    ├── pointer_attributes.cu            cudaPointerGetAttributes 指针类型查询
    └── prefetch_and_advise.cu           cudaMemPrefetchAsync / cudaMemAdvise
```

## CMake 关键配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `CMAKE_CUDA_ARCHITECTURES` | nvidia-smi 动态探测 | 自动生成 `--generate-code=arch=compute_86,code=[compute_86,sm_86]` |
| `CMAKE_CUDA_RUNTIME_LIBRARY` | `Shared` | 动态链接 `libcudart.so`（`-lcudart`） |
| `CMAKE_CUDA_STANDARD` | `20` | `.cu` host 端使用 C++20 |
| `CMAKE_CUDA_FLAGS_DEBUG` | `-G -g` | Debug 模式生成设备端调试信息（cuda-gdb 断点必须） |
| `--keep --keep-dir` | per-target | 中间文件（.ptx/.cubin）保存至 `cmake-build-debug/temp/<目标名>/` |

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
