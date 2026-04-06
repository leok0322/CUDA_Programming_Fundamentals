/*
 * cross_sector_and_bank.cu
 *
 * 演示两种跨越问题：
 *   Part 1: HBM 跨 Sector —— 单个线程的数据横跨两个 32-byte sector 边界
 *   Part 2: Shared Memory 跨 Bank —— 单个线程的数据横跨两个 bank
 *
 * 编译：nvcc -O2 -arch=sm_80 cross_sector_and_bank.cu -o cross
 * 分析：ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,
 *                     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
 *             ./cross
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <functional>   // std::function

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _e = (expr);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error %s:%d : %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

#define N      (1 << 20)
#define NWARM  5
#define NREP   100

// ═════════════════════════════════════════════════════════════════════════════
// Part 1: HBM 跨 Sector
//
// sector 边界：每 32 bytes 一个
//   [0x00~0x1F] [0x20~0x3F] [0x40~0x5F] ...
//
// 跨 sector 触发条件：
//   数据起始地址不是数据大小的倍数，导致数据横跨两个 sector
//
// 例：double (8B) 起始地址 = 0x1C（28）
//   [0x1C~0x1F] → Sector 0 的最后 4 bytes
//   [0x20~0x23] → Sector 1 的前   4 bytes
//   → 需要 2 次 transaction 才能读完这 1 个 double
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1a: 对齐的 double 访问（不跨 sector）
//
//   double 数组起始地址天然 8B 对齐
//   每个 double 完整落在一个 sector 内（或跨 sector 但对齐，不产生额外开销）
//
//   内存布局（每个 double = 8 bytes）：
//   Sector 0 [0x00~0x1F]: d[0]d[1]d[2]d[3]  （4个double = 32bytes）
//   Sector 1 [0x20~0x3F]: d[4]d[5]d[6]d[7]CUDA_CHECK(cudaMalloc(&d_char_raw,   N * sizeof(double) + 8));  // 额外 8B 供偏移，
//   → 每个 double 完整落在 sector 内 ✓
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_aligned_double(const double* __restrict__ in,
                                       double* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] * 2.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1b: 未对齐的 double 访问（跨 sector）
//
//   通过偏移 1 个 char（1 byte）使 double 起始地址变为奇数，
//   强制每个 double 横跨两个 sector：
//
//   base 地址 = 0x0001（偏移 1 byte）
//   d[0] 起始 = 0x0001, 结束 = 0x0008
//              ↑ 跨越了 sector 边界 0x0020/0x0040 等
//
//   实际内存布局：
//   Sector 0 [0x00~0x1F]: [pad][d0前7B][d1前1B]...
//   Sector 1 [0x20~0x3F]: [d1后7B][d2前1B]...
//   → 每个 double 被 sector 边界切割 → 需要 2 次 transaction ✗
//
//   注意：用 char* 偏移再 memcpy 读取，避免编译器对齐优化
// ─────────────────────────────────────────────────────────────────────────────

// ● __builtin_memcpy 是 GCC/Clang 的内置版本的 memcpy，语义完全相同：                                                                                                                                                                                                   
//   __builtin_memcpy(dst, src, size);                                                                                                                                                                                                                                         
//   // 从 src 地址拷贝 size 个字节到 dst 地址                                                                                                                                                                                                                   
//   ---                                                                                                                                                                                                                                                                     
//   逐项解析：

//   __builtin_memcpy(
//       &val,                              // dst：目标地址，val 变量的地址
//       in_raw + 1 + tid * sizeof(double), // src：源地址
//       sizeof(double)                     // size：拷贝 8 个字节
//   );

//   ---
//   in_raw + 1 + tid * sizeof(double) 的含义：

//   in_raw                    // char*，显存基地址
//   in_raw + 1                // 偏移 1 byte，使地址不再对齐
//   in_raw + 1 + tid * 8      // 再偏移 tid * 8，跳到第 tid 个 double 的位置

//   // tid=0 → in_raw + 1        （从 byte 1 读 8 bytes）
//   // tid=1 → in_raw + 9        （从 byte 9 读 8 bytes）
//   // tid=2 → in_raw + 17       （从 byte 17 读 8 bytes）

//   每个 double 的起始地址都是奇数（不是 8 的倍数），强制跨 sector。

//   ---
//   为什么用 __builtin_memcpy 而不是直接赋值：

//   // 直接赋值：编译器看到 double* 会假设地址是 8B 对齐
//   // 生成对齐读取指令，在未对齐地址上执行会崩溃或静默产生错误结果
//   double val = *(double*)(in_raw + 1 + tid * sizeof(double));  // ✗ 危险

//   // __builtin_memcpy：按字节拷贝，不假设对齐
//   // 编译器生成不依赖对齐的字节读取指令
//   __builtin_memcpy(&val, in_raw + 1 + tid * sizeof(double), sizeof(double));  // ✓

//   本质：绕过编译器的对齐假设，强制产生未对齐的内存访问，才能演示跨 sector 的效果。

__global__ void kernel_unaligned_double(const char* __restrict__ in_raw,
                                         double* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 从偏移 1 byte 的地址读取 double，强制跨 sector
        double val;
        // 用 memcpy 绕过编译器的对齐检查

        __builtin_memcpy(&val, in_raw + 1 + tid * sizeof(double), sizeof(double));
        out[tid] = val * 2.0;
    }
}


//   ---                                                    
//   对齐情况（sector 利用率 100%）：                                                                                                                                                                                                                           
//   Sector 0: [d0][d1][d2][d3]  → 4 个完整 double，32/32 bytes 有用                       
//   Sector 1: [d4][d5][d6][d7]  → 4 个完整 double
//   每个 sector 装 4 个 double，利用率 100%

//   ---
//   偏移 1 byte 后（跨 sector）：

//   byte:  0  1     8     16    24   31 32    39    47    55   63 64
//  [pad][d0    ][d1    ][d2    ][d3 ←切→d3][d4    ][d5    ][d6    ][d7←切
//   |←────── Sector 0 (32B) ─────────→|←────── Sector 1 (32B) ──────

//   - Sector 0：d0、d1、d2 完整（24 bytes），d3 的前 7 bytes
//   - Sector 1：d3 的后 1 byte，d4、d5、d6 完整，d7 的前 7 bytes
//   - 每个 sector 只有 3 个完整 double = 24 bytes 有用，利用率 = 24/32 = 75%

//   ---
//   warp 层面的代价（32 线程，每线程 1 个 double）：

//   对齐：32 × 8 = 256 bytes，需要 256/32 = 8 个 sector
//   偏移：数据从 byte1 到 byte256，跨 9 个 sector（多 1 个）
//   利用率 = 256 / (9×32) = 256/288 ≈ 88.9%


//   ---
//   如果每个线程偏移不同（更乱的情况）：

//   // 每个线程偏移量不同
//   in_raw + tid * sizeof(double) + (tid % 3)  // 各线程偏移 0、1、2 字节

//   这种情况才会导致每个线程都跨 sector，最坏情况每个线程需要 2 个 sector，warp 总共需要 最多 64 个 sector，利用率极低。

//   ---
//   本例的关键：偏移量对所有线程一致（都是 +1 byte），所以整个 warp 只整体偏移了一个 sector 边界，代价只是多 1 个 sector。





// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1c: 结构体未对齐导致跨 sector（最常见的实际场景）
//
//   典型错误：结构体内有 char + float，char 导致 float 地址偏移
//
//   struct BadStruct {
//       char  flag;    // 1 byte, offset 0
//       // 编译器默认插入 3 bytes padding 使 float 对齐
//       float value;   // 4 bytes, offset 4（对齐后）
//   };
//
//   用 __attribute__((packed)) 禁止 padding，强制 float 从 offset 1 开始：
//   struct PackedStruct {
//       char  flag;    // offset 0
//       float value;   // offset 1 ← 不是 4B 对齐！可能跨 sector
//   };
// ─────────────────────────────────────────────────────────────────────────────




// ● __align__(1) 和 __attribute__((packed)) 都是 GCC/Clang 编译器扩展，不是 C++ 标准语法。                                                                                                                                                                                    

//   ---                                                                                                                                                                                                                                                                       
//   __align__(1) 的含义：
                                                                                                                                                                                                                                                                            
//   指定结构体的对齐要求为 1 字节，即结构体起始地址只需是 1 的倍数（任意地址都满足）。

//   // 默认情况：结构体对齐 = 最大成员的对齐要求
//   struct Normal {
//       char  flag;   // 1B
//       float value;  // 4B → 结构体对齐 = 4B
//   };
//   // sizeof = 8（编译器插入 3B padding）

//   // __align__(1)：强制对齐为 1B
//   struct __align__(1) Packed {
//       char  flag;
//       float value;
//   };
//   // 结构体起始地址可以是任意地址

//   ---
//   __attribute__((packed)) 的含义：

//   禁止编译器在成员之间插入 padding，成员紧密排列：

//   struct __attribute__((packed)) Packed {
//       char  flag;   // offset 0
//       float value;  // offset 1（紧跟 char，不插 padding）
//   };
//   // sizeof = 5（1 + 4），而非默认的 8

//   ---
//   两者的关系：

//   struct __align__(1) PackedStruct {
//       char  flag;
//       float value;
//   } __attribute__((packed));

//   __align__(1) 和 __attribute__((packed)) 在这里效果重叠，都是为了去掉 padding，写两个是冗余的，保留 __attribute__((packed)) 即可。

struct __align__(1) PackedStruct {
    char  flag;
    float value;   // 起始地址 = base + 1，不是 4B 对齐
} __attribute__((packed));


struct AlignedStruct {
    char  flag;
    // 编译器自动插入 3 bytes padding
    float value;   // 起始地址 = base + 4，4B 对齐 ✓
};



__global__ void kernel_packed_struct(const PackedStruct* __restrict__ in,
                                      float* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // value 在 PackedStruct 内偏移 1 byte，未对齐，可能跨 sector
        out[tid] = in[tid].value * 2.0f;
    }
}

__global__ void kernel_aligned_struct(const AlignedStruct* __restrict__ in,
                                       float* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // value 在 AlignedStruct 内偏移 4 bytes，4B 对齐，不跨 sector ✓
        out[tid] = in[tid].value * 2.0f;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Part 2: Shared Memory 跨 Bank
//
// bank 边界：每 4 bytes 一个 bank，共 32 个 bank
//   bank_id = (byte_address / 4) % 32
//
// 跨 bank 触发条件：
//   数据大小 > 4 bytes 且起始地址不是数据大小的倍数
//   → 数据横跨两个 bank
//
// 例：double (8B) 起始地址 = 0x04（bank 1）
//   [0x04~0x07] → bank 1
//   [0x08~0x0B] → bank 2
//   → 需要访问 bank 1 和 bank 2 各一次再拼合
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2a: 对齐的 double 访问 shared memory（不跨 bank）
//
//   double (8B) 对齐到 8B → 起始地址是 8 的倍数
//   bank_id(高4B) = bank_id(低4B) + 1 → 跨 2 个连续 bank（合法，无冲突）
//
//   smem[0]: addr 0x00 → bank 0(0x00~0x03) + bank 1(0x04~0x07) ← 连续2bank
//   smem[1]: addr 0x08 → bank 2(0x08~0x0B) + bank 3(0x0C~0x0F) ← 连续2bank
//   T0→bank(0,1), T1→bank(2,3), ... T15→bank(30,31) → 无冲突 ✓
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_smem_aligned_double(const double* __restrict__ g_in,
                                            double* __restrict__ g_out, int n)
{
    // double 数组：每个 double 8B，天然 8B 对齐
    __shared__ double smem[256];   // 256 × 8B = 2048B

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // T_i → smem[i]，stride-1，每个 double 跨 2 个连续 bank
    // T0 →bank( 0, 1), T1 →bank( 2, 3), ... T15→bank(30,31)
    // T16→bank( 0, 1), T17→bank( 2, 3), ... T31→bank(30,31) ← 和 T0~T15 撞上！
    // 32 线程 × 2 bank = 64 次 bank 访问，只有 32 个 bank → 每 bank 被访问 2 次
    // → 2-way conflict（T0 和 T16 撞 bank0/1，T1 和 T17 撞 bank2/3，...）
    if (gid < n)
        g_out[gid] = smem[tid] * 2.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2b: 未对齐的 double 访问 shared memory（跨 bank，产生冲突）
//
//   强制偏移 4 bytes，使 double 从奇数 bank 开始：
//
//   偏移前（对齐）：       偏移后（未对齐，+4 bytes）：
//   smem[0]: bank 0,1     偏移后 smem[0]: bank 1,2
//   smem[1]: bank 2,3     偏移后 smem[1]: bank 3,4
//   ...                   ...
//
//   T0 → bank(1,2), T1 → bank(3,4) ... → 还是不冲突，但跨了奇偶 bank
//
//   极端情况（同一 bank 的不同 4B 段）：
//   smem_char 偏移 4B 后 double 的高低各 4B 落在非连续的 bank
//   在特定 stride 下会和其他线程形成冲突
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_smem_unaligned_double(const double* __restrict__ g_in,
                                              double* __restrict__ g_out, int n)
{
    // 用 char 数组，手动偏移 8 bytes 改变起始 bank
    // 256 个 double（256×8=2048B）+ 偏移 8B，最后一个 double[255] 结束 byte=8+255×8+7=2055
    // smem_raw 需要至少 2056 字节：2048+8 = 2056
    // 注意：偏移必须是 8 的倍数（double 的对齐要求），sm_75+ 对 shared memory 中 8B 访问
    //       严格要求 8B 对齐，非 8B 对齐会触发 misaligned address 硬件错误。
    __shared__ char smem_raw[2048 + 8];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 偏移 8 bytes：起始 bank 从 0 变为 2（而非对齐版本的 bank 0）
    // 例：smem_raw + 8 → 地址 0x08 → bank 2（0x08/4%32=2）
    double* smem = (double*)(smem_raw + 8);

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // 偏移后：T0→bank(2,3), T1→bank(4,5), ... T15→bank(0,1)
    //         T16→bank(2,3), T17→bank(4,5), ... T31→bank(0,1) ← 和 T0~T15 撞上！
    // 冲突模式与对齐版本完全相同，都是 2-way conflict
    // 原因：偏移只改变了起始 bank，不改变 32线程×2bank=64次访问 vs 32个bank 的根本矛盾
    if (gid < n)
        g_out[gid] = smem[tid] * 2.0;
}


//  ---                                                                                                                                                                                                                                                                       
//   kernel_smem_aligned_double（32线程访问 double）：                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                            
//   T0  → smem[0]  → bank 0, 1   (byte 0~7)                                                                                                                                                                                                                                   
//   T1  → smem[1]  → bank 2, 3   (byte 8~15)                                                                                                                                                                                                                                
//   ...                                                                                                                                                                                                                                                                       
//   T15 → smem[15] → bank 30, 31 (byte 120~127)                                                                                                                                                                                                                             
//   T16 → smem[16] → bank 0, 1   (byte 128~135) ← 和 T0 同 bank！
//   T17 → smem[17] → bank 2, 3   (byte 136~143) ← 和 T1 同 bank！
//   ...
//   T31 → smem[31] → bank 30, 31 ← 和 T15 同 bank！

//   32 个 bank 循环，32 个线程每个占 2 个 bank，T0~T15 和 T16~T31 撞上 → 2-way conflict。

//   ---
//   kernel_smem_unaligned_double（偏移 4 bytes）：

//   T0  → bank 1, 2
//   T1  → bank 3, 4
//   ...
//   T15 → bank 31, 0  (wrap)
//   T16 → bank 1, 2   ← 和 T0 同 bank！
//   ...

//   同样是 2-way conflict，模式完全一样。




// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2c: char 数组导致的跨 bank（最典型场景）
//
//   4 个 char 共享一个 bank（每个 bank 4 bytes）：
//
//   smem_char[0] → addr 0x00, bank 0
//   smem_char[1] → addr 0x01, bank 0  ← 同 bank！
//   smem_char[2] → addr 0x02, bank 0  ← 同 bank！
//   smem_char[3] → addr 0x03, bank 0  ← 同 bank！
//   smem_char[4] → addr 0x04, bank 1
//   ...
//
//   T0 → smem_char[0] → bank 0
//   T1 → smem_char[1] → bank 0 ← 4-way conflict！
//   T2 → smem_char[2] → bank 0 ← 4-way conflict！
//   T3 → smem_char[3] → bank 0 ← 4-way conflict！
//   T4 → smem_char[4] → bank 1
//   ...
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_smem_char_conflict(const char* __restrict__ g_in,
                                           int* __restrict__ g_out, int n)
{
    __shared__ char smem_char[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem_char[tid] = g_in[gid];
    __syncthreads();

    // 每 4 个连续线程打到同一个 bank → 4-way conflict
    // T0,T1,T2,T3 → bank 0
    // T4,T5,T6,T7 → bank 1
    // ...
    if (gid < n)
        g_out[gid] = (int)smem_char[tid];
}

// char 访问的正确方式：用 int 读取后提取字节
//
// 核心思路：
//   char 数组 4 个元素共享一个 bank，直接访问有 4-way conflict。
//   改用 int 数组：每 4 个 char 打包成 1 个 int，写入 smem_int[]。
//   读取时每个线程从 smem_int[] 取出对应字节，4 个线程访问同一 int → broadcast（无冲突）。
//
// 内存/bank 对应关系：
//   smem_int[0] → bank 0（4 bytes，存 char[0]~char[3]）
//   smem_int[1] → bank 1（4 bytes，存 char[4]~char[7]）
//   smem_int[k] → bank k
//   T0~T3 全读 smem_int[0]（同一地址）→ broadcast，不是 conflict ✓
__global__ void kernel_smem_char_fixed(const char* __restrict__ g_in,
                                        int* __restrict__ g_out, int n)
{
    // ── 改用 int 数组：每 int 一个 bank，避免 char 的 4-way conflict ──
    // smem_int[256]：256 个 int，对应 1024 个 char（= blockDim.x = 256 个线程的数据）
    // 每 4 个线程的 char 数据合并到同一个 int 槽位
    __shared__ int smem_int[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // ──────────────────────────────────────────────────────────────────
    // 【写入阶段】：每组 4 个线程只让 tid%4==0 的代表线程负责打包写入
    // ──────────────────────────────────────────────────────────────────
    //
    // 为什么只让 tid%4==0 的线程写，而不是 4 个线程各自写？
    //
    //   反例：如果 T0、T1、T2、T3 同时写 smem_int[0]：
    //     T0: smem_int[0] = (char[0])           // 只放 byte0，byte1~3 是 0
    //     T1: smem_int[0] = (char[1] << 8)      // 只放 byte1，byte0/2/3 是 0
    //     T2: smem_int[0] = (char[2] << 16)     // 只放 byte2
    //     T3: smem_int[0] = (char[3] << 24)     // 只放 byte3
    //     → 4 个线程写同一地址，即使值不同也是数据竞争（Race Condition），结果未定义
    //
    //   正确做法：指定一个"代表"线程，由它读取 4 个 char，打包后一次写入：
    //     T0 读 char[0]、char[1]、char[2]、char[3] → 打包为 1 个 int → 写 smem_int[0]
    //     T4 读 char[4]、char[5]、char[6]、char[7] → 打包为 1 个 int → 写 smem_int[1]
    //     ...
    //
    // 条件：gid < n（边界检查）且 tid % 4 == 0（只有代表线程执行）
    if (gid < n && tid % 4 == 0) {

        // idx：本线程负责写的 smem_int 槽位编号
        // T0  → idx=0，写 smem_int[0]（存 char[0~3]）
        // T4  → idx=1，写 smem_int[1]（存 char[4~7]）
        // T8  → idx=2，写 smem_int[2]（存 char[8~11]）
        // T252→ idx=63，写 smem_int[63]（存 char[252~255]）
        int idx = tid / 4;

        // packed：将 4 个 char 按小端序拼入同一个 int
        // 内存布局（小端）：
        //   bit 31~24: char[3]（高位）
        //   bit 23~16: char[2]
        //   bit 15~8 : char[1]
        //   bit  7~0 : char[0]（低位，i=0，shift=0）
        int packed = 0;

        // 循环 4 次，把连续的 4 个 char 逐字节拼入 packed
        // (gid/4)*4 = 本组的起始全局 index（对齐到 4 的倍数）
        //   T0:  gid=0,   (gid/4)*4 = 0   → 读 g_in[0], g_in[1], g_in[2], g_in[3]
        //   T4:  gid=4,   (gid/4)*4 = 4   → 读 g_in[4], g_in[5], g_in[6], g_in[7]
        //   T252:gid=252, (gid/4)*4 = 252 → 读 g_in[252..255]
        for (int i = 0; i < 4 && (gid/4)*4+i < n; i++) {
            // (unsigned char) 强制转换：防止 char 符号扩展
            //   char c = -1 (0xFF)，直接 (int)c = 0xFFFFFFFF（符号扩展，高 24 位全 1）
            //   (unsigned char)c = 0xFF，(int)(unsigned char)c = 0x000000FF（正确）
            //   不转换的话 << (i*8) 后会污染高位
            // i*8：字节在 int 中的位偏移
            //   i=0 → shift 0   → char 放在 bit[7:0]
            //   i=1 → shift 8   → char 放在 bit[15:8]
            //   i=2 → shift 16  → char 放在 bit[23:16]
            //   i=3 → shift 24  → char 放在 bit[31:24]
            // |= 按位或：逐步把 4 个字节组合进 packed，不会相互覆盖
            packed |= ((unsigned char)g_in[(gid/4)*4 + i]) << (i*8);
        }

        // 一次写入：代表线程把 4 个 char 打包后写入 smem_int
        // 写入的 smem_int[idx] 位于 bank idx，无冲突（每个代表线程写不同 bank）
        smem_int[idx] = packed;
    }

    // 等待所有代表线程完成写入，再进入读取阶段
    // 否则后续线程读到的 smem_int 可能还未被写入
    __syncthreads();

    // ──────────────────────────────────────────────────────────────────
    // 【读取阶段】：4 个线程读同一 int 槽，各自提取对应字节
    // ──────────────────────────────────────────────────────────────────
    //
    // bank conflict 分析：
    //   T0, T1, T2, T3 全读 smem_int[0] → 同一地址 → Broadcast（无冲突）✓
    //   T4, T5, T6, T7 全读 smem_int[1] → 同一地址 → Broadcast ✓
    //   ...
    //   broadcast 条件：同一 warp 内多个线程读取"完全相同"的地址 → 硬件广播，只需 1 次读
    //   vs. conflict：多个线程读取"同一 bank 的不同地址" → 串行化，需要多次读
    if (gid < n) {
        // idx：和写入时相同，T0~T3 的 idx=0，读 smem_int[0]
        // T0: idx=0, T1: idx=0, T2: idx=0, T3: idx=0 → 全读 smem_int[0] → broadcast
        // T4: idx=1, T5: idx=1, T6: idx=1, T7: idx=1 → 全读 smem_int[1] → broadcast
        int idx   = tid / 4;

        // shift：每个线程提取自己对应的字节
        // tid%4 决定取哪个字节，×8 转换为位偏移：
        //   T0: tid%4=0 → shift=0  → 取 bit[7:0]   （char[0]）
        //   T1: tid%4=1 → shift=8  → 取 bit[15:8]  （char[1]）
        //   T2: tid%4=2 → shift=16 → 取 bit[23:16] （char[2]）
        //   T3: tid%4=3 → shift=24 → 取 bit[31:24] （char[3]）
        int shift = (tid % 4) * 8;

        // >> shift：把目标字节移到最低位
        // & 0xFF：屏蔽高 24 位，只保留最低 8 位（防止有符号右移的符号扩展）
        //
        // 举例（smem_int[0] = 0xDDCCBBAA，AA=char[0], BB=char[1], ...）：
        //   T0: (0xDDCCBBAA >> 0 ) & 0xFF = 0xAA ✓  （char[0]）
        //   T1: (0xDDCCBBAA >> 8 ) & 0xFF = 0xBB ✓  （char[1]）
        //   T2: (0xDDCCBBAA >> 16) & 0xFF = 0xCC ✓  （char[2]）
        //   T3: (0xDDCCBBAA >> 24) & 0xFF = 0xDD ✓  （char[3]）
        g_out[gid] = (smem_int[idx] >> shift) & 0xFF;
    }
}


// ● 是的，padding 也是一种解决 char bank conflict 的方法。                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                            
//   核心思路：在每个 char 后面填充 3 个字节，使每个 char 独占一个 bank（4 bytes）：                                                                                                                                                                                           
                                                                                                                                                                                                                                                                            
//   // 原来：4 个 char 挤在同一个 bank → 4-way conflict                                                                                                                                                                                                                       
//   __shared__ char smem_char[256];
//   // smem_char[0] → bank 0, byte 0
//   // smem_char[1] → bank 0, byte 1  ← 撞了！
//   // smem_char[2] → bank 0, byte 2  ← 撞了！
//   // smem_char[3] → bank 0, byte 3  ← 撞了！

//   // padding：每个 char 后填 3 字节，使每个 char 起始地址是 4 的倍数
//   __shared__ char smem_pad[256 * 4];   // 每个槽位 4 bytes
//   // smem_pad[tid * 4] → bank tid → 每个线程独占一个 bank，无冲突 ✓

//   两种方案对比：

//   ┌──────────────┬────────────────────────────┬────────────────────────────────────┐
//   │              │        int 打包方案        │            padding 方案            │
//   ├──────────────┼────────────────────────────┼────────────────────────────────────┤
//   │ smem 用量    │ 256 × 4B = 1024B（无浪费） │ 256 × 4B = 1024B（3/4 是 padding） │
//   ├──────────────┼────────────────────────────┼────────────────────────────────────┤
//   │ 实际有效数据 │ 1024 bytes                 │ 256 bytes                          │
//   ├──────────────┼────────────────────────────┼────────────────────────────────────┤
//   │ smem 利用率  │ 100%                       │ 25%                                │
//   ├──────────────┼────────────────────────────┼────────────────────────────────────┤
//   │ 读写复杂度   │ 需要位操作打包/解包        │ 直接访问，代码简单                 │
//   ├──────────────┼────────────────────────────┼────────────────────────────────────┤
//   │ 适用场景     │ smem 紧张时                │ smem 充裕，代码简洁优先            │
//   └──────────────┴────────────────────────────┴────────────────────────────────────┘

//   padding 方案代码：

//   __global__ void kernel_smem_char_padding(const char* g_in, int* g_out, int n)
//   {
//       // 每个 char 占 4 bytes，stride=4，使每个线程访问不同 bank
//       __shared__ char smem[256 * 4];

//       int tid = threadIdx.x;
//       int gid = blockIdx.x * blockDim.x + tid;

//       // 写入：stride=4，T_i → bank i，无冲突
//       if (gid < n) smem[tid * 4] = g_in[gid];
//       __syncthreads();

//       // 读取：stride=4，T_i → bank i，无冲突
//       if (gid < n) g_out[gid] = (unsigned char)smem[tid * 4];
//   }

//   实际工程中更常见的写法是用 int 或 short 数组替代 char 数组，而不是手动 padding，因为 padding 浪费 smem 会降低 occupancy。


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2d: float 对齐访问 shared memory（无 bank conflict，最理想情况）
//
//   float (4B) 天然等于 1 个 bank（每个 bank = 4 bytes）：
//   → 每个 float 恰好落在 1 个 bank 内，stride-1 时 32 个线程访问 32 个不同 bank
//
//   bank 分布：
//   smem[0]: bank 0  (byte 0~3)
//   smem[1]: bank 1  (byte 4~7)
//   smem[2]: bank 2  (byte 8~11)
//   ...
//   smem[31]: bank 31 (byte 124~127)
//   smem[32]: bank 0  (byte 128~131，循环）
//
//   T0→bank0, T1→bank1, ..., T31→bank31 → 32 线程 × 1 bank = 32 次访问，
//   恰好覆盖 32 个不同 bank → 无冲突 ✓（float 是 shared memory 的"完美尺寸"）
//
//   对比 double（8B = 2 bank）：
//   double 每个元素跨 2 个 bank，32 线程 × 2 bank = 64 次访问 / 32 bank → 2-way conflict
//   float 每个元素只占 1 个 bank，32 线程 × 1 bank = 32 次访问 / 32 bank → 无冲突
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_smem_aligned_float(const float* __restrict__ g_in,
                                           float* __restrict__ g_out, int n)
{
    __shared__ float smem[256];   // 256 × 4B = 1024B，每个 float 恰好占 1 个 bank

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = g_in[gid];
    __syncthreads();

    // T_i → smem[i] → bank (i % 32)
    // 一个 warp 内 T0~T31 → bank0~bank31，各不相同 → 无冲突 ✓
    if (gid < n)
        g_out[gid] = smem[tid] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2e: float stride-2 访问 shared memory（2-way bank conflict）
//
//   ⚠️  注意：sm_75+ 对 shared memory 严格执行对齐：float 要求 4B 对齐，
//   double 要求 8B 对齐。非对齐访问会触发 misaligned address 硬件错误，
//   而非像旧架构（sm_52 等）那样静默拆分成多次事务。
//   因此无法用指针偏移来演示 float 跨 bank，改用 stride-2 访问实现同等的 2-way conflict。
//
//   stride-2 访问（每个线程步长为 2）：
//   T_i → smem[i*2] → bank (i*2 % 32)
//
//   warp 内冲突分析（bank_id = smem_index % 32）：
//   T0  → smem[0]  → bank 0
//   T1  → smem[2]  → bank 2
//   T2  → smem[4]  → bank 4
//   ...
//   T15 → smem[30] → bank 30
//   T16 → smem[32] → bank 0  ← 和 T0 同 bank → 2-way conflict!
//   T17 → smem[34] → bank 2  ← 和 T1 同 bank → 2-way conflict!
//   ...
//   T31 → smem[62] → bank 30 ← 和 T15 同 bank → 2-way conflict!
//
//   每个 bank 被 T_i 和 T_{i+16} 同时访问 → 全 warp 产生 2-way conflict
//   硬件串行化：每次读取需要 2 个时钟周期而非 1 个 → 理论上慢 2x
//
//   与 aligned float（stride-1）的区别：
//   stride-1: T0~T31 → bank 0~31，各不相同 → 无冲突
//   stride-2: T0/T16 → bank 0, T1/T17 → bank 2, ... → 2-way conflict
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_smem_unaligned_float(const float* __restrict__ g_in,
                                             float* __restrict__ g_out, int n)
{
    // stride-2：256 个线程，每个线程占 2 个 float 槽位，共需 512 个槽位
    __shared__ float smem[256 * 2];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 写入时步长为 2：T_i 写 smem[i*2]，每隔一个 bank 写一次
    if (gid < n) smem[tid * 2] = g_in[gid];
    __syncthreads();

    // T_i 和 T_{i+16} 访问同一 bank → 2-way conflict（间隔 16 的线程对互相冲突）
    if (gid < n)
        g_out[gid] = smem[tid * 2] * 2.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// 计时辅助
// ─────────────────────────────────────────────────────────────────────────────
// std::function<void()> 是 C++ 标准库的可调用对象包装器，可以包装任何无参数、无返回值的可调用对象。                                                                                                                                                                         
                                                                                                                                                                                                                                                                            
//   ---                                                                                                                                                                                                                                                                       
//   为什么这里用它而不是函数指针：
                                                                                                                                                                                                                                                                            
//   time_ms 需要计时不同的 kernel 调用，每次调用的参数都不一样：

//   // 每次调用参数不同，无法用同一个函数指针类型
//   kernel_aligned_double<<<GRID,BLOCK>>>(d_double_in, d_double_out, N);
//   kernel_unaligned_double<<<GRID,BLOCK>>>(d_char_raw, d_double_out, N);
//   kernel_packed_struct<<<GRID,BLOCK>>>(d_packed, d_float_out, N);

//   如果用函数指针，每种参数组合都需要一个不同签名的 time_ms，非常繁琐。

//   ---
//   用 lambda + std::function 的方案：

//   // 调用时传入 lambda，把参数捕获进去
//   float t1a = time_ms([&]{
//       kernel_aligned_double<<<GRID,BLOCK>>>(d_double_in, d_double_out, N);
//   });

//   float t1b = time_ms([&]{
//       kernel_unaligned_double<<<GRID,BLOCK>>>(d_char_raw, d_double_out, N);
//   });

//   lambda [&] 捕获外部所有变量，把参数差异封装进 lambda 内部，time_ms 只看到统一的 void() 接口。

//   ---
//   std::function<void()> 的语法含义：

//   std::function < void () >
//                 ↑返回类型 ↑参数列表（空）

//   ┌──────────────────────────────┬────────────────────────┐
//   │             类型             │          含义          │
//   ├──────────────────────────────┼────────────────────────┤
//   │ std::function<void()>        │ 无参数，无返回值       │
//   ├──────────────────────────────┼────────────────────────┤
//   │ std::function<int(float)>    │ 接受 float，返回 int   │
//   ├──────────────────────────────┼────────────────────────┤
//   │ std::function<void(int,int)> │ 接受两个 int，无返回值 │
//   └──────────────────────────────┴────────────────────────┘


static float time_ms(std::function<void()> fn)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < NWARM; i++) {
        CUDA_CHECK(cudaGetLastError());
        fn();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < NREP;  i++) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / NREP;
}




int main()
{
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    // ── Part 1 数据准备 ───────
    double *d_double_in, *d_double_out;
    char   *d_char_raw;
    float  *d_float_out;
                                                                                                                                                                                                                                                                 
    // C/C++ 整数类型家族（从小到大）：                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                
    // ┌────────────────────────────────────┬──────┬──────┐                                                                                                                                                                                                                      
    // │                类型                │ 字节 │ 位数 │
    // ├────────────────────────────────────┼──────┼──────┤
    // │ char / signed char / unsigned char │ 1    │ 8    │
    // ├────────────────────────────────────┼──────┼──────┤
    // │ short / unsigned short             │ 2    │ 16   │
    // ├────────────────────────────────────┼──────┼──────┤
    // │ int / unsigned int                 │ 4    │ 32   │
    // ├────────────────────────────────────┼──────┼──────┤
    // │ long long / unsigned long long     │ 8    │ 64   │
    // └────────────────────────────────────┴──────┴──────┘

    // ---
    // char 和 int 的关系：

    // char c = 65;
    // int  i = c;   // 隐式提升（integral promotion）：char → int
    //                 // c 和 i 都表示整数 65，只是存储大小不同

    // C/C++ 中所有比 int 小的整数类型（char、short）在参与运算时都会自动提升为 int，这叫做整数提升（integral promotion）。


    CUDA_CHECK(cudaMalloc(&d_double_in,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_double_out, N * sizeof(double)));
    // 所有线程统一偏移 +1 byte，最后一个线程（tid=N-1）访问结束 byte = 1+(N-1)*8+7 = N*8
    // 基础分配 N*8 字节索引 0~N*8-1，实际只越界 1B，+8 是保守的安全余量
    CUDA_CHECK(cudaMalloc(&d_char_raw,   N * sizeof(double) + 8));
    CUDA_CHECK(cudaMalloc(&d_float_out,  N * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_double_in,  0, N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_char_raw,   0, N * sizeof(double) + 8));

    // PackedStruct / AlignedStruct 
    // 指针变量是局部变量                                                                                                                                                            
    // PackedStruct *d_packed;      // 未初始化，值不确定（野指针）                                                                                                                                                                                                              
    // PackedStruct *d_packed {};   // 值初始化，等价于 = nullptr                                                                                                                                                                                                                
    // PackedStruct *d_packed = nullptr;  // 明确初始化为空指针
    // PackedStruct  *d_packed {};等价于PackedStruct *d_packed = nullptr;  // 明确初始化为空指针

    // 指针变量是全局变量 
    // 全局作用域
    // PackedStruct *d_packed;        // ✓ 自动初始化为 nullptr
    // PackedStruct *d_packed {};     // ✓ 显式值初始化为 nullptr（和上面等价）
    // PackedStruct *d_packed = nullptr; // ✓ 显式初始化为 nullptr（和上面等价）
    PackedStruct  *d_packed;  
    AlignedStruct *d_aligned;
    CUDA_CHECK(cudaMalloc(&d_packed,  N * sizeof(PackedStruct)));
    CUDA_CHECK(cudaMalloc(&d_aligned, N * sizeof(AlignedStruct)));


    // cudaMemset 是按字节填充，不理解结构体语义：

    //   cudaMemset(ptr, value, count);                                                                                                                                                                                                                                            
    //   // value：填充的字节值（0~255）
    //   // count：填充的字节数                                                                                                                                                                                                                                                    
    //   // 效果：把 ptr 开始的 count 个字节全部设为 value

    //   ---
    //   cudaMemset(d_packed, 0, N * sizeof(PackedStruct)) 的实际效果：

    //   把 d_packed 指向的内存的每一个字节都设为 0x00
    //   包括 flag、padding、value 的每一个字节全变成 0

    //   对于 float value：4 个字节全为 0x00 → IEEE 754 中 0x00000000 = 0.0f ✓

    //   所以 cudaMemset(..., 0, ...) 对于数值类型（int、float、double）全部置 0 是正确的，因为这些类型的 0 在内存中恰好就是全 0 字节。

    //   ---
    //   为什么不能用 cudaMemset(d_packed, {0,0}, ...)：

    //   cudaMemset 的第二个参数是 int，只接受单个字节值（0~255），不接受结构体：

    //   cudaMemset(ptr, 0,     count);  // ✓ 每字节填 0x00
    //   cudaMemset(ptr, 0xFF,  count);  // ✓ 每字节填 0xFF
    //   cudaMemset(ptr, {0,0}, count);  // ✗ 编译错误，类型不匹配    

    //   ---
    //   如果要用非零值初始化结构体，才需要在 host 端手动初始化再 memcpy：

    //   // host 端构造初始值
    //   std::vector<AlignedStruct> h_data(N);
    //   for (auto& s : h_data) { s.flag = 1; s.value = 3.14f; }

    //   // 拷贝到 device
    //   cudaMemcpy(d_aligned, h_data.data(), N * sizeof(AlignedStruct),
    //              cudaMemcpyHostToDevice);
    // 
    //   ---                                                                                                                                                                                                                                                                       
    //   std::vector<AlignedStruct> h_data(N);                                                                                                                                                                                                                                     
    //   h_data.data();  // 返回 AlignedStruct*，指向第一个元素                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                            
    //   ---
    //   为什么需要 data()：

    //   cudaMemcpy 的参数类型是 const void*，需要原始指针，不接受 std::vector 对象本身：

    //   cudaMemcpy(d_aligned, h_data,        ...);  // ✗ 编译错误，vector 不能隐式转指针
    //   cudaMemcpy(d_aligned, h_data.data(), ...);  // ✓ AlignedStruct* → const void*
    //   cudaMemcpy(d_aligned, &h_data[0],    ...);  // ✓ 等价写法，取第一个元素的地址

    //   ---
    //   std::vector 内存布局：

    //   h_data 对象（栈上）：
    //   ┌─────────┬──────┬──────────┐
    //   │ pointer │ size │ capacity │
    //   └────┬────┴──────┴──────────┘
    //        │
    //        ↓ data() 返回这个指针
    //   ┌──────────┬──────────┬─────┬──────────┐
    //   │ h_data[0]│ h_data[1]│ ... │h_data[N-1]│  ← 堆上连续内存
    //   └──────────┴──────────┴─────┴──────────┘


    // ● h_data[0] 是 std::vector 的下标运算符，返回第 0 个元素，类型是 AlignedStruct。                                                                                                                                                                                            
                                                                                                                                                                                                                                                                            
    //   ---                                                                                                                                                                                                                                                                       
    //   std::vector<AlignedStruct> h_data(N);                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                
    //   h_data[0]        // 第 0 个 AlignedStruct 对象                                                                                                                                                                                                                          
    //   h_data[0].flag   // 第 0 个元素的 flag 成员
    //   h_data[0].value  // 第 0 个元素的 value 成员

    //   h_data[1]        // 第 1 个 AlignedStruct 对象
    //   h_data[N-1]      // 最后一个 AlignedStruct 对象

    //   ---
    //   &h_data[0] 和 h_data.data() 的等价性：

    //   &h_data[0]    // 取第 0 个元素的地址 → AlignedStruct*
    //   h_data.data() // 返回底层数组首地址 → AlignedStruct*

    //   // 两者指向同一地址，完全等价
    //   assert(&h_data[0] == h_data.data());  // 永远成立



    //   ---                                                                                                                                                                                                                                                                       
    //   元素访问等价：
                                                                                                                                                                                                                                                                                
    //   AlignedStruct arr[N];                                                                                                                                                                                                                                                   
    //   arr[0]          // 第 0 个元素（AlignedStruct 对象）

    //   std::vector<AlignedStruct> h_data(N);
    //   h_data[0]       // 第 0 个元素（AlignedStruct 对象）

    //   // arr[0] 和 h_data[0] 类型相同，都是 AlignedStruct ✓

    //   ---
    //   地址等价：

    //   // 普通数组
    //   &arr[0]  ==  arr        // ✓ 数组名本身就是首元素地址

    //   // vector
    //   &h_data[0]  ==  h_data.data()  // ✓ 两者都是首元素地址
    //   // 但 h_data 本身（vector 对象）≠ h_data.data()（首元素地址）



    //   ---                                                                                                                                                                                                                                                                       
    //   退化（decay）发生在数组名上：                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                
    //   AlignedStruct arr[N];                                                                                                                                                                                                                                                   

    //   arr       // 退化为 AlignedStruct*，指向第 0 个元素
    //   arr[0]    // 第 0 个元素本身，类型是 AlignedStruct（对象，不是指针）
    //   &arr[0]   // 第 0 个元素的地址，类型是 AlignedStruct*

    //   ---
    //   退化的本质：

    //   arr[0] 等价于 *(arr + 0)，是对指针解引用，得到的是对象本身，不是指针：

    //   arr        // AlignedStruct*（数组名退化为指针）
    //   arr + 0    // AlignedStruct*（指针偏移）
    //   *(arr + 0) // AlignedStruct （解引用，得到对象）
    //   arr[0]     // AlignedStruct （同上，语法糖）
    CUDA_CHECK(cudaMemset(d_packed,  0, N * sizeof(PackedStruct)));
    CUDA_CHECK(cudaMemset(d_aligned, 0, N * sizeof(AlignedStruct)));

    // ── Part 2 数据准备 ─────
    char  *d_char_in;
    int   *d_int_out;
    float *d_float_in;   // float smem 测试的输入
    CUDA_CHECK(cudaMalloc(&d_char_in,  N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_int_out,  N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_float_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_char_in,  0, N * sizeof(char)));
    CUDA_CHECK(cudaMemset(d_float_in, 0, N * sizeof(float)));



    
    // [&] 捕获的是 lambda 定义处外部作用域中所有变量的引用。                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                
    // ---                                                                                                                                                                                                                                                                       
    // lambda 捕获列表含义：
                                                                                                                                                                                                                                                                                
    // ┌──────┬────────────────────────────────┐
    // │ 写法 │              含义              │
    // ├──────┼────────────────────────────────┤
    // │ [&]  │ 捕获所有外部变量，按引用       │
    // ├──────┼────────────────────────────────┤
    // │ [=]  │ 捕获所有外部变量，按值（拷贝） │
    // ├──────┼────────────────────────────────┤
    // │ [&x] │ 只捕获变量 x，按引用           │
    // ├──────┼────────────────────────────────┤
    // │ [x]  │ 只捕获变量 x，按值             │
    // ├──────┼────────────────────────────────┤
    // │ []   │ 不捕获任何外部变量             │
    // └──────┴────────────────────────────────┘

    // ---
    // 为什么这里必须用 [&]：

    // // main 中定义的变量
    // double *d_double_in, *d_double_out;
    // const int GRID = ...;
    // const int BLOCK = ...;

    // // lambda 内部用到了这些变量
    // float t1a = time_ms([&]{
    //     kernel_aligned_double<<<GRID, BLOCK>>>(d_double_in, d_double_out, N);
    //     //                      ↑     ↑        ↑           ↑
    //     //                  外部变量，必须捕获才能在 lambda 内访问
    // });

    // lambda 本质是一个匿名函数对象，不能直接访问外部变量，必须通过捕获列表声明要用哪些变量。

    // ---
    // [&] vs [=] 在这里的区别：

    // // [=]：拷贝指针值 → 拷贝的是指针本身（地址值）
    // // 指针指向的显存数据不会被拷贝，只是复制了一份指针变量
    // // 效果和 [&] 几乎一样，但语义不同

    // double* p = d_double_in;   // [=] 拷贝这个地址值
    // // lambda 内用的是拷贝出来的 p，值和原来相同
    // // 对于指针变量，[=] 和 [&] 效果等价

    // // [&]：引用外部指针变量本身
    // // 若 lambda 执行期间外部指针被修改（如 realloc），lambda 能看到新值

    // 这里用 [&] 是惯用写法，对于指针类型变量效果等同于 [=]，但 [&] 更简洁，不需要逐个列出要捕获的变量。    




    // [&]{ ... } 是完整 lambda 语法的省略形式。                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                
    // ---                                                                                                                                                                                                                                                                       
    // 完整 lambda 语法：                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    // [捕获列表] (参数列表) -> 返回类型 { 函数体 }
    
    // 省略规则：
    // // 完整写法
    // [&]() -> void { kernel<<<G,B>>>(args); }
    // // 省略参数列表（无参数时可省略 ()）
    // [&] -> void { kernel<<<G,B>>>(args); }

    // // 省略返回类型（编译器可推导时可省略 -> void）
    // [&]() { kernel<<<G,B>>>(args); }

    // // 两者都省略（最常见写法）
    // [&]{ kernel<<<G,B>>>(args); }

    // ---
    // 省略条件：

    // ┌─────────────┬──────────────────────────────────────────────────┐
    // │    部分     │                     省略条件                     │
    // ├─────────────┼──────────────────────────────────────────────────┤
    // │ () 参数列表 │ 无参数且无 mutable/noexcept/-> 时可省            │
    // ├─────────────┼──────────────────────────────────────────────────┤
    // │ -> 返回类型 │ 编译器能推导返回类型时可省（void 或单一 return） │
    // └─────────────┴──────────────────────────────────────────────────┘


    // ── Part 1 测试 ──────
    printf("\n===== Part 1: HBM 跨 Sector =====\n\n");

    float t1a = time_ms([&]{ kernel_aligned_double<<<GRID,BLOCK>>>(d_double_in, d_double_out, N); });
    // ● d_char_raw 声明的是 char*，kernel 参数也是 const char*，所以这里的 (char*) 转换确实是多余的，可以去掉。  
    float t1b = time_ms([&]{ kernel_unaligned_double<<<GRID,BLOCK>>>((char*)d_char_raw, d_double_out, N); });
    float t1c_bad  = time_ms([&]{ kernel_packed_struct<<<GRID,BLOCK>>>(d_packed, d_float_out, N); });
    float t1c_good = time_ms([&]{ kernel_aligned_struct<<<GRID,BLOCK>>>(d_aligned, d_float_out, N); });




    printf("%-35s  time=%6.3f ms\n", "aligned double (不跨sector)", t1a);
    printf("%-35s  time=%6.3f ms  slowdown=%.2fx\n",
           "unaligned double (跨sector)", t1b, t1b / t1a);
    printf("%-35s  time=%6.3f ms\n", "aligned struct (不跨sector)", t1c_good);
    printf("%-35s  time=%6.3f ms  slowdown=%.2fx\n",
           "packed struct (float跨sector)", t1c_bad, t1c_bad / t1c_good);

    printf("\n跨 sector 的原因：\n");
    printf("  double (8B) 在 addr=0x01 → [0x01~0x04] 跨 Sector 0 和 Sector 1\n");
    printf("  packed struct → char(1B) 使 float 偏移到非 4B 对齐地址\n");
    printf("  → 每次读取需要 2 次 transaction 而非 1 次\n");


    // ── Part 2 测试 ─────
    printf("\n===== Part 2: Shared Memory 跨 Bank =====\n\n");
    printf("\n===== 执行: kernel_smem_aligned_double =====\n\n");
    float t2a = time_ms([&]{ kernel_smem_aligned_double<<<GRID,BLOCK>>>(d_double_in, d_double_out, N); });
    printf("\n===== 执行: kernel_smem_unaligned_double =====\n\n");
    float t2b = time_ms([&]{ kernel_smem_unaligned_double<<<GRID,BLOCK>>>(d_double_in, d_double_out, N); });
    printf("\n===== 执行: kernel_smem_char_conflict =====\n\n");
    float t2c_bad  = time_ms([&]{ kernel_smem_char_conflict<<<GRID,BLOCK>>>(d_char_in, d_int_out, N); });
    printf("\n===== 执行: kernel_smem_char_fixed =====\n\n");
    float t2c_good = time_ms([&]{ kernel_smem_char_fixed<<<GRID,BLOCK>>>(d_char_in, d_int_out, N); });
    // float：4B = 1 bank，对齐时无冲突；未对齐时每个 float 跨 2 bank → 2-way conflict
    printf("\n===== 执行: kernel_smem_aligned_float =====\n\n");
    float t2d_good = time_ms([&]{ kernel_smem_aligned_float<<<GRID,BLOCK>>>(d_float_in, d_float_out, N); });
    printf("\n===== 执行: kernel_smem_unaligned_float =====\n\n");
    float t2d_bad  = time_ms([&]{ kernel_smem_unaligned_float<<<GRID,BLOCK>>>(d_float_in, d_float_out, N); });

    printf("%-38s  time=%6.3f ms\n", "float aligned smem (无冲突，基准)", t2d_good);
    printf("%-38s  time=%6.3f ms  slowdown=%.2fx\n",
           "float stride-2 smem (2-way冲突)", t2d_bad, t2d_bad / t2d_good);
    printf("%-38s  time=%6.3f ms  slowdown=%.2fx\n",
           "double aligned smem (2-way冲突)", t2a, t2a / t2d_good);
    printf("%-38s  time=%6.3f ms  slowdown=%.2fx\n",
           "double unaligned smem (2-way冲突)", t2b, t2b / t2d_good);
    printf("%-38s  time=%6.3f ms\n", "char fixed smem (broadcast，无冲突)", t2c_good);
    printf("%-38s  time=%6.3f ms  slowdown=%.2fx\n",
           "char conflict smem (4-way冲突)", t2c_bad, t2c_bad / t2c_good);

    printf("\n冲突总结：\n");
    printf("  float 4B = 1 bank：对齐时无冲突（理想基准）\n");
    printf("  float stride-2：T_i 和 T_{i+16} 共享 bank → 2-way（sm_75+ 不允许指针非对齐）\n");
    printf("  double 8B = 2 bank：即使对齐，T0 和 T16 撞 bank → 2-way（结构性冲突）\n");
    printf("  char 1B：T0~T3 全在 bank0 → 4-way conflict\n");
    printf("  bank_id = (byte_address / 4) %% 32\n");

    printf("\n===== 用 Nsight Compute 验证 =====\n");
    printf("ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\\\n");
    printf("             l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\\n");
    printf("             ./cross\n");

    // ── 清理 ──────
    cudaFree(d_double_in);  cudaFree(d_double_out);
    cudaFree(d_char_raw);   cudaFree(d_float_out);
    cudaFree(d_packed);     cudaFree(d_aligned);
    cudaFree(d_char_in);    cudaFree(d_int_out);
    cudaFree(d_float_in);
    return 0;
}
