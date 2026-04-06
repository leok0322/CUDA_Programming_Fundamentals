# ════════════════════════════════
# CMakeLists.txt — CUDA Programming Guide 示例项目
#
# 本文件位于项目根目录：
#   Programming GPUs in CUDA/        ← 本文件所在位置（项目根目录）
#   ├── Asynchronous Execution/
#   ├── Into to CUDA C++/
#   ├── Unified and System Memory/
#   └── Writing  CUDA SIMT kernel/
#
# 上一级目录 "CUDA Programming Guide/" 仅为包装文件夹，不含源文件，
# 因此 CMakeLists.txt 放在 "Programming GPUs in CUDA/" 下即为正确的项目根。
# ═════════════════════════════════

# cmake_minimum_required：CMake 的最低版本要求
# VERSION 3.20 起原生支持 CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.20)

# ─────────────────────────────────
# 编译器路径（Toolchain）
# 必须在 project() 之前设置，project() 会触发编译器探测，之后设置无效。
#
# 不显式指定时的探测优先级（从高到低）：
#   1. 本文件 set()                          ← 最高，当前做法，锁定版本
#   2. cmake 命令行 -DCMAKE_CUDA_COMPILER=...
#   3. CLion Toolchains                      CLion 将其转为第 2 级命令行参数
#   4. 系统 PATH 自动探测（which nvcc）       最低优先级
#
# 本机单版本 CUDA 环境下 PATH 自动探测与显式指定结果相同；
# 多版本 CUDA 共存时显式指定可避免探测到错误的 nvcc。
#
# 本机路径（已确认）：
#   nvcc  → /usr/local/cuda/bin/nvcc   （which nvcc）
#   g++   → /usr/bin/g++               （which g++，版本 11.4.0）
#   cmake → /usr/local/bin/cmake       （which cmake）
#
# CLion Toolchains 与此的关系：
#   本项目已在文件中显式 set()，CLion Toolchains 的编译器配置对本项目无效。
#   可通过以下命令确认 CMake 实际使用的编译器路径：
#     cmake .. && cmake -LA | grep COMPILER
# ───────────────────────────────────
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")   # 本机 nvcc（which nvcc）
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")               # 本机 g++（which g++，版本 11.4.0）

# ────────────────────────────────
# 项目声明
#   NAME        — 项目名称，同时设置 CMAKE_PROJECT_NAME 变量
#   VERSION     — 版本号，自动拆分为 _MAJOR / _MINOR / _PATCH 三个变量
#   DESCRIPTION — 项目描述（CMake 3.9+），存入 CMAKE_PROJECT_DESCRIPTION 变量
#   LANGUAGES   — 启用的编程语言
#
# 编译器区分（第一处）：LANGUAGES 注册编译器
#   CXX  → 注册 C++ 编译器（g++），存入 CMAKE_CXX_COMPILER
#   CUDA → 注册 CUDA 编译器（nvcc），存入 CMAKE_CUDA_COMPILER
#   两个编译器在此同时激活，后续 CMake 根据文件扩展名决定调用哪一个。
# ─────────────────────────────
project(CUDAProgrammingGuide
    VERSION     1.0
    DESCRIPTION "CUDA Programming Guide 配套示例：涵盖统一内存、异步执行、SIMT kernel 等核心主题"
    LANGUAGES   CXX CUDA
)

# message(STATUS ...)：在 cmake 配置阶段打印信息（前缀 "-- "）
# 其他级别：
#   message(STATUS  "...")   — 普通信息，前缀 "--"
#   message(WARNING "...")   — 警告，继续执行
#   message(FATAL_ERROR "…") — 致命错误，立即中止
message(STATUS "项目：${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "描述：${PROJECT_DESCRIPTION}")

# ──────────────────────
# C++ / CUDA 标准
# ────────────────────────

# CMAKE_CXX_STANDARD：设置全局 C++ 标准；20 对应 -std=c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # 若编译器不支持则报错，而非静默降级
set(CMAKE_CXX_EXTENSIONS OFF)         # 禁用 GNU 扩展（-std=c++20 而非 -std=gnu++20）

# CMAKE_CUDA_STANDARD：设置全局 CUDA host 端 C++ 标准
#
# .cu 文件由 nvcc 编译，nvcc 内部把 .cu 拆成两部分分别处理：
#   · device 端代码（__global__ / __device__ 函数）→ nvcc 自身编译为 PTX / SASS
#   · host 端代码（普通 C++ 部分）                 → nvcc 转交给底层 g++ 编译
#
# CMAKE_CUDA_STANDARD 控制的是 host 端部分交给 g++ 时使用的 C++ 标准，
# nvcc 将其翻译为 -std=c++20 传给 g++。
# device 端代码的语言标准由 nvcc 自身决定，受 CUDA 版本约束，
# 但大多数 C++17/20 特性（constexpr、if constexpr、structured bindings 等）
# 在 device 端也是支持的（CUDA 11+）。
#
# CMAKE_CXX_STANDARD 与 CMAKE_CUDA_STANDARD 的关系：
#   · CMAKE_CXX_STANDARD  → 控制 .cpp 文件由 g++ 编译时的 -std 参数
#   · CMAKE_CUDA_STANDARD → 控制 .cu 文件的 host 端部分由 g++ 编译时的 -std 参数
#   · 两者独立设置，通常保持一致（都设为 20），避免同一项目中 .cpp 和 .cu 的
#     host 端语言特性不一致导致链接时 ABI 冲突。
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)  # 若 nvcc/g++ 不支持 C++20 则报错，而非静默降级

message(STATUS "C++ 标准：C++${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA 标准：C++${CMAKE_CUDA_STANDARD}")

# ────────────────────────────
# 调试标志（Debug 模式，CLion + cuda-gdb）
#
# CMAKE_BUILD_TYPE 不是本文件 set() 定义的，而是从外部传入：
#   来源优先级：
#     1. cmake 命令行：cmake -DCMAKE_BUILD_TYPE=Debug ..   ← 最常用
#     2. 本文件 set(CMAKE_BUILD_TYPE Debug)                ← 可设默认值，但会覆盖命令行（不推荐）
#     3. 不传则为空字符串 ""，_DEBUG / _RELEASE 后缀的标志均不生效
#   CLion CMake Profile 中的 "Build type" 由用户手动设置：
#     Settings → Build, Execution, Deployment → CMake → + 新增 Profile
#       Profile "Debug"   → Build type: Debug   → CLion 传 -DCMAKE_BUILD_TYPE=Debug
#       Profile "Release" → Build type: Release → CLion 传 -DCMAKE_BUILD_TYPE=Release
#     右上角下拉框切换 Profile，CLion 自动用对应参数重新 configure，生成独立的 build 目录
#
# _DEBUG 后缀的变量仅在 Debug 模式下生效，不影响 Release 编译。
#
# CMAKE_CUDA_FLAGS_DEBUG：
#   -G   生成 device 端调试信息（DWARF），cuda-gdb 断点进入 kernel 必须加此选项
#        副作用：禁用几乎所有 GPU 优化，运行速度大幅下降，仅用于调试
#   -g   生成 host 端调试信息，使 cuda-gdb 能显示 host 代码的行号和变量
#
# CMAKE_CXX_FLAGS_DEBUG：
#   -g   生成 host C++ 调试信息
#   -O0  关闭优化，防止编译器重排/内联代码导致调试器行为异常
#
# Debug vs Release 效果对比：
#                    Debug                        Release
#   nvcc 标志        -G -g（本文件设置）           CMake 默认 -O3
#   g++ 标志         -g -O0（本文件设置）          CMake 默认 -O3 -DNDEBUG
#   kernel 断点      可用（-G 生成设备调试信息）    不可用（无 -G）
#   运行速度         慢（禁用 GPU/CPU 优化）        快（全优化）
#   二进制大小       大（含调试符号）               小
#   assert()         生效                          禁用（-DNDEBUG 屏蔽所有断言）
# ──────────────────────────────
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")   # 传给 nvcc：-G 生成 device 端调试信息（kernel 断点必须）
                                      #             -g 生成 host 端调试信息（行号/变量）
set(CMAKE_CXX_FLAGS_DEBUG  "-g -O0") # 传给 g++：  -g 生成 C++ 调试信息
                                      #             -O0 关闭优化，防止调试器跳行

# ──────────────────────────────
# nvcc --keep：保留编译中间文件
#
# nvcc 的编译流程分为多个阶段，每个阶段产生中间文件：
#   .cu
#    │ cudafe（CUDA Frontend，头文件展开 + __global__/__device__ 分离）
#    ▼
#   .cudafe1.cpp   — host 端 C++ 代码（交给 g++ 编译）
#   .cudafe1.gpu   — device 端代码（交给 ptxas 编译）
#    │ ptxas（PTX Assembler）
#    ▼
#   .ptx           — PTX 汇编（架构无关的虚拟 ISA）
#   .cubin         — SASS 机器码（与具体 SM 版本绑定）
#    │ fatbinary（打包所有架构的 PTX / CUBIN）
#    ▼
#   .fatbin        — 包含多架构代码的 fat binary
#    │ g++（链接 host 代码与 fat binary）
#    ▼
#   可执行文件
#
# 不加 --keep 时，上述中间文件在编译结束后自动删除。
# 加 --keep 后，所有中间文件保留在构建目录中，用于：
#   · 查看 PTX 代码（验证编译器生成的虚拟指令）
#   · 查看 SASS 代码（配合 cuobjdump --dump-sass 分析实际机器码）
#   · 调试 nvcc 编译流程问题
#   · 学习 CUDA 编译器的代码生成行为
#
# ── "${CMAKE_CUDA_FLAGS} <flag>" 追加惯用法 ──────────────────────
#
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} <flag>")
#   ↑ 写回同名变量      ↑ 读取旧值            ↑ 追加新标志
#
# ${CMAKE_CUDA_FLAGS} 先读取变量的当前值（CMake 已写入的默认标志），
# 拼接新标志后再写回。原有标志保留，新标志附加在末尾。
# 等价于 shell 中的：export CFLAGS="$CFLAGS -O2"
#
# 直接写 set(CMAKE_CUDA_FLAGS "<flag>") 会覆盖 CMake 内部默认标志
# （如 -arch=sm_86、--generate-code 等），导致编译失败或行为异常。
#
# --keep / --keep-dir 改为在 cuda_example 函数中按目标单独设置（见下方），
# 不在此处全局追加，原因：
#   · --keep-dir 需要包含目标名的路径，全局变量无法引用目标名
#   · 全局 --keep 会对所有目标生效，包括 CMake 自身的探测目标，产生干扰
# ──────────────────────────────
message(STATUS "构建类型：${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Debug 标志：${CMAKE_CUDA_FLAGS_DEBUG}")

# ──────────────────
# GPU 架构
#
# CMAKE_CUDA_ARCHITECTURES：指定目标 GPU 的 Compute Capability（SM 版本）
# CMAKE_CUDA_ARCHITECTURES 与 nvcc -gencode 的关系：
#
#   CMake 把 CMAKE_CUDA_ARCHITECTURES 的整数值自动翻译为两条 -gencode 标志：
#     CMAKE_CUDA_ARCHITECTURES = 86  →  nvcc 实际收到：
#       -gencode arch=compute_86,code=sm_86       ← 生成 SM 8.6 SASS 机器码
#       -gencode arch=compute_86,code=compute_86  ← 同时嵌入 PTX（前向兼容）
#
#   arch=compute_XY：编译时使用的虚拟架构（PTX ISA 版本），决定可用的 CUDA 特性
#   code=sm_XY    ：生成与 SM XY 绑定的 SASS 机器码，只能在 SM XY 及以上运行
#   code=compute_XY：嵌入 PTX，驱动在运行时 JIT 编译为当前 GPU 的 SASS
#                    作用：新 GPU 上即使没有对应 SASS 也能运行（兼容性保底）
#
#   细粒度写法（CMAKE_CUDA_ARCHITECTURES 支持后缀控制）：
#     86-real    → 只生成 SASS，不嵌入 PTX（二进制更小，只支持 SM 8.6）
#     86-virtual → 只嵌入 PTX，不生成 SASS（运行时 JIT，启动慢）
#     86         → 同时生成 SASS + PTX（默认，兼顾性能与兼容性）
#
#   手动指定写法（以分号分隔多个架构）：
#     set(CMAKE_CUDA_ARCHITECTURES 89)          # 单个 GPU，如 RTX 4090（Ada，SM 8.9）
#     set(CMAKE_CUDA_ARCHITECTURES 80)          # 单个 GPU，如 A100（Ampere，SM 8.0）
#     set(CMAKE_CUDA_ARCHITECTURES 80;86;89)    # 同时兼容 A100 / RTX 3090 / RTX 4090
#
#   常见 SM 版本对照：
#     75 — Turing  （RTX 2080）
#     80 — Ampere  （A100）
#     86 — Ampere  （RTX 3090 / RTX 3060）
#     89 — Ada     （RTX 4090）
#     90 — Hopper  （H100）
#
# ── 动态查询：configure 阶段调用 nvidia-smi 获取实际 SM 版本 ─────
#
# execute_process：在 cmake configure 阶段（读取本文件时）执行 shell 命令。
# 与 add_custom_command（编译阶段执行）不同，execute_process 是同步的，
# 命令执行完毕后 cmake 才继续解析后续指令。
#
# nvidia-smi --query-gpu=compute_cap --format=csv,noheader
#   --query-gpu=compute_cap  只查询计算能力字段
#   --format=csv,noheader    输出纯 CSV，不含表头行
#   单卡输出：8.6
#   多卡输出：8.6\n9.0\n（每张卡一行）
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _gpu_cap_raw          # 原始输出存入临时变量（下划线前缀表示内部变量）
    OUTPUT_STRIP_TRAILING_WHITESPACE      # 去掉末尾换行符
    ERROR_QUIET                           # nvidia-smi 不存在时不打印错误，静默失败
    RESULT_VARIABLE _nvidia_smi_exit_code # 0 = 成功，非 0 = 失败（nvidia-smi 不存在等）
)

if(_nvidia_smi_exit_code EQUAL 0 AND _gpu_cap_raw)
    # 多卡时取第一行（REGEX MATCH 只匹配第一个结果）
    # 匹配形如 "8.6" 的计算能力字符串
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _compute_cap "${_gpu_cap_raw}")

    # "8.6" → "86"：去掉小数点，得到 CMAKE_CUDA_ARCHITECTURES 需要的整数格式
    string(REPLACE "." "" _compute_cap_num "${_compute_cap}")

    set(CMAKE_CUDA_ARCHITECTURES ${_compute_cap_num})
    message(STATUS "nvidia-smi 探测到计算能力：${_compute_cap}  →  CMAKE_CUDA_ARCHITECTURES = ${_compute_cap_num}")
    message(STATUS "nvcc 等价 gencode 标志：-gencode arch=compute_${_compute_cap_num},code=sm_${_compute_cap_num}  -gencode arch=compute_${_compute_cap_num},code=compute_${_compute_cap_num}")
    # ── CMAKE_CUDA_ARCHITECTURES → --generate-code 的完整翻译链 ──
    #
    # set(CMAKE_CUDA_ARCHITECTURES 86)
    #         │
    #         │  configure 阶段：全局变量传播给每个目标
    #         ▼
    # set_target_properties(target PROPERTIES CUDA_ARCHITECTURES "86")
    #         │  cuda_example 函数中显式调用，每个目标从全局变量继承此值
    #         │
    #         │  generate 阶段：CMake C++ 源码（cmGeneratorTarget.cxx）
    #         │  读取每个目标的 CUDA_ARCHITECTURES 属性，按内置规则翻译：
    #         │
    #         │    整数 86        → code=[compute_86,sm_86]   SASS + PTX 均生成
    #         │    86-real        → code=sm_86                只生成 SASS（无 PTX，无前向兼容）
    #         │    86-virtual     → code=compute_86           只嵌入 PTX（JIT，前向兼容）
    #         │
    #         │  翻译逻辑在 CMake C++ 源码中硬编码，不在任何 .cmake 脚本里，
    #         │  ${CMAKE_ROOT}/Modules/Compiler/NVIDIA-CUDA.cmake 只做格式校验，
    #         │  ${CMAKE_ROOT}/Modules/CUDA/architectures.cmake 只维护合法架构列表。
    #         ▼
    # 写入 cmake-build-debug/CMakeFiles/<target>.dir/flags.make：
    #   CUDA_FLAGS = ... --generate-code=arch=compute_86,code=[compute_86,sm_86]
    #         │  code=[...] 是 nvcc 的括号合并写法，
    #         │  等价于两条分开的 --generate-code：
    #         │    --generate-code arch=compute_86,code=sm_86
    #         │    --generate-code arch=compute_86,code=compute_86
    #         ▼
    # nvcc 编译时同时产出：
    #   · SM 8.6 SASS 机器码（直接执行，无 JIT 开销）
    #   · PTX 虚拟指令（嵌入二进制，新 GPU 上驱动 JIT 编译为对应 SASS）
else()
    # 回退：nvidia-smi 不可用（无 GPU、CI 环境、容器内等）时使用 native
    # native：CMake 3.24+ 在 configure 阶段用另一套机制（编译探测程序）自动探测
    set(CMAKE_CUDA_ARCHITECTURES native)
    message(WARNING "nvidia-smi 查询失败（exit=${_nvidia_smi_exit_code}），回退到 native 自动探测")
endif()
message(STATUS "CUDA 架构：${CMAKE_CUDA_ARCHITECTURES}")

# ═══════════
# 辅助函数
# ═══════════

# ────────────────────
# cuda_example：为 .cu 示例创建可执行目标
#
# 用法：cuda_example(<目标名> <源文件路径>)
#
# 路径说明：
#   <目标名>      不是路径，是编译产物的名字
#                 可执行文件输出到构建目录：cmake-build-debug/<目标名>
#   <源文件路径>  相对于本文件所在目录（CMAKE_CURRENT_SOURCE_DIR）
#                 例："Unified and System Memory/pointer_attributes.cu"
#                 展开为：.../Programming GPUs in CUDA/Unified and System Memory/pointer_attributes.cu
#   cmake 命令中的 ".."（如 cmake ..）相对于 shell 当前目录，与本文件无关
#
# 编译器区分（第二处）：add_executable 按文件扩展名分派编译器
#   .cu  → CMake 调用 nvcc 编译
#   .cpp → CMake 调用 g++ 编译
#   两种文件可混在同一 add_executable 中，CMake 逐文件分派
#
# target_sources 说明：
#   target_sources(<目标> PRIVATE/PUBLIC/INTERFACE <源文件...>)
#   用于向已有目标追加源文件，将源文件声明与目标创建分离。
#   本项目每个目标只有一个 .cu 文件，直接在 add_executable 中传入即可；
#   若将来需要拆分为多个源文件，可用 target_sources 追加：
#     cuda_example(my_target "main.cu")
#     target_sources(my_target PRIVATE "helper.cu" "kernels.cu")
#   PRIVATE   — 只有本目标编译这些文件
#   PUBLIC    — 本目标及依赖本目标的目标都编译
#   INTERFACE — 只有依赖本目标的目标编译，本目标自身不编译
# ──────────────────────────────────
function(cuda_example target_name source_file)
    # add_executable：告知 CMake 从 source_file 生成名为 target_name 的可执行文件
    add_executable(${target_name} "${source_file}")

    # set_target_properties：对单个目标设置属性
    # CUDA_ARCHITECTURES 属性覆盖该目标的 GPU 架构（此处沿用全局 native）
    set_target_properties(${target_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    # ── nvcc 中间文件：--keep + --keep-dir ───────────────────────
    #
    # 中间文件输出到：<构建目录>/temp/<目标名>/
    #   例：cmake-build-debug/temp/pinned_host_kernel/
    #
    # 注意：路径不能是 <构建目录>/<目标名>/，
    #       因为链接器会在构建目录根生成同名可执行文件（如 cmake-build-debug/pinned_host_kernel），
    #       若同名目录已存在，ld 报 "Is a directory" 错误。
    #       将目标名放在 temp/ 子目录下可避免冲突。
    #
    # CMAKE_BINARY_DIR：构建目录根（CLion 默认为 cmake-build-debug/）
    # ${target_name}：当前目标名，函数参数，每个目标独立子目录，互不干扰
    #
    # file(MAKE_DIRECTORY ...)：在 cmake 配置阶段预先创建目录。
    # nvcc 不会自动创建 --keep-dir 指定的目录，目录不存在时编译报错。
    set(_keep_dir "${CMAKE_BINARY_DIR}/temp/${target_name}")
    file(MAKE_DIRECTORY "${_keep_dir}")

    # ── target_compile_options vs set_source_files_properties ───────
    #
    # 两者都能为文件追加编译选项，粒度不同：
    #
    #   target_compile_options(<目标> PRIVATE <选项>)
    #     → 作用于目标下的所有源文件
    #     → 本项目每个目标只有一个 .cu，与文件级写法效果相同
    #     → 多源文件目标时，所有文件都会加上该选项
    #
    #   set_source_files_properties(<文件> PROPERTIES COMPILE_OPTIONS <选项>)
    #     → 只作用于指定的那一个源文件，其他文件不受影响
    #     → 适合目标内部分文件需要特殊编译参数的场景：
    #         add_executable(my_target a.cu b.cu c.cu)
    #         set_source_files_properties(b.cu PROPERTIES
    #             COMPILE_OPTIONS "--use_fast_math"  # 只有 b.cu 开启快速数学
    #         )
    #     → 另一个常见用途：强制用 nvcc 编译 .cpp 文件（文件内含 __global__ 等）：
    #         set_source_files_properties(kernel.cpp PROPERTIES LANGUAGE CUDA)
    #         # CMake 默认按扩展名分派编译器（.cpp → g++），
    #         # LANGUAGE CUDA 覆盖此行为，让 nvcc 处理该 .cpp 文件
    #
    # PRIVATE 传播语义（target_compile_options 的第二个参数）：
    #   PRIVATE   → 选项只作用于本目标，不传播给依赖本目标的其他目标
    #   PUBLIC    → 本目标及所有依赖本目标的目标都加该选项
    #   INTERFACE → 只传播给依赖本目标的目标，本目标自身不加
    #   本项目各目标均为独立可执行文件，无依赖关系，PRIVATE 即可。
    #
    # $<$<COMPILE_LANGUAGE:CUDA>:...>：生成器表达式（Generator Expression）
    #   格式：$<条件:值>  条件为真时展开为值，否则展开为空字符串
    #   $<COMPILE_LANGUAGE:CUDA> 在编译 .cu 文件时为真，编译 .cpp 时为假
    #   作用：确保 --keep --keep-dir 只传给 nvcc，不传给 g++
    #         （g++ 不认识这两个参数，传过去会报错）
    #   若目标只含 .cu 文件（本项目情况），可省略生成器表达式直接写：
    #     target_compile_options(${target_name} PRIVATE --keep --keep-dir "${_keep_dir}")
    #   加生成器表达式是防御性写法，避免将来混入 .cpp 文件时报错。
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--keep --keep-dir "${_keep_dir}">
    )
endfunction()

# ──────────────────────────────────
# cxx_example：为纯 .cpp 示例创建可执行目标
#
# 纯 C++ 文件不调用任何 CUDA API，无需链接 CUDA::cudart。
# 若将来某个 .cpp 文件中使用了 CUDA 运行时 API（如 cudaMalloc），
# 则需要先启用 find_package(CUDAToolkit REQUIRED)，再取消下面注释：
#   target_link_libraries(${target_name} PRIVATE CUDA::cudart)
# ────────────────────────────────
function(cxx_example target_name source_file)
    add_executable(${target_name} "${source_file}")
    # target_link_libraries(${target_name} PRIVATE CUDA::cudart)
endfunction()

# ──────────────────────────────────
# find_package(CUDAToolkit)：查找并导入 CUDA 工具包
# 提供 CUDA::cudart、CUDA::cublas 等导入目标，供 target_link_libraries 使用。
# 本项目所有 .cu 文件由 LANGUAGES CUDA 驱动 nvcc 编译，cudart 会自动链接；
# 唯一的 .cpp 文件也不调用 CUDA API，因此当前无需导入。
# 若将来需要显式链接 CUDA 库（如 cublas、cufft），取消下面两行注释：
# find_package(CUDAToolkit REQUIRED)
# message(STATUS "CUDA 工具包版本：${CUDAToolkit_VERSION}  路径：${CUDAToolkit_INCLUDE_DIRS}")
# ─────────────────────────────────

# ──────────────────────────────────
# compile_commands.json（CLion 代码索引）
#
# CMAKE_EXPORT_COMPILE_COMMANDS：生成 compile_commands.json
#   记录每个源文件的完整编译命令（编译器路径、标志、包含路径等）。
#   CLion 读取此文件为每个 .cu / .cpp 提供精确的代码补全、跳转定义、错误高亮。
#   生成位置：cmake-build-debug/compile_commands.json
#   ON  → 每次 cmake configure 后自动生成/更新
#   OFF → 不生成（默认值）
# ──────────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ══════════════════════════════
# 示例目标
# 目录结构（相对于本 CMakeLists.txt）：
#   Into to CUDA C++/
#   Writing  CUDA SIMT kernel/   （注意目录名含双空格）
#   Asynchronous Execution/
#   Unified and System Memory/
# ════════════════════════════════

# ── 1. Intro to CUDA C++ ──────────────
set(INTRO_DIR "Into to CUDA C++")

cuda_example(vecAdd_explicitMemory  "${INTRO_DIR}/vecAdd_explicitMemory.cu")
cuda_example(vecAdd_unifiedMemory   "${INTRO_DIR}/vecAdd_unifiedMemory.cu")

# ── 2. Writing CUDA SIMT Kernel ──────────────
# 注意：目录名含双空格 "Writing  CUDA SIMT kernel"
set(SIMT_DIR "Writing  CUDA SIMT kernel")

cuda_example(conv3d                          "${SIMT_DIR}/3D conv.cu")
cuda_example(conv3d_cudaLaunchKernel         "${SIMT_DIR}/3D conv cudaLaunchKernel.cu")
cuda_example(conv3d_cudaLaunchKernelEx       "${SIMT_DIR}/3D conv cudaLaunchKernelEx.cu")
cuda_example(distributed_shared_memory       "${SIMT_DIR}/Distributed Shared Memory.cu")
cuda_example(dynamic_shared_memory           "${SIMT_DIR}/Dynamic Allocation of Shared Memory.cu")
cuda_example(global_memory                   "${SIMT_DIR}/Global Memory.cu")
cuda_example(global_memory_device            "${SIMT_DIR}/Global Memory device.cu")
cuda_example(global_memory_unified           "${SIMT_DIR}/Global Memory unified.cu")
cuda_example(shared_memory                   "${SIMT_DIR}/Shared Memory.cu")
cuda_example(atomic                          "${SIMT_DIR}/atomatic.cu")
cuda_example(atomic_scope                    "${SIMT_DIR}/atomic_scope.cu")
cuda_example(cluster                         "${SIMT_DIR}/cluster.cu")
cuda_example(constant_memory                 "${SIMT_DIR}/constant memory.cu")
cuda_example(cross_sector_and_bank           "${SIMT_DIR}/cross_sector_and_bank.cu")
cuda_example(global_memory_coalescing        "${SIMT_DIR}/global_memory_coalescing.cu")
cuda_example(matrix_transpose_gm             "${SIMT_DIR}/matrix_transpose_with_GM.cu")
cuda_example(matrix_transpose_sm             "${SIMT_DIR}/matrix_transpose_with_sm.cu")
cuda_example(multi_cluster                   "${SIMT_DIR}/multi_cluster.cu")
cuda_example(shared_memory_bank_conflict     "${SIMT_DIR}/shared_memory_bank_conflict.cu")

# ── 3. Asynchronous Execution ──────────
set(ASYNC_DIR "Asynchronous Execution")

# cuda_example(cuda_default_stream   "${ASYNC_DIR}/cuda_default_stream.cu")
#              │                      └─ 源文件路径：相对于本文件所在目录（CMAKE_CURRENT_SOURCE_DIR）
#              │                         展开为：.../Programming GPUs in CUDA/Asynchronous Execution/cuda_default_stream.cu
#              └─ 目标名：不是路径，是编译产物的名字
#                 可执行文件输出到构建目录（CMAKE_CURRENT_BINARY_DIR）：cmake-build-debug/cuda_default_stream
cuda_example(cuda_default_stream       "${ASYNC_DIR}/cuda_default_stream.cu")
cuda_example(cuda_event                "${ASYNC_DIR}/cuda_event.cu")
cuda_example(cuda_free_comparison      "${ASYNC_DIR}/cuda_free_comparison.cu")
cuda_example(cuda_graph                "${ASYNC_DIR}/cuda_graph.cu")
cuda_example(cuda_host_alloc           "${ASYNC_DIR}/cuda_host_alloc_examples.cu")
cuda_example(cuda_host_func_data       "${ASYNC_DIR}/cuda_host_func_data.cu")
cuda_example(cuda_host_get_device_ptr  "${ASYNC_DIR}/cuda_host_get_device_pointer.cu")
cuda_example(cuda_host_register        "${ASYNC_DIR}/cuda_host_register.cu")
cuda_example(cuda_launch_host_func     "${ASYNC_DIR}/cuda_launch_host_func.cu")
cuda_example(cuda_memcpy_async         "${ASYNC_DIR}/cuda_memcpy_async.cu")
cuda_example(cuda_pdl                  "${ASYNC_DIR}/cuda_pdl.cu")
cuda_example(cuda_pipeline_hostfunc    "${ASYNC_DIR}/cuda_pipeline_hostfunc.cu")
cuda_example(cuda_stream               "${ASYNC_DIR}/cuda_stream.cu")
cuda_example(hostfunc_unified_mem      "${ASYNC_DIR}/hostfunc_unified_mem.cu")

# ptr_init_test 是纯 C++ 文件（.cpp），使用 cxx_example
cxx_example(ptr_init_test              "${ASYNC_DIR}/ptr_init_test.cpp")

# ── 4. Unified and System Memory ────────
set(UNI_DIR "Unified and System Memory")

cuda_example(pointer_attributes    "${UNI_DIR}/pointer_attributes.cu")
cuda_example(prefetch_and_advise   "${UNI_DIR}/prefetch_and_advise.cu")
cuda_example(peer_access           "${UNI_DIR}/peer_access.cu")
cuda_example(device_properties_um  "${UNI_DIR}/device_properties_um.cu")
cuda_example(managed_alloc_compare "${UNI_DIR}/managed_alloc_compare.cu")
cuda_example(pinned_host_kernel    "${UNI_DIR}/pinned_host_kernel.cu")

message(STATUS "配置完成，共注册以下目标：")
message(STATUS "  [Intro]       vecAdd_explicitMemory  vecAdd_unifiedMemory")
message(STATUS "  [SIMT]        conv3d  distributed_shared_memory  dynamic_shared_memory")
message(STATUS "                global_memory  shared_memory  atomic  cluster ...")
message(STATUS "  [Async]       cuda_stream  cuda_event  cuda_graph  cuda_memcpy_async ...")
message(STATUS "  [UnifiedMem]  pointer_attributes  prefetch_and_advise  peer_access  device_properties_um  managed_alloc_compare  pinned_host_kernel")

# ═══════════════════════════════════
# 完整编译调试流程：命令行 与 CLion 等价操作 对照
# ═══════════════════════════════════

# ─────────────────────────────────
# 前置：CLion 一次性配置（仅首次需要）
# ──────────────────────────────
#
#   [1] 打开项目
#       File → Open → 选择本文件所在目录（Programming GPUs in CUDA/）
#       CLion 检测到 CMakeLists.txt 后弹出提示 → 点击 "Open as Project"
#
#   [2] 配置 Toolchain（编译器路径）
#       Settings → Build, Execution, Deployment → Toolchains
#         C++ 编译器：/usr/bin/g++                    （which g++）
#         CUDA 编译器：由 LANGUAGES CUDA 自动识别，或在本文件 set(CMAKE_CUDA_COMPILER ...) 指定
#         调试器：     /usr/local/cuda/bin/cuda-gdb   （cuda-gdb，kernel 断点必须）
#
#       与本文件设置的优先级关系：
#         编译器路径（C++ / CUDA）优先级从高到低：
#           1. 本文件 set(CMAKE_CUDA_COMPILER ...)  ← 最高，当前已显式指定，覆盖一切
#           2. cmake 命令行 -DCMAKE_CUDA_COMPILER=...
#           3. CLion Toolchains                     CLion 将其转为第 2 级命令行参数
#           4. 系统 PATH 自动探测（which nvcc）     最低优先级
#           → 本项目已显式指定，CLion Toolchains 的编译器配置对本项目无效
#
#         调试器（cuda-gdb）：
#           CMakeLists.txt 中无对应设置，CLion Toolchains → Debugger 是唯一入口
#           cmake 的 -G -g 标志只负责在二进制中嵌入调试信息，两者缺一不可
#
#   [3] 新增 CMake Profile
#       Settings → Build, Execution, Deployment → CMake → +
#         Profile "Debug"：
#           Build type:      Debug
#           CMake options:   -G "Unix Makefiles"     （或留空使用默认 Ninja）
#           Build directory: cmake-build-debug
#           Build options:   -- -j 18                （-- 是分隔符，-j 18 传给 make/ninja）
#         Profile "Release"：
#           Build type:      Release
#           Build directory: cmake-build-release
#           Build options:   -- -j 18
#
#       CMake options 字段与本文件的优先级：
#         CLion CMake options 中的 -DKEY=VALUE 优先级高于本文件 set(KEY VALUE)
#         例如填 -DCMAKE_CUDA_ARCHITECTURES=89 会覆盖本文件的 set(...native)
#         本文件已显式 set() 的变量，不要在 CMake options 中重复填写
#
#       CLion 自动重新 configure 的时机：
#         · 保存 CMakeLists.txt 后自动触发
#         · 右键 CMakeLists.txt → Reload CMake Project 可手动触发
#         · configure 完成后目标列表、代码索引自动更新
#
#   [4] 创建 Run/Debug Configuration
#       右上角下拉框 → Edit Configurations → + → CMake Application
#         Name:          pointer_attributes（或任意目标名）
#         Target:        pointer_attributes（从下拉列表选择）
#         CMake profile: Debug

# ───────────────────────────────
# make / nvcc / g++ 的关系
# ──────────────────────────────
#
#   三者是上下级调用关系，职责完全不同：
#
#   make                      构建编排器，读取 cmake 生成的 Makefile，
#                             决定调用哪个编译器、以什么顺序、哪些文件需要重新编译。
#                             make 本身不编译任何代码。
#     ├── 调用 nvcc           CUDA 编译器，将 .cu 编译成目标文件 .o
#     └── 调用 g++            C++ 编译器，将 .cpp 编译成目标文件 .o
#           └── g++ 负责链接  将所有 .o 合并成最终可执行文件
#
#   CLion 中点击"锤子（Build）"等价于 make，
#   make 再依次调用 nvcc 和 g++，用户感知到的是一个动作，底层是三层调用。

# ───────────────────────────────────
# 清理构建产物
# ───────────────────────────────────
#
#   命令行：
#     make clean          删除编译产物（.o 和可执行文件），保留 Makefile 和 CMakeCache.txt
#     rm -rf build/       删除整个构建目录，相当于完全重置（包括 CMakeCache.txt）
#
#   CLion 等价操作：
#     Build → Clean Project         清理所有目标的编译产物（等价于 make clean）
#     Build → Clean 'target_name'   只清理指定目标
#     删除 cmake-build-debug/ 目录   完全重置，等价于 rm -rf build/
#                                    下次 configure 会重新生成所有构建文件

# ───────────────────────────────────
# 步骤 1：cmake configure（生成构建文件）
# ────────────────────────────────────
#
#   命令行：
#     mkdir build && cd build
#     cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
#     │     │                    └─ 向 CMakeCache.txt 写入变量
#     │     │                       激活 CMAKE_CUDA_FLAGS_DEBUG / CMAKE_CXX_FLAGS_DEBUG
#     │     │                       -G -g -O0 固化进生成的 Makefile，make 无需再传参
#     │     └─ 指定生成器：Unix Makefiles → 生成 Makefile
#     │        不指定则默认 Ninja → 生成 build.ninja（CLion 默认用 Ninja）
#     └─ cmake 只生成构建文件，不执行编译
#        ".." 相对于当前 shell 目录，指向 CMakeLists.txt 所在的项目根
#

#   CLion 在以下时机触发 configure：
#      1. 打开项目时          → 自动 cmake configure（生成构建系统）
#      2. CMakeLists.txt 修改 → 右上角弹出"Reload CMake Project"，点击后重新 configure
#      3. 手动触发            → Tools → CMake → Reset Cache and Reload Project
#
#
# ── CLion / CMakeLists.txt / cmake 三者的交互与决策流程 ──────────
#
# ── 哪些参数能在 CMakeLists.txt 里设置？─────────────────────────
#
#   cmake 的参数分为两类，能否在文件中设置完全不同：
#
#   ① cmake 可执行文件自身的启动参数（-G、-B、-S）
#       在 cmake 读取任何 CMakeLists.txt 之前就已解析并固定，
#       文件里的 set() 对它们没有任何效果：
#         -G "Unix Makefiles" → cmake 决定生成 Makefile，之后才读文件
#         -B cmake-build-debug → cmake 确定构建目录，之后才读文件
#       这两个参数只能由调用 cmake 的一方（CLion Profile / 命令行）指定，
#       CMakeLists.txt 无法覆盖。
#
#   ② CMake 变量（-DCMAKE_BUILD_TYPE 等）
#       可以在 CMakeLists.txt 里用 set() 设置，但存在优先级冲突问题：
#
#       CMake 变量有两种存储层次：
#
#         普通变量：set(VAR value)
#           · 存在于 cmake 进程的内存中，每次 configure 重新执行文件时重新赋值
#           · 不持久化，cmake 进程退出后消失
#           · 作用域：当前 CMakeLists.txt 及其子目录
#
#         缓存变量：set(VAR value CACHE TYPE "说明") 或命令行 -DVAR=value
#           · 写入磁盘上的 cmake-build-debug/CMakeCache.txt
#           · 跨次 configure 持久保留：第二次 configure 时，若不重新传 -D，
#             CMakeCache.txt 里的旧值仍然有效
#           · 作用域：整个构建目录，所有子目录均可见
#
#       cmake 变量查找规则（两层作用域，普通 → 缓存）：
#         cmake 读到 ${VAR} 时，先在当前作用域查普通变量，找不到再查缓存变量。
#         但 set() 写普通变量时，若同名缓存变量已存在，
#         cmake 在读取 ${VAR} 时优先返回普通变量的值（普通变量遮蔽缓存变量）。
#
#         "缓存变量优先"指的是另一种场景——命令行 -D 的时机早于文件执行：
#           cmake 启动 → 解析命令行 -DCMAKE_BUILD_TYPE=Debug → 写入 CMakeCache.txt
#                ↓
#           cmake 开始读 CMakeLists.txt
#                ↓
#           文件里 set(CMAKE_BUILD_TYPE Release)  ← 此时才执行，写入普通变量
#                ↓
#           文件读完，cmake 实际使用的值：
#             · 若后续代码读 ${CMAKE_BUILD_TYPE}：读到普通变量 "Release"（遮蔽了缓存）
#             · 但 _DEBUG / _RELEASE 后缀的条件标志（CMAKE_CUDA_FLAGS_DEBUG 等）
#               由 cmake 在解析完文件后统一根据 CMakeCache.txt 中的值决定，
#               此时缓存变量 "Debug" 胜出，调试标志生效
#
#         实际效果上，-D 命令行传入的值控制了构建类型的实质行为（调试标志等），
#         文件里的 set() 只影响文件内部 ${CMAKE_BUILD_TYPE} 的字符串读取，
#         因此习惯上说"缓存变量（-D）优先"，文件里直接 set() 容易造成混乱。
#
#       结论：本文件不 set(CMAKE_BUILD_TYPE)，完全交给 CLion Profile 控制。
#       若要提供命令行默认值（不影响 CLion）：
#         if(NOT CMAKE_BUILD_TYPE)
#             set(CMAKE_BUILD_TYPE Release)  # 缓存中无值时才生效
#         endif()
#
# ── 完整交互流程 ─────────────────────────────────────────────────
#
#   第一步：CLion 读取 CMake Profile（Settings → Build → CMake）
#     CLion Profile 提供以下输入，其余全部由 CMakeLists.txt 决定：
#       · Build type（Debug / Release）→ 决定 -DCMAKE_BUILD_TYPE=xxx

#       · Generator（Unix Makefiles / Ninja）→ 决定 -G xxx
#       · Build directory → 决定 -B <dir>（CLion 默认 cmake-build-debug）
#       · CMake options（用户在 Profile 里手动填写的额外 -D 参数）
#
#   第二步：CLion 组装并执行 cmake configure 命令
#     CLion 将 Profile 设置翻译成命令行参数，执行：
#       cmake \
#         -DCMAKE_BUILD_TYPE=Debug \        ← 来自 Profile "Build type"（缓存变量）
#         -G "Unix Makefiles" \             ← 来自 Profile "Generator"（启动参数，文件不可覆盖）
#         -S . \                            ← 源码目录（CLion 固定传入，启动参数）
#         -B cmake-build-debug              ← 来自 Profile "Build directory"（启动参数，文件不可覆盖）
#     注意：CMAKE_CUDA_COMPILER / CMAKE_CXX_COMPILER 不是 CLion 传入的，
#           而是本文件 set() 显式写死，cmake 读文件时自己读到，
#           普通变量在此场景下不存在缓存变量覆盖问题（CLion 不传这两个 -D）。
#
#   第三步：cmake 解析 CMakeLists.txt，做出编译决策
#     cmake 按顺序执行本文件每条指令：
#       · project(... LANGUAGES CXX CUDA) → 探测并注册 nvcc / g++ 编译器
#       · set(CMAKE_CXX_STANDARD 20)      → 后续所有目标默认用 C++20
#       · set(CMAKE_CUDA_ARCHITECTURES native) → nvcc 自动探测本机 SM 版本
#       · set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")  → Debug 模式追加 nvcc 调试标志
#       · cuda_example(target src)        → 每调用一次，注册一个编译目标：
#             add_executable            → 记录"目标名 ↔ 源文件"映射
#             set_target_properties     → 为该目标固化 GPU 架构
#             target_compile_options    → 为该目标追加 --keep --keep-dir
#             file(MAKE_DIRECTORY ...)  → 立即在磁盘创建 temp/<目标名>/ 目录
#
#   第四步：cmake 将决策写入构建目录
#     · cmake-build-debug/CMakeCache.txt  → 所有变量的键值快照（可用 cmake -LA 查看）
#     · cmake-build-debug/Makefile        → 每个目标的完整编译/链接命令
#       （含 nvcc 的全部参数：-arch、-std、-G、--keep、--keep-dir 等，全部固化在此）
#     · cmake-build-debug/temp/<目标名>/  → 已由 file(MAKE_DIRECTORY) 预先创建
#     之后 make / CLion Build 按钮只读取这份 Makefile，不再解析 CMakeLists.txt。
#     CMakeLists.txt 修改后必须重新 configure 才能生效。
#
#   完整决策来源汇总：
#     ┌──────────────────────────────┬──────────────────┬───────────────────────────┐
#     │ 参数                         │ 来源             │ 文件能否覆盖              │
#     ├──────────────────────────────┼──────────────────┼───────────────────────────┤
#     │ -G（生成器）                  │ CLion Profile    │ ✗ 启动参数，文件无效      │
#     │ -B（构建目录）                │ CLion Profile    │ ✗ 启动参数，文件无效      │
#     │ -DCMAKE_BUILD_TYPE           │ CLion Profile    │ ✗ 缓存变量优先，set() 被忽略│
#     │ CMAKE_CUDA_COMPILER          │ 本文件 set()     │ ✓ CLion 不传此 -D         │
#     │ CMAKE_CXX_COMPILER           │ 本文件 set()     │ ✓ CLion 不传此 -D         │
#     │ CMAKE_CUDA_STANDARD          │ 本文件 set()     │ ✓ CLion 不传此 -D         │
#     │ CMAKE_CUDA_ARCHITECTURES     │ 本文件 set()     │ ✓ CLion 不传此 -D         │
#     │ CMAKE_CUDA_FLAGS_DEBUG       │ 本文件 set()     │ ✓ CLion 不传此 -D         │
#     │ --keep --keep-dir            │ 本文件 cuda_example│ ✓ target 级，CLion 不干预│
#     │ temp/<目标名>/ 目录创建       │ 本文件 cuda_example│ ✓ configure 阶段执行     │
#     └──────────────────────────────┴──────────────────┴───────────────────────────┘
# ──────────────────────────────────────────────────────────────────

#   CLion 等价操作：
#     右上角 Profile 下拉框切换到 "Debug"
#     → CLion 自动执行：
#         cmake -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles"
#               -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
#               -DCMAKE_CXX_COMPILER=/usr/bin/g++
#               -S .                  # 源码目录（CMakeLists.txt 所在位置）
#               -B cmake-build-debug  # 构建目录，不存在则自动创建
#                                     # 等价于：mkdir cmake-build-debug && cd cmake-build-debug && cmake ..
#                                     # 每个 Profile 对应独立的 -B 目录，Debug/Release 互不干扰
#     → 读取本文件所有 set() / project() / message()，输出 "-- " 前缀的配置信息
#     → 生成 cmake-build-debug/Makefile 和 CMakeCache.txt
#     → 填充右上角目标下拉列表（35 个目标）


#
#   make 命令说明：
#     make -j$(nproc)    并行编译所有目标，$(nproc) 为 CPU 核心数
#                        CLion Building Options "-- -j 18" 等价于此
#     make <target>      只编译指定目标，例如 make pointer_attributes
#     make VERBOSE=1     打印完整命令行，用于排查编译参数问题

# ───────────────────────────────
# 步骤 2：编译 .cu 文件（make → nvcc）
# ───────────────────────────────
#
#   命令行（make VERBOSE=1 可查看完整命令）：
#     /usr/local/cuda/bin/nvcc \
#       -forward-unknown-to-host-compiler \  # 未知 flag 转发给底层 g++，而非报错
#       -G -g \                              # Debug：device 调试信息 + host 调试信息
#       -std=c++20 \                         # CUDA 标准（CMAKE_CUDA_STANDARD）
#       -arch=native \                       # GPU 架构（CMAKE_CUDA_ARCHITECTURES）
#       -x cu \                              # 强制按 CUDA 文件解析
#       -c "Unified and System Memory/pointer_attributes.cu" \
#       #   -c：只编译，不链接；输出必须是 .o 中间目标文件，而非最终可执行文件
#       -o CMakeFiles/pointer_attributes.dir/.../pointer_attributes.cu.o
#       #   -o 此处是中间目标文件（.o），不是最终可执行文件
#       #   最终可执行文件由步骤 4 的 g++ 链接生成：
#       #     编译（nvcc -c）：.cu → .o              中间产物，不可直接运行
#       #     链接（g++）：    .o  → pointer_attributes   最终产物，可直接运行
#
#   CLion 等价操作：
#     点击锤子图标（Build）或 Ctrl+F9，选择目标 pointer_attributes
#     → CLion 调用 ninja pointer_attributes（或 make pointer_attributes）
#     → 上述 nvcc 命令由生成的构建文件自动驱动

# ──────────────────────────────────
# 步骤 3：编译 .cpp 文件（make → g++）
# ──────────────────────────────────
#
#   命令行：
#     /usr/bin/g++ \
#       -g -O0 \                             # Debug：调试信息 + 关闭优化
#       -std=c++20 \                         # C++ 标准（CMAKE_CXX_STANDARD）
#       -c "Asynchronous Execution/ptr_init_test.cpp" \
#       -o CMakeFiles/ptr_init_test.dir/.../ptr_init_test.cpp.o
#
#   CLion 等价操作：同步骤 2，选择目标 ptr_init_test

# ────────────────────────────────
# 步骤 4：链接 .cu 目标（make → g++）
# ────────────────────────────────
#
#   命令行（g++ 负责链接，自动附加 CUDA 库路径）：
#     /usr/bin/g++ \
#       CMakeFiles/pointer_attributes.dir/.../pointer_attributes.cu.o \
#       -o pointer_attributes \
#       -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
#       -L"/usr/local/cuda/targets/x86_64-linux/lib"
#
#   CLion 等价操作：链接由 Build 步骤自动完成，无需额外操作

# ────────────────────────────────
# 步骤 5：链接 .cpp 目标（make → g++）
# ─────────────────────────────────
#
#   命令行（纯 g++ 链接，无 CUDA 库路径）：
#     /usr/bin/g++ \
#       -g -O0 \
#       CMakeFiles/ptr_init_test.dir/.../ptr_init_test.cpp.o \
#       -o ptr_init_test

# ───────────────────────────
# 步骤 6：运行
# ───────────────────────────
#
#   命令行：
#     ./pointer_attributes
#
#   CLion 等价操作：
#     点击运行图标或 Shift+F10
#     → 在 Run 工具窗口显示程序输出

# ───────────────────────────
# 步骤 7：调试（cuda-gdb）
# ──────────────────────────
#
#   CLion 使用方式（两步缺一不可）：
#     第一步（本文件已配置）：编译时通过 -G -g 将调试信息嵌入二进制
#     第二步（CLion 手动配置）：Settings → Toolchains → Debugger 指定 cuda-gdb
#
#   命令行：
#     /usr/local/cuda/bin/cuda-gdb ./pointer_attributes
#     (cuda-gdb) break pointer_attributes.cu:42    # 在第 42 行设断点
#     (cuda-gdb) run
#     (cuda-gdb) info cuda threads                 # 查看所有 GPU 线程状态
#
#   CLion 等价操作：
#     在 .cu 文件行号左侧单击设置断点（红点）
#     点击虫子图标或 Shift+F9
#     → CLion 调用 cuda-gdb，在断点处中断 GPU 线程
#     → Variables 窗口查看线程局部变量
#     → Frames 窗口查看调用栈

# ───────────────────────────────────────
# 补充：其他 CLion 与 CMakeLists.txt 的交互
# ───────────────────────────────────────
#
#   [A] CMakeCache.txt 的覆盖关系
#       cmake configure 后生成 cmake-build-debug/CMakeCache.txt，
#       缓存中的变量值优先级高于本文件的 set()。
#       CLion 提供 GUI 编辑入口：
#         CMake 工具窗口（左下角）→ 右键目标 → CMake Settings
#         或直接编辑 cmake-build-debug/CMakeCache.txt
#       修改后需 Reload CMake Project 才生效。
#       若想清除缓存恢复本文件的 set() 值：删除整个 cmake-build-debug/ 后重新 configure。
#
#   [B] add_executable 目标名 → CLion Run/Debug 下拉列表
#       本文件每条 cuda_example(...) / cxx_example(...) 中的目标名，
#       直接对应 CLion 右上角 Run/Debug Configuration 下拉框中的可选目标。
#       目标名改变后需重新 configure，CLion 才更新列表。
#
#   [C] compile_commands.json → CLion 代码索引
#       cmake 可生成 compile_commands.json，记录每个文件的完整编译命令。
#       CLion 读取此文件为每个 .cu / .cpp 提供精确的代码补全、跳转、错误高亮。
#       本文件已启用：set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#       生成位置：cmake-build-debug/compile_commands.json
#       CLion 自动读取，无需额外配置。
#
#   [D] Run/Debug Configuration 的工作目录
#       CLion Run/Debug Configuration → Working directory 默认为构建目录
#       （cmake-build-debug/），即可执行文件所在位置。
#       若程序需要读取相对路径的文件，需在此字段改为项目根目录或源码目录。
# ═════════════════════════════════
