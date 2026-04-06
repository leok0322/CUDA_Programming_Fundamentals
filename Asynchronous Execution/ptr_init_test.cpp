/**
 * ptr_init_test.cpp
 *
 * 验证：全局指针 vs 局部指针的初始化行为
 *
 * 编译：
 *   g++ -std=c++17 -o ptr_init_test ptr_init_test.cpp
 *
 * 结论：
 *   全局指针 → 零初始化 → nullptr
 *   局部指针 → 不初始化 → 栈上垃圾值
 */

#include <stdio.h>

// ═════════════════════
// C++ 类型系统简述
//
// 内置类型（built-in type）：int、float、double、bool、指针（T*）
//   · 没有构造函数
//   · 函数内局部声明：不初始化，值是栈上残留的垃圾值
//   · 全局/静态变量：零初始化（int→0，指针→nullptr）
//
// 类类型（class type）：class/struct（有构造函数）
//   · 有默认构造函数
//   · 无论局部还是全局，声明时都调用默认构造函数
//   · 例：std::string s; → 空字符串（构造函数保证）
//
// 注意：struct*（指向 struct 的指针）是内置类型，struct 本身才是类类型
//   struct Foo { int x; };   // Foo 是类类型
//   Foo  obj;                // 类类型，调用构造函数
//   Foo *ptr;                // 指针，内置类型，不初始化
// ════════════════════════


// ── 全局变量：无论内置类型还是类类型，都零初始化 ─────────────────────────
//
// C++ 标准规定：所有具有静态存储期（static storage duration）的变量
// 在程序启动时零初始化：
//   · 全局变量（global）
//   · 静态局部变量（static local）
//   · 静态成员变量（static member）
// 指针零初始化 = nullptr，整数零初始化 = 0

int   *g_int_ptr;       // 全局 int 指针，零初始化 → nullptr
void  *g_void_ptr;      // 全局 void 指针，零初始化 → nullptr
int    g_int;           // 全局 int，零初始化 → 0

struct MyStruct { int x; int y; };
MyStruct *g_struct_ptr; // 全局 struct 指针（内置类型），零初始化 → nullptr


int main()
{
    printf("══ 全局指针（静态存储期，零初始化）══\n");

    // %p 打印指针地址；nullptr 打印为 0x0 或 (nil)
    printf("  g_int_ptr    = %p  (期望 nullptr)\n", (void *)g_int_ptr);
    printf("  g_void_ptr   = %p  (期望 nullptr)\n", g_void_ptr);
    printf("  g_struct_ptr = %p  (期望 nullptr)\n", (void *)g_struct_ptr);
    printf("  g_int        = %d  (期望 0)\n",        g_int);

    // nullptr 判断
    printf("  g_int_ptr == nullptr : %s\n",
           g_int_ptr    == nullptr ? "true" : "false");
    printf("  g_void_ptr == nullptr: %s\n",
           g_void_ptr   == nullptr ? "true" : "false");

    printf("\n══ 局部指针（自动存储期，不初始化，垃圾值）══\n");

    // ── 局部指针：不初始化，栈上残留值 ───────────────────────────────────
    //
    // 警告：读取未初始化变量是未定义行为（undefined behavior）。
    // 编译器可能发出警告：warning: 'local_ptr' is used uninitialized
    // 此处仅用于演示，生产代码中绝不应读取未初始化变量。
    //
    // 用 volatile 阻止编译器将未初始化读取优化掉或报错
    // （部分编译器会将未初始化读直接优化为 0，volatile 强制实际读栈内存）
    int   *local_int_ptr;
    void  *local_void_ptr;
    MyStruct *local_struct_ptr;

    // 通过 volatile 中间变量读取，防止编译器优化
    volatile int   *vi = local_int_ptr;
    volatile void  *vv = local_void_ptr;
    volatile MyStruct *vs = local_struct_ptr;

    printf("  local_int_ptr    = %p  (期望垃圾值，非 nullptr)\n",
           (void *)vi);
    printf("  local_void_ptr   = %p  (期望垃圾值，非 nullptr)\n",
           (void *)vv);
    printf("  local_struct_ptr = %p  (期望垃圾值，非 nullptr)\n",
           (void *)vs);

    printf("\n══ 静态局部变量（静态存储期，零初始化）══\n");

    // static 局部变量：虽然在函数内，但具有静态存储期，零初始化
    static int   *s_int_ptr;
    static void  *s_void_ptr;

    printf("  static local int_ptr  = %p  (期望 nullptr)\n",
           (void *)s_int_ptr);
    printf("  static local void_ptr = %p  (期望 nullptr)\n",
           s_void_ptr);
    printf("  s_int_ptr == nullptr  : %s\n",
           s_int_ptr == nullptr ? "true" : "false");

    printf("\n══ {} 显式初始化对比 ══\n");

    // 显式值初始化：= {} 或 {} 强制零初始化，不依赖存储期
    int   *init_ptr{};        // 值初始化 → nullptr
    void  *init_void_ptr{};   // 值初始化 → nullptr
    int    init_int{};        // 值初始化 → 0

    printf("  int*  init_ptr{}      = %p  (期望 nullptr)\n",
           (void *)init_ptr);
    printf("  void* init_void_ptr{} = %p  (期望 nullptr)\n",
           init_void_ptr);
    printf("  int   init_int{}      = %d  (期望 0)\n",
           init_int);

    return 0;
}


// liam@leo:~/cpp_linux/CUDA Programming Guide/Programming GPUs in CUDA/Asynchronous Execution$ ./ptr
// ══ 全局指针（静态存储期，零初始化）══
//   g_int_ptr    = (nil)  (期望 nullptr)
//   g_void_ptr   = (nil)  (期望 nullptr)
//   g_struct_ptr = (nil)  (期望 nullptr)
//   g_int        = 0  (期望 0)
//   g_int_ptr == nullptr : true
//   g_void_ptr == nullptr: true

// ══ 局部指针（自动存储期，不初始化，垃圾值）══
//   local_int_ptr    = 0x7ffcfb7fb000  (期望垃圾值，非 nullptr)
//   local_void_ptr   = 0x2  (期望垃圾值，非 nullptr)
//   local_struct_ptr = 0x7ffcfb7ee1a9  (期望垃圾值，非 nullptr)

// ══ 静态局部变量（静态存储期，零初始化）══
//   static local int_ptr  = (nil)  (期望 nullptr)
//   static local void_ptr = (nil)  (期望 nullptr)
//   s_int_ptr == nullptr  : true

// ══ {} 显式初始化对比 ══
//   int*  init_ptr{}      = (nil)  (期望 nullptr)
//   void* init_void_ptr{} = (nil)  (期望 nullptr)
//   int   init_int{}      = 0  (期望 0)