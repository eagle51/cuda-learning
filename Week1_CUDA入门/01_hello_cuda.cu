/**
 * 01_hello_cuda.cu
 * 第一个CUDA程序 - 理解Grid/Block/Thread
 * 
 * 编译: nvcc 01_hello_cuda.cu -o 01_hello_cuda
 * 运行: 01_hello_cuda.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif
/**
 * Kernel函数：在GPU上执行
 * __global__ 表示这是一个kernel函数
 */
__global__ void hello_cuda() {
    // blockIdx.x: 当前block的索引
    // blockDim.x: 每个block的thread数量
    // threadIdx.x: 当前thread在block中的索引
    
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Hello from block %d, thread %d, global idx %d\n",
           blockIdx.x, threadIdx.x, idx);
}

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== CUDA Hello World ===\n\n");
    
    // 配置kernel启动参数
    int blocks = 2;          // 2个blocks
    int threads_per_block = 4;  // 每个block 4个threads
    
    printf("启动配置:\n");
    printf("  Blocks: %d\n", blocks);
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Total threads: %d\n\n", blocks * threads_per_block);
    
    // 启动kernel
    // <<<blocks, threads>>> 是CUDA的特殊语法
    hello_cuda<<<blocks, threads_per_block>>>();
    CUDA_CHECK_KERNEL();
    
    printf("\n=== 程序结束 ===\n");
    
    return 0;
}

/*
 * 练习:
 * 1. 修改为3个blocks，5个threads，观察输出
 * 2. 计算总共启动了多少个threads？
 * 3. 理解全局索引的计算公式
 * 
 * 预期输出:
 * Block 0:
 *   Thread 0 -> Global idx 0
 *   Thread 1 -> Global idx 1
 *   Thread 2 -> Global idx 2
 *   Thread 3 -> Global idx 3
 * Block 1:
 *   Thread 0 -> Global idx 4
 *   Thread 1 -> Global idx 5
 *   Thread 2 -> Global idx 6
 *   Thread 3 -> Global idx 7
 */
