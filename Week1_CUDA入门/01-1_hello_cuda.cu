#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif

// 最简单的kernel，每个thread打印自己的信息
__global__ void hello_kernel() {
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from thread %d (block %d, thread %d)\n",
        idx, blockIdx.x, threadIdx.x);
}

int main() {
    // 设置控制台输出为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("启动GPU kernel...\n");

    // 配置：2个blocks，每个block 4个threads
    int blocks = 2;
    int threads_per_block = 4;

    // 启动kernel
    hello_kernel<<<blocks, threads_per_block>>>();
	CUDA_CHECK_KERNEL();

    printf("GPU执行完成\n");

    return 0;
}
