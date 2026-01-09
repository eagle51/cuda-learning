/**
 * 02_memory_basic.cu
 * CUDA内存管理基础 - Host和Device内存操作
 * 
 * 编译: nvcc 02_memory_basic.cu -o 02_memory_basic
 * 运行: 02_memory_basic.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif
/**
 * Kernel: 将数组中的每个元素乘以2
 */
__global__ void double_array(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== CUDA内存管理演示 ===\n\n");
    
    // 数组大小
    const int n = 10;
    const int bytes = n * sizeof(float);
    
    printf("数组大小: %d 个元素\n", n);
    printf("内存大小: %d 字节\n\n", bytes);
    
    // === 1. 分配Host内存 (CPU内存) ===
    float *h_data = new float[n];  // h_ 表示 host
    
    // 初始化数据
    printf("初始数据 (CPU):\n");
    for (int i = 0; i < n; i++) {
        h_data[i] = i * 1.0f;
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");
    
    // === 2. 分配Device内存 (GPU内存) ===
    float *d_data;  // d_ 表示 device
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    printf("✓ 在GPU上分配了 %d 字节内存\n", bytes);
    
    // === 3. 将数据从Host拷贝到Device ===
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    printf("✓ 数据从CPU拷贝到GPU\n\n");
    
    // === 4. 在GPU上处理数据 ===
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    printf("启动kernel处理数据...\n");
    printf("  配置: %d blocks × %d threads\n", blocks, threads);
    
    double_array<<<blocks, threads>>>(d_data, n);
    CUDA_CHECK_KERNEL();
    
    printf("✓ GPU处理完成\n\n");
    
    // === 5. 将结果从Device拷贝回Host ===
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("✓ 结果从GPU拷贝回CPU\n\n");
    
    // 显示结果
    printf("处理后数据 (CPU):\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");
    
    // === 6. 释放内存 ===
    delete[] h_data;  // 释放Host内存
    CUDA_CHECK(cudaFree(d_data));  // 释放Device内存
    
    printf("✓ 内存已释放\n");
    printf("\n=== 程序结束 ===\n");
    
    return 0;
}

/*
 * 关键概念:
 * 
 * 1. Host (CPU) vs Device (GPU):
 *    - Host memory: CPU可以访问
 *    - Device memory: GPU可以访问
 *    - 两者不能直接互访，需要显式拷贝
 * 
 * 2. 内存管理三步曲:
 *    - cudaMalloc: 在GPU上分配内存
 *    - cudaMemcpy: 在CPU和GPU间传输数据
 *    - cudaFree: 释放GPU内存
 * 
 * 3. cudaMemcpy的方向:
 *    - cudaMemcpyHostToDevice: CPU → GPU
 *    - cudaMemcpyDeviceToHost: GPU → CPU
 *    - cudaMemcpyDeviceToDevice: GPU → GPU
 * 
 * 4. 数据流动:
 *    CPU内存 → GPU内存 → GPU处理 → GPU内存 → CPU内存
 */
