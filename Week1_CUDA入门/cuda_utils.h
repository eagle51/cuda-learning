/**
 * cuda_utils.h
 * CUDA工具函数库 - 错误检查和GPU信息
 * 
 * 使用方法:
 * #include "cuda_utils.h"
 * CUDA_CHECK(cudaMalloc(&ptr, size));
 * CUDA_CHECK_KERNEL();
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================
// CUDA错误检查宏
// ============================================

/**
 * 检查CUDA API调用
 * 用法: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "❌ CUDA错误 in %s at line %d:\n", \
                __FILE__, __LINE__); \
        fprintf(stderr, "   %s\n", cudaGetErrorString(err)); \
        fprintf(stderr, "   调用: %s\n", #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * 检查Kernel执行
 * 用法: 
 * kernel<<<blocks, threads>>>(args);
 * CUDA_CHECK_KERNEL();
 */
#define CUDA_CHECK_KERNEL() \
do { \
    /* 检查kernel启动错误 */ \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "❌ Kernel启动错误 in %s at line %d:\n", \
                __FILE__, __LINE__); \
        fprintf(stderr, "   %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
    /* 同步并检查kernel执行错误 */ \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "❌ Kernel执行错误 in %s at line %d:\n", \
                __FILE__, __LINE__); \
        fprintf(stderr, "   %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================
// GPU信息查询
// ============================================

/**
 * 打印GPU设备信息
 */
inline void print_gpu_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("========================================================\n");
    printf("                   GPU Device Info                      \n");
    printf("========================================================\n");
    printf("Device Name        : %s\n", prop.name);
    printf("Compute Capability : %d.%d\n", prop.major, prop.minor);
    printf("Global Memory      : %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("SM Count           : %d\n", prop.multiProcessorCount);
    printf("Shared Mem/Block   : %.0f KB\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("Max Threads/Block  : %d\n", prop.maxThreadsPerBlock);
    printf("Max Block Dim      : (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Dim       : (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp Size          : %d\n", prop.warpSize);
    printf("========================================================\n\n");
}

/**
 * 打印简短的GPU信息
 */
inline void print_gpu_info_short() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU: %s (计算能力 %d.%d, %.2f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

// ============================================
// 内存工具
// ============================================

/**
 * 安全的内存分配（自动检查）
 */
inline void* cuda_malloc_safe(size_t bytes, const char* name = "data") {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

/**
 * 安全的内存拷贝（自动检查）
 */
inline void cuda_memcpy_safe(void* dst, const void* src, size_t bytes,
                             cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

/**
 * 安全的内存释放（自动检查）
 */
inline void cuda_free_safe(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

#endif // CUDA_UTILS_H

/*
 * 使用示例:
 * 
 * #include "cuda_utils.h"
 * 
 * int main() {
 *     // 打印GPU信息
 *     print_gpu_info();
 *     
 *     // 分配内存（自动错误检查）
 *     float *d_data;
 *     CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));
 *     
 *     // 拷贝数据
 *     CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, 
 *                           cudaMemcpyHostToDevice));
 *     
 *     // 启动kernel
 *     kernel<<<blocks, threads>>>(d_data);
 *     CUDA_CHECK_KERNEL();
 *     
 *     // 释放内存
 *     CUDA_CHECK(cudaFree(d_data));
 *     
 *     return 0;
 * }
 * 
 * 好处:
 * 1. 自动检查所有CUDA调用
 * 2. 错误信息详细（文件、行号、调用）
 * 3. 快速定位问题
 * 4. 提高代码质量
 */
