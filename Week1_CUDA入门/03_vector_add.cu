/**
 * 03_vector_add.cu
 * 向量加法 - 第一个真正的GPU计算程序
 * 
 * 功能: 计算 c[i] = a[i] + b[i]
 * 编译: nvcc 03_vector_add.cu -o 03_vector_add
 * 运行: 03_vector_add.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif
/**
 * Kernel: 向量加法
 * @param a 输入向量a
 * @param b 输入向量b
 * @param c 输出向量c (c = a + b)
 * @param n 向量长度
 */
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查 (非常重要!)
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== 向量加法 (GPU) ===\n\n");
    
    // === 1. 设置向量大小 ===
    const int n = 1000;
    const int bytes = n * sizeof(float);
    
    printf("向量大小: %d 个元素\n", n);
    printf("内存大小: %d 字节\n\n", bytes);
    
    // === 2. 分配Host内存 ===
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    // === 3. 初始化数据 ===
    printf("初始化数据...\n");
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    printf("✓ 数据初始化完成\n\n");
    
    // === 4. 分配Device内存 ===
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    printf("✓ GPU内存分配完成\n");
    
    // === 5. 将数据拷贝到GPU ===
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    printf("✓ 数据拷贝到GPU完成\n\n");
    
    // === 6. 配置kernel启动参数 ===
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("Kernel配置:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Blocks: %d\n", blocks);
    printf("  Total threads: %d\n\n", blocks * threads_per_block);
    
    // === 7. 启动kernel ===
    printf("启动GPU计算...\n");
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_KERNEL();
    printf("✓ GPU计算完成\n\n");
    
    // === 8. 将结果拷贝回CPU ===
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("✓ 结果拷贝回CPU完成\n\n");
    
    // === 9. 验证结果 ===
    printf("验证结果...\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("Error at index %d: %f + %f = %f (expected %f)\n",
                   i, h_a[i], h_b[i], h_c[i], expected);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ 结果验证通过！\n");
    } else {
        printf("✗ 结果验证失败！\n");
    }
    
    // 显示一些结果
    printf("\n前10个结果:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // === 10. 释放内存 ===
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    printf("\n=== 程序结束 ===\n");
    return 0;
}

/*
 * 关键概念:
 * 
 * 1. 为什么需要边界检查?
 *    例如: n=1000, threads_per_block=256
 *    blocks = (1000 + 255) / 256 = 4
 *    总threads = 4 * 256 = 1024
 *    
 *    Block 0: 处理 idx 0-255    ✓
 *    Block 1: 处理 idx 256-511  ✓
 *    Block 2: 处理 idx 512-767  ✓
 *    Block 3: 处理 idx 768-1023 ⚠️
 *             其中 1000-1023 越界!
 *    
 *    没有 if (idx < n) → 访问越界 → 崩溃
 * 
 * 2. blocks数量计算:
 *    blocks = (n + threads_per_block - 1) / threads_per_block
 *    这是"向上取整"的技巧
 *    
 *    例如: n=1000, threads=256
 *    blocks = (1000 + 256 - 1) / 256
 *          = 1255 / 256
 *          = 4 (整数除法)
 * 
 * 3. 完整的GPU计算流程:
 *    ① 分配Host内存
 *    ② 初始化数据
 *    ③ 分配Device内存
 *    ④ Host → Device 拷贝
 *    ⑤ 启动Kernel
 *    ⑥ Device → Host 拷贝
 *    ⑦ 验证结果
 *    ⑧ 释放所有内存
 * 
 * 练习:
 * 1. 修改向量长度为10000，观察blocks数量变化
 * 2. 修改threads_per_block为128，看需要多少blocks
 * 3. 故意去掉边界检查，看会发生什么
 */
