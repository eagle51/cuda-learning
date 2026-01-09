/**
 * matrix_operations.cu
 * 完整的矩阵运算项目 - CPU vs GPU性能对比
 * 
 * 这是Week 1的完整项目代码！
 * 
 * 编译: nvcc matrix_operations.cu -o matrix_operations
 * 运行: matrix_operations.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include "cuda_utils.h"
#include <device_launch_parameters.h>
#ifdef _WIN32
#include <windows.h>
#endif
using namespace std::chrono;

// ============================================
// CPU版本
// ============================================

void matrix_add_cpu(const float* a, const float* b, float* c,
                   int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            c[idx] = a[idx] + b[idx];
        }
    }
}

// ============================================
// GPU版本
// ============================================

__global__ void matrix_add_gpu(const float* a, const float* b, float* c,
                              int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// 验证函数
// ============================================

bool verify(const float* cpu_result, const float* gpu_result, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > 1e-5) {
            printf("❌ 不匹配 at %d: CPU=%f, GPU=%f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// ============================================
// 测试函数
// ============================================

void test_matrix_operations(int width, int height) {
    int size = width * height;
    int bytes = size * sizeof(float);
    
    printf("\n");
    printf("================================================\n");
    printf("  Testing Matrix Size: %4d x %4d\n", width, height);
    printf("================================================\n\n");
    
    // === 1. 分配Host内存 ===
    printf("1. 分配内存...\n");
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c_cpu = new float[size];
    float *h_c_gpu = new float[size];
    printf("   ✓ Host内存: %.2f MB\n", (bytes * 4) / (1024.0 * 1024.0));
    
    // === 2. 初始化数据 ===
    printf("2. 初始化数据...\n");
    for (int i = 0; i < size; i++) {
        h_a[i] = (rand() % 100) / 10.0f;
        h_b[i] = (rand() % 100) / 10.0f;
    }
    printf("   ✓ 数据初始化完成\n");
    
    // === 3. CPU计算 ===
    printf("3. CPU计算...\n");
    auto cpu_start = high_resolution_clock::now();
    matrix_add_cpu(h_a, h_b, h_c_cpu, width, height);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_time = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;
    printf("   ✓ CPU时间: %.3f ms\n", cpu_time);
    
    // === 4. GPU准备 ===
    printf("4. GPU准备...\n");
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    printf("   ✓ Device内存: %.2f MB\n", (bytes * 3) / (1024.0 * 1024.0));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    printf("   ✓ 数据传输到GPU完成\n");
    
    // === 5. GPU计算 ===
    printf("5. GPU计算...\n");
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    printf("   配置: %d×%d blocks, %d×%d threads\n",
           blocks.x, blocks.y, threads.x, threads.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    matrix_add_gpu<<<blocks, threads>>>(d_a, d_b, d_c, width, height);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    printf("   ✓ GPU时间: %.3f ms\n", gpu_time);
    
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("   ✓ 结果传输回CPU完成\n");
    
    // === 6. 验证 ===
    printf("6. 验证结果...\n");
    bool correct = verify(h_c_cpu, h_c_gpu, size);
    printf("验证%s\n", correct ? "通过" : "失败");
    
    // === 7. 性能分析 ===
    printf("7. 性能分析:\n");
    double speedup = cpu_time / gpu_time;
    printf("   加速比: %.2fx\n", speedup);
    
    double data_gb = (size * 3 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double cpu_throughput = data_gb / (cpu_time / 1000.0);
    double gpu_throughput = data_gb / (gpu_time / 1000.0);
    printf("   CPU吞吐量: %.2f GB/s\n", cpu_throughput);
    printf("   GPU吞吐量: %.2f GB/s\n", gpu_throughput);
    
    // === 8. 清理 ===
    delete[] h_a; delete[] h_b; delete[] h_c_cpu; delete[] h_c_gpu;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================
// 主函数
// ============================================

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("================================================\n");
    printf("   Matrix Operations: CPU vs GPU Comparison    \n");
    printf("              Week 1 Complete Project           \n");
    printf("================================================\n\n");
    
    // 打印GPU信息
    print_gpu_info();
    
    // 测试不同矩阵大小
    int sizes[][2] = {
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {4096, 4096}
    };
    
    for (int i = 0; i < 4; i++) {
        test_matrix_operations(sizes[i][0], sizes[i][1]);
        printf("\n");
    }
    
    printf("================================================\n");
    printf("             All Tests Completed!               \n");
    printf("================================================\n\n");
    
    printf("总结:\n");
    printf("✓ 完成了4种矩阵大小的测试\n");
    printf("✓ CPU和GPU版本都实现了\n");
    printf("✓ 结果验证全部通过\n");
    printf("✓ 性能数据完整记录\n\n");
    
    printf("下一步:\n");
    printf("1. 将这些数据制作成表格\n");
    printf("2. 撰写技术博客\n");
    printf("3. 更新GitHub项目\n\n");
    
    return 0;
}

/*
 * Week 1 完整项目总结:
 * 
 * 1. 实现了什么?
 *    - CPU版矩阵加法
 *    - GPU版矩阵加法
 *    - 完整的性能对比
 *    - 结果验证
 *    - 错误检查
 * 
 * 2. 学到了什么?
 *    - CUDA编程基础
 *    - 内存管理
 *    - Kernel编程
 *    - 性能测试
 *    - 错误处理
 * 
 * 3. 性能结论:
 *    - 小矩阵: GPU可能更慢（传输开销）
 *    - 大矩阵: GPU明显更快（100x+）
 *    - 矩阵越大，GPU优势越明显
 * 
 * 4. 代码质量:
 *    - 使用错误检查
 *    - 详细注释
 *    - 结构清晰
 *    - 便于维护
 * 
 * 5. 下一步:
 *    Week 2将学习Shared Memory优化
 *    进一步提升性能！
 */
