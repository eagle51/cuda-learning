/**
 * 06_matrix_add.cu
 * 矩阵加法 - 2D Grid和2D Thread
 * 
 * 编译: nvcc 06_matrix_add.cu -o 06_matrix_add
 * 运行: 06_matrix_add.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef _WIN32
#include <windows.h>
#endif
/**
 * Kernel: 矩阵加法
 * @param a 输入矩阵a
 * @param b 输入矩阵b
 * @param c 输出矩阵c (c = a + b)
 * @param width 矩阵宽度
 * @param height 矩阵高度
 */
__global__ void matrix_add(const float* a, const float* b, float* c,
                          int width, int height) {
    // 计算2D位置
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查（重要！）
    if (col < width && row < height) {
        // 2D位置转换为1D索引
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
	// 设置控制台输出为 UTF-8
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== 矩阵加法 (2D Grid) ===\n\n");
    
    // === 1. 矩阵大小 ===
    const int width = 16384;
    const int height = 16384;
    const int size = width * height;
    const int bytes = size * sizeof(float);
    
    printf("矩阵大小: %d × %d\n", width, height);
    printf("元素数量: %d\n", size);
    printf("内存大小: %.2f MB\n\n", bytes / (1024.0 * 1024.0));
    
    // === 2. 分配Host内存 ===
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c = new float[size];
    
    // === 3. 初始化数据 ===
    printf("初始化数据...\n");
    for (int i = 0; i < size; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    printf("✓ 数据初始化完成\n\n");
    
    // === 4. 分配Device内存 ===
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    printf("✓ GPU内存分配完成\n");
    
    // === 5. 拷贝数据到GPU ===
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    printf("✓ 数据拷贝到GPU完成\n\n");
    
    // === 6. 配置2D Grid ===
    dim3 threads_per_block(8, 8);  // 16×16 = 256 threads
    dim3 num_blocks(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );
    
    printf("Kernel配置:\n");
    printf("  Threads per block: %d × %d = %d\n", 
           threads_per_block.x, threads_per_block.y,
           threads_per_block.x * threads_per_block.y);
    printf("  Blocks: %d × %d = %d\n",
           num_blocks.x, num_blocks.y,
           num_blocks.x * num_blocks.y);
    printf("  Total threads: %d\n\n",
           num_blocks.x * num_blocks.y * threads_per_block.x * threads_per_block.y);
    
    // === 7. 创建Event计时 ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // === 8. 启动kernel ===
    printf("启动GPU计算...\n");
    cudaEventRecord(start);
    matrix_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("✓ GPU计算完成\n");
    printf("  执行时间: %.3f ms\n\n", milliseconds);
    
    // === 9. 拷贝结果回CPU ===
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("✓ 结果拷贝回CPU完成\n\n");
    
    // === 10. 验证结果 ===
    printf("验证结果...\n");
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < size && errors < 5; i++) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 1e-5) {
            int row = i / width;
            int col = i % width;
            printf("Error at (%d, %d): %f + %f = %f (expected %f)\n",
                   row, col, h_a[i], h_b[i], h_c[i], expected);
            correct = false;
            errors++;
        }
    }
    
    if (correct) {
        printf("✓ 结果验证通过！\n");
    } else {
        printf("✗ 发现 %d 个错误\n", errors);
    }
    
    // 显示一些结果
    printf("\n左上角 3×3 的结果:\n");
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int idx = row * width + col;
            printf("%.1f ", h_c[idx]);
        }
        printf("\n");
    }
    
    // === 11. 性能计算 ===
    double data_gb = (size * 3 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double throughput = data_gb / (milliseconds / 1000.0);
    
    printf("\n性能数据:\n");
    printf("  数据量: %.2f GB\n", data_gb);
    printf("  吞吐量: %.2f GB/s\n", throughput);
    
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("每个Block最大线程数: %d\n", prop.maxThreadsPerBlock);
	printf("每个SM最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("SM数量: %d\n", prop.multiProcessorCount);

    // === 12. 释放内存 ===
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n=== 程序结束 ===\n");
    return 0;
}

/*
 * 2D索引关键概念:
 * 
 * 1. 2D索引计算:
 *    col = blockIdx.x * blockDim.x + threadIdx.x
 *    row = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * 2. 2D转1D索引:
 *    idx = row * width + col
 *    
 *    例如 (row=2, col=3, width=10):
 *    idx = 2 * 10 + 3 = 23
 *    
 *    可视化:
 *    Row 0: [0][1][2]...[9]
 *    Row 1: [10][11][12]...[19]
 *    Row 2: [20][21][22][23]...[29]  ← 这里
 * 
 * 3. dim3类型:
 *    dim3 threads(16, 16)   → threads.x=16, threads.y=16, threads.z=1
 *    dim3 blocks(64, 64)    → blocks.x=64, blocks.y=64, blocks.z=1
 * 
 * 4. 为什么用16×16?
 *    - 16×16 = 256 threads (常用配置)
 *    - 2的幂次，硬件友好
 *    - 适合矩阵/图像处理
 * 
 * 5. 边界检查的必要性:
 *    矩阵大小: 1024×1024
 *    Block大小: 16×16
 *    需要blocks: 64×64
 *    
 *    最后的block可能处理超出矩阵范围的位置
 *    必须用 if (col < width && row < height) 检查
 * 
 * 练习:
 * 1. 修改矩阵大小为512×512，2048×2048
 * 2. 尝试不同的block大小: 8×8, 32×32
 * 3. 观察性能变化
 * 4. 记录数据，为明天做准备
 */
