/*
 * 程序：向量加法
 * 目的：第一个真正的GPU计算程序
 * 知识点：kernel函数、索引计算、边界检查
 */

#include <stdio.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

 // GPU Kernel: 向量加法
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // 每个thread计算自己负责的元素
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查（重要！）
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 设置控制台输出为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== GPU向量加法 ===\n\n");

    // ========== 参数设置 ==========
    int n = 10000;  // 向量长度
    size_t bytes = n * sizeof(float);

    printf("向量长度: %d\n", n);
    printf("内存大小: %zu bytes\n\n", bytes);

    // ========== 1. 分配CPU内存 ==========
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // ========== 2. 初始化数据 ==========
    printf("初始化数据...\n");
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }

    // 显示前5个元素
    printf("前5个元素:\n");
    for (int i = 0; i < 5; i++) {
        printf("  a[%d]=%.1f, b[%d]=%.1f\n", i, h_a[i], i, h_b[i]);
    }
    printf("\n");

    // ========== 3. 分配GPU内存 ==========
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ========== 4. 数据拷贝到GPU ==========
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // ========== 5. 配置并启动kernel ==========
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    printf("Kernel配置:\n");
    printf("  Blocks: %d\n", blocks);
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  总threads: %d (实际需要: %d)\n\n", blocks * threads_per_block, n);

    printf("启动GPU计算...\n");
    vector_add << <blocks, threads_per_block >> > (d_a, d_b, d_c, n);

    // ========== 6. 等待GPU完成 ==========
    cudaDeviceSynchronize();
    printf("GPU计算完成！\n\n");

    // ========== 7. 结果拷贝回CPU ==========
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // ========== 8. 验证结果 ==========
    printf("验证结果（前10个）:\n");
    bool all_correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        bool correct = (h_c[i] == expected);

        printf("  [%d] %.1f + %.1f = %.1f (期望: %.1f) %s\n",
            i, h_a[i], h_b[i], h_c[i], expected,
            correct ? "[OK]" : "[FAIL]");

        if (!correct) all_correct = false;
    }

    // 验证所有元素
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            all_correct = false;
            break;
        }
    }

    if (all_correct) {
        printf("\n[OK] 所有%d个结果都正确！\n", n);
    }
    else {
        printf("\n[FAIL] 发现错误结果！\n");
    }

    // ========== 9. 清理内存 ==========
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}