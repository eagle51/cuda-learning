/*
 * 程序：内存操作基础
 * 目的：理解Host和Device内存
 * 知识点：cudaMalloc, cudaMemcpy, cudaFree
 */

#include <stdio.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {

    // 设置控制台输出为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== CUDA内存操作演示 ===\n\n");

    int n = 5;
    size_t bytes = n * sizeof(float);

    // ========== 步骤1：分配CPU内存 ==========
    printf("步骤1: 分配CPU内存 (%zu bytes)\n", bytes);
    float* h_data = (float*)malloc(bytes);

    // ========== 步骤2：初始化数据 ==========
    printf("步骤2: 初始化CPU数据\n");
    printf("CPU数据: ");
    for (int i = 0; i < n; i++) {
        h_data[i] = i * 1.5f;
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");

    // ========== 步骤3：分配GPU内存 ==========
    printf("步骤3: 分配GPU内存\n");
    float* d_data;
    cudaMalloc(&d_data, bytes);
    printf("GPU内存地址: %p\n\n", d_data);

    // ========== 步骤4：CPU → GPU ==========
    printf("步骤4: 拷贝数据 CPU → GPU\n");
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    printf("拷贝完成\n\n");

    // ========== 步骤5：验证GPU独立性 ==========
    printf("步骤5: 修改CPU数据（测试GPU独立）\n");
    for (int i = 0; i < n; i++) {
        h_data[i] = 0;
    }
    printf("CPU数据清零: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n\n");

    // ========== 步骤6：GPU → CPU ==========
    printf("步骤6: 从GPU拷贝回来\n");
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("从GPU恢复的数据: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("\n");
    printf("✓ 数据完整，GPU内存独立！\n\n");

    // ========== 步骤7：清理内存 ==========
    printf("步骤7: 清理内存\n");
    cudaFree(d_data);
    free(h_data);
    printf("清理完成\n");

    return 0;
}