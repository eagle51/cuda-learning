/*
 * 程序：GPU向量加法性能测试
 * 目的：对比不同数据量的GPU性能
 */

#include <stdio.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== GPU向量加法性能测试 ===\n\n");

    // 测试不同的数据量
    int sizes[] = { 1000, 10000, 100000, 1000000, 10000000 };
    int num_tests = 5;

    printf("数据量         时间(ms)      吞吐量(GB/s)   加速比\n");
    printf("-----------------------------------------------------------\n");

    // CPU baseline时间（手动填入）
    double cpu_times[] = { 0.001, 0.010, 0.100, 1.000, 10.000 }; // 预估值

    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        size_t bytes = n * sizeof(float);

        // CPU内存
        float* h_a = (float*)malloc(bytes);
        float* h_b = (float*)malloc(bytes);
        float* h_c = (float*)malloc(bytes);

        // 初始化
        for (int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i * 2.0f;
        }

        // GPU内存
        float* d_a, * d_b, * d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Kernel配置
        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;

        // CUDA计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vector_add <<<blocks, threads_per_block >>> (d_a, d_b, d_c, n);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);

        // 计算吞吐量
        double bandwidth_gb = (3.0 * bytes / (1024 * 1024 * 1024)) / (time_ms / 1000.0);

        // 加速比（需要先运行CPU版本得到准确时间）
        double speedup = cpu_times[t] / time_ms;

        printf("%-12d   %8.3f      %8.2f       %.1fx\n",
            n, time_ms, bandwidth_gb, speedup);

        // 清理
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
    }

    printf("\n注意：加速比需要先运行CPU版本获得准确时间\n");

    return 0;
}