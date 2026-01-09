/**
 * 05_vector_add_gpu_timed.cu
 * GPU版向量加法 - 精确计时和性能测试
 * 
 * 编译: nvcc 05_vector_add_gpu_timed.cu -o 05_vector_add_gpu_timed
 * 运行: 05_vector_add_gpu_timed.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void test_size(int n) {
    int bytes = n * sizeof(float);
    
    // 分配Host内存
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    // 初始化
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // 分配Device内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // 拷贝数据到GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // 配置kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // 创建CUDA Event用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 开始计时
    cudaEventRecord(start);
    
    // 启动kernel
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    // 结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 获取时间
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // 拷贝结果
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // 验证
    bool correct = true;
    for (int i = 0; i < n && i < 10; i++) {
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    // 计算吞吐量
    double data_gb = (n * 3 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double throughput = data_gb / (gpu_time / 1000.0);
    
    printf("数据量: %10d, GPU时间: %8.3f ms, 吞吐量: %6.2f GB/s, 结果: %s\n",
           n, gpu_time, throughput, correct ? "正确" : "错误");
    
    // 清理
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("=== GPU向量加法性能测试 ===\n\n");
    
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    
    printf("%-12s %-15s %-15s %s\n", "数据量", "GPU时间(ms)", "吞吐量(GB/s)", "结果");
    printf("─────────────────────────────────────────────────────\n");
    
    for (int i = 0; i < 5; i++) {
        test_size(sizes[i]);
    }
    
    printf("\n=== 测试完成 ===\n");
    printf("\n现在制作CPU vs GPU对比表:\n");
    printf("把CPU和GPU的数据放在一起，计算加速比！\n");
    
    return 0;
}

/*
 * CUDA Event计时说明:
 * 
 * 1. 为什么用cudaEvent而不是std::chrono?
 *    - cudaEvent在GPU上计时，更精确
 *    - std::chrono在CPU上计时，包含同步开销
 * 
 * 2. 计时步骤:
 *    ① cudaEventCreate - 创建event
 *    ② cudaEventRecord(start) - 记录开始时间
 *    ③ kernel<<<>>>() - 执行kernel
 *    ④ cudaEventRecord(stop) - 记录结束时间
 *    ⑤ cudaEventSynchronize(stop) - 等待完成
 *    ⑥ cudaEventElapsedTime - 获取时间差
 * 
 * 3. 性能对比:
 *    - 小数据量: GPU可能更慢(数据传输开销)
 *    - 大数据量: GPU明显更快
 *    - 记录下你的数据！
 */
