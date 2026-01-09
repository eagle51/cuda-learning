/**
 * 04_vector_add_cpu.cpp
 * CPU版向量加法 - 性能基准测试
 * 
 * 编译: g++ 04_vector_add_cpu.cpp -o 04_vector_add_cpu -O3
 * 或者: cl 04_vector_add_cpu.cpp /O2 /Fe:04_vector_add_cpu.exe
 * 运行: 04_vector_add_cpu.exe
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif
using namespace std;
using namespace std::chrono;

/**
 * CPU版向量加法
 */
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * 测试指定大小的向量加法性能
 */
void test_size(int n) {
    // 分配内存
    vector<float> a(n), b(n), c(n);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }
    
    // 计时
    auto start = high_resolution_clock::now();
    vector_add_cpu(a.data(), b.data(), c.data(), n);
    auto end = high_resolution_clock::now();
    
    // 计算时间(毫秒)
    auto duration = duration_cast<microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < min(n, 10); i++) {
        if (abs(c[i] - (a[i] + b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    // 计算吞吐量 (GB/s)
    double data_gb = (n * 3 * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double throughput = data_gb / (time_ms / 1000.0);
    
    printf("数据量: %10d, CPU时间: %8.3f ms, 吞吐量: %6.2f GB/s, 结果: %s\n",
           n, time_ms, throughput, correct ? "正确" : "错误");
}

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
	printf("=== 向量加法 (GPU) ===\n\n");
    printf("=== CPU向量加法性能测试 ===\n\n");
    
    // 测试不同数据量
    vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    printf("%-12s %-15s %-15s %s\n", "数据量", "CPU时间(ms)", "吞吐量(GB/s)", "结果");
    printf("─────────────────────────────────────────────────────\n");
    
    for (int n : sizes) {
        test_size(n);
    }
    
    printf("\n=== 测试完成 ===\n");
    printf("\n提示: 将这些数据与GPU版本对比！\n");
    
    return 0;
}

/*
 * 关键点:
 * 
 * 1. 使用 std::chrono 进行高精度计时
 * 2. 测试多个数据量，观察性能趋势
 * 3. 计算吞吐量 = 数据量 / 时间
 * 4. 记录这些数据，明天与GPU对比
 * 
 * 预期现象:
 * - 小数据量: 时间很短 (< 1ms)
 * - 大数据量: 时间线性增长
 * - CPU是顺序执行，一个一个加
 */
