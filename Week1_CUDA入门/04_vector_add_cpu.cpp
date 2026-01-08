/*
 * 程序：CPU向量加法（用于性能对比）
 * 目的：对比CPU和GPU性能
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== CPU向量加法性能测试 ===\n\n");
    
    // 测试不同的数据量
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int num_tests = 5;
    
    printf("数据量         时间(ms)      吞吐量(GB/s)\n");
    printf("----------------------------------------------\n");
    
    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        size_t bytes = n * sizeof(float);
        
        // 分配内存
        float *a = (float*)malloc(bytes);
        float *b = (float*)malloc(bytes);
        float *c = (float*)malloc(bytes);
        
        // 初始化
        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = i * 2.0f;
        }
        
        // 计时
        clock_t start = clock();
        vector_add_cpu(a, b, c, n);
        clock_t end = clock();
        
        double time_ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
        
        // 计算吞吐量：读a + 读b + 写c = 3 * n * sizeof(float)
        double bandwidth_gb = (3.0 * bytes / (1024*1024*1024)) / (time_ms / 1000.0);
        
        printf("%-12d   %8.3f      %8.2f\n", n, time_ms, bandwidth_gb);
        
        free(a);
        free(b);
        free(c);
    }
    
    return 0;
}