/**
 * 07_error_handling.cu
 * CUDA错误处理示例
 *
 * 编译: nvcc 07_error_handling.cu -o 07_error_handling
 * 运行: 07_error_handling.exe
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <device_launch_parameters.h>
#ifdef _WIN32
#include <windows.h>
#endif

__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2.0f;
    }
}

int main() {
    // 设置控制台输出为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    printf("=== CUDA错误处理示例 ===\n\n");

    // === 1. 打印GPU信息 ===
    print_gpu_info();

    // === 2. 测试正常的API调用 ===
    printf("测试1: 正常的内存分配 \n");
    float *d_data;
    int size = 1024;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    printf("  [OK] 内存分配成功\n\n");

    // === 3. 测试Kernel ===
    printf("测试2: Kernel执行\n");
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    printf("  配置: %d blocks x %d threads\n", blocks, threads);

    test_kernel<<<blocks, threads>>>(d_data, size);
    CUDA_CHECK_KERNEL();
    printf("  [OK] Kernel执行成功\n\n");

    // === 4. 测试内存拷贝 ===
    printf("测试3: 内存拷贝\n");
    float *h_data = new float[size];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("  [OK] 内存拷贝成功\n\n");

    // 验证结果
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (h_data[i] != i * 2.0f) {
            correct = false;
            break;
        }
    }
    printf("  [OK] 结果验证: %s\n\n", correct ? "通过" : "失败");

    // === 5. 清理 ===
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));

    printf("=== 所有测试通过! ===\n\n");

    // === 6. 演示错误检查的重要性 ===
    printf("提示: 如果没有错误检查会怎样?\n");
    printf("  1. 错误会被忽略\n");
    printf("  2. 程序可能崩溃\n");
    printf("  3. 难以定位问题\n");
    printf("  4. 调试困难\n\n");

    printf("使用CUDA_CHECK的好处:\n");
    printf("  + 自动检查所有调用 \n");
    printf("  + 详细的错误信息 \n");
    printf("  + 快速定位问题 \n");
    printf("  + 提高代码质量\n\n");
    // === 7. 可选：故意制造错误（取消注释测试）===
    
    printf("测试4: 故意制造错误 \n");
    printf("(取消下面的注释来测试错误检查)\n\n");

    // 错误1: 分配负数大小的内存
    //float *bad_ptr;
    //CUDA_CHECK(cudaMalloc(&bad_ptr, -1));

    // 错误2: 拷贝过大的数据
    //float ss[10];
    //CUDA_CHECK(cudaMemcpy(ss, d_data, size * sizeof(float),
    //                      cudaMemcpyDeviceToHost));

    // 错误3: 启动过多的threads
    test_kernel<<<1, 2048>>>(d_data, size);  // 超过1024限制
    CUDA_CHECK_KERNEL();
    
    

    printf("=== 程序结束 ===\n");
    return 0;
}

/*
 * 错误处理最佳实践:
 *
 * 1. 总是检查CUDA API调用:
 *    ❌ cudaMalloc(&ptr, size);
 *    ✅ CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * 2. 总是检查Kernel执行:
 *    ❌ kernel<<<blocks, threads>>>(args);
 *    ✅ kernel<<<blocks, threads>>>(args);
 *       CUDA_CHECK_KERNEL();
 *
 * 3. 开发时使用错误检查，发布时也保留
 *    - 性能影响很小
 *    - 安全性大幅提升
 *
 * 4. 使用工具头文件（cuda_utils.h）
 *    - 统一错误处理
 *    - 减少重复代码
 *    - 便于维护
 *
 * 常见错误类型:
 *
 * 1. cudaErrorMemoryAllocation
 *    - 原因: GPU内存不足
 *    - 解决: 减少数据量或释放不用的内存
 *
 * 2. cudaErrorInvalidValue
 *    - 原因: 参数错误（如负数大小）
 *    - 解决: 检查参数有效性
 *
 * 3. cudaErrorInvalidConfiguration
 *    - 原因: Kernel配置错误（如threads > 1024）
 *    - 解决: 检查blocks和threads配置
 *
 * 4. cudaErrorLaunchOutOfResources
 *    - 原因: 资源不足（如shared memory）
 *    - 解决: 减少资源使用
 */
