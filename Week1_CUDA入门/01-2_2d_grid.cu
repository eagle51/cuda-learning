#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#ifdef _WIN32
#include <windows.h>
#endif

// 2D Grid示例：处理二维矩阵
__global__ void matrix_kernel() {
    // 2D block内的线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 2D grid内的block索引
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算全局2D坐标
    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    printf("Thread (%d,%d) in Block (%d,%d) -> Global position (%d,%d)\n",
           tx, ty, bx, by, col, row);
}

int main() {
    // 设置控制台输出为 UTF-8
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("启动2D Grid kernel...\n\n");

    // 定义2D grid: 2×2 = 4个blocks
    dim3 grid(2, 2);

    // 定义2D block: 每个block 3×3 = 9个threads
    dim3 block(3, 3);

    printf("Grid配置: %d × %d blocks\n", grid.x, grid.y);
    printf("Block配置: %d × %d threads per block\n", block.x, block.y);
    printf("总线程数: %d × %d = %d threads\n\n",
           grid.x * block.x, grid.y * block.y,
           grid.x * block.x * grid.y * block.y);

    // 启动kernel
    matrix_kernel<<<grid, block>>>();
    CUDA_CHECK_KERNEL();

    printf("\nGPU执行完成\n");

    return 0;
}
