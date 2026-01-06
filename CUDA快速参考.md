# CUDA编程快速参考

## 常用函数

### 内存管理
```cpp
// 分配GPU内存
cudaError_t cudaMalloc(void** ptr, size_t size);

// 释放GPU内存
cudaError_t cudaFree(void* ptr);

// 内存拷贝
cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, 
                       cudaMemcpyKind kind);
// kind: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, 
//       cudaMemcpyDeviceToDevice

// 内存设置
cudaError_t cudaMemset(void* ptr, int value, size_t size);
```

### Kernel启动
```cpp
// 1D配置
kernel<<<blocks, threads>>>(args...);

// 2D配置
dim3 threads(16, 16);
dim3 blocks(width/16, height/16);
kernel<<<blocks, threads>>>(args...);

// 3D配置
dim3 threads(8, 8, 8);
dim3 blocks(w/8, h/8, d/8);
kernel<<<blocks, threads>>>(args...);
```

### 同步
```cpp
// 等待GPU完成
cudaError_t cudaDeviceSynchronize();

// Block内同步
__syncthreads();
```

### 事件计时
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel ...
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

## 索引计算

### 1D
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 2D
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;
```

### 3D
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;
```

## Blocks/Threads计算
```cpp
// 1D
int threads = 256;
int blocks = (n + threads - 1) / threads;

// 2D
dim3 threads(16, 16);
dim3 blocks((width + 15) / 16, (height + 15) / 16);
```

## 常见配置

| 场景 | Threads/Block | 说明 |
|------|---------------|------|
| 向量操作 | 256 | 通用推荐 |
| 矩阵操作 | 16×16 | 2D操作 |
| Reduction | 256/512 | 需要Shared Memory |
| 简单操作 | 128/256 | 平衡性能 |

## Shared Memory
```cpp
__global__ void kernel() {
    __shared__ float cache[256];
    
    int tid = threadIdx.x;
    cache[tid] = data[tid];
    
    __syncthreads();  // 同步
    
    // 使用cache...
}
```

## 错误码

| 错误码 | 含义 |
|--------|------|
| cudaSuccess | 成功 |
| cudaErrorMemoryAllocation | 内存分配失败 |
| cudaErrorInvalidValue | 无效参数 |
| cudaErrorInvalidDevicePointer | 无效设备指针 |
| cudaErrorLaunchFailure | Kernel启动失败 |

## 调试技巧

### 1. 打印调试
```cpp
__global__ void kernel() {
    printf("Thread %d\n", threadIdx.x);
}
```

### 2. 检查错误
```cpp
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```

### 3. CUDA-MEMCHECK
```bash
cuda-memcheck ./program
```

## 性能优化Checklist

- [ ] 合理的Block/Thread配置
- [ ] 最小化Host-Device传输
- [ ] 使用Shared Memory
- [ ] 避免Bank Conflict
- [ ] Coalesced内存访问
- [ ] 避免Warp Divergence
- [ ] 充分的Occupancy
- [ ] 向量化加载（float4）