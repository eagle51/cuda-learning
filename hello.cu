#include <iostream>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", idx);
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;

    helloKernel<<<2, 4>>>();

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "CUDA execution completed!" << std::endl;

    return 0;
}
