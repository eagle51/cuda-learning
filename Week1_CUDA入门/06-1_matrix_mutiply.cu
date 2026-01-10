#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <device_launch_parameters.h>
#ifdef _WIN32
#include <windows.h>
#endif
#define BLOCK_SIZE 16

__global__ void matrix_mul(float* A, float* B, float* C,
	int M, int K, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += A[row * K + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

int main() {
	// 设置控制台输出为 UTF-8
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
	int M = 1024, K = 1024, N = 1024;

	size_t size_A = M * K * sizeof(float);
	size_t size_B = K * N * sizeof(float);
	size_t size_C = M * N * sizeof(float);

	// 分配Host内存并初始化
	float* h_A = (float*)malloc(size_A);
	float* h_B = (float*)malloc(size_B);
	float* h_C = (float*)malloc(size_C);

	for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
	for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

	// 分配Device内存
	float* d_A, * d_B, * d_C;
	CUDA_CHECK(cudaMalloc(&d_A, size_A));
	CUDA_CHECK(cudaMalloc(&d_B, size_B));
	CUDA_CHECK(cudaMalloc(&d_C, size_C));

	// 拷贝数据到GPU
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

	// 配置Grid和Block
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// 执行kernel
	matrix_mul << <gridDim, blockDim >> > (d_A, d_B, d_C, M, K, N);
	CUDA_CHECK_KERNEL();
	// 拷贝结果回CPU
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

	// 清理
	free(h_A); free(h_B); free(h_C);
	CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

	return 0;
}
