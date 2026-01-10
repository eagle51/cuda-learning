#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include <device_launch_parameters.h>
#include <chrono>

#define BLOCK_SIZE 16

using namespace std::chrono;

// GPU版矩阵乘法kernel
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

// CPU版矩阵乘法
void matrix_mul_cpu(float* A, float* B, float* C, int M, int K, int N) {
	for (int row = 0; row < M; row++) {
		for (int col = 0; col < N; col++) {
			float sum = 0.0f;
			for (int k = 0; k < K; k++) {
				sum += A[row * K + k] * B[k * N + col];
			}
			C[row * N + col] = sum;
		}
	}
}

void test_matrix_size(int M, int K, int N) {
	size_t size_A = M * K * sizeof(float);
	size_t size_B = K * N * sizeof(float);
	size_t size_C = M * N * sizeof(float);

	// 分配Host内存并初始化
	float* h_A = (float*)malloc(size_A);
	float* h_B = (float*)malloc(size_B);
	float* h_C_gpu = (float*)malloc(size_C);
	float* h_C_cpu = (float*)malloc(size_C);

	for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
	for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

	// ========== CPU版本 ==========
	auto cpu_start = high_resolution_clock::now();
	matrix_mul_cpu(h_A, h_B, h_C_cpu, M, K, N);
	auto cpu_end = high_resolution_clock::now();
	auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
	double cpu_time = cpu_duration.count() / 1000.0; // 转换为毫秒

	// ========== GPU版本 ==========
	// 分配Device内存
	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);

	// 拷贝数据到GPU
	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

	// 配置Grid和Block
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// 创建CUDA Event用于计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// 开始GPU计时
	cudaEventRecord(start);

	// 执行kernel
	matrix_mul << <gridDim, blockDim >> > (d_A, d_B, d_C, M, K, N);

	// 结束GPU计时
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// 获取GPU时间
	float gpu_time;
	cudaEventElapsedTime(&gpu_time, start, stop);

	// 拷贝结果回CPU
	cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

	// 验证结果
	bool correct = true;
	for (int i = 0; i < M * N && i < 100; i++) {
		if (abs(h_C_cpu[i] - h_C_gpu[i]) > 1e-3) {
			correct = false;
			break;
		}
	}

	// 计算加速比
	double speedup = cpu_time / gpu_time;

	// 打印结果
	printf("%-15s %-20.3f %-25.3f %.2fx%-12s %s\n",
		(std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N)).c_str(),
		cpu_time,
		gpu_time,
		speedup, "",
		correct ? "OK" : "FAIL");

	// 清理
	free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
	printf("=== 矩阵乘法性能对比 (CPU vs GPU) ===\n\n");
	printf("%-15s %-20s %-25s %-15s %s\n",
		"矩阵大小", "CPU耗时(ms)", "GPU耗时(基础版)(ms)", "加速比", "验证");
	printf("────────────────────────────────────────────────────────────────────────────────\n");

	// 测试不同大小的矩阵
	test_matrix_size(128, 128, 128);
	test_matrix_size(256, 256, 256);
	test_matrix_size(512, 512, 512);
	test_matrix_size(1024, 1024, 1024);
	test_matrix_size(2048, 2048, 2048);

	printf("\n=== 测试完成 ===\n");

	return 0;
}
