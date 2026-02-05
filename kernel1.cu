#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

__global__ void kernel (int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}



void matmul(cudaStream_t stream, int M, int N, int K, float alpha, float *A,  float *B, float beta, float *C) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel<<<blocks, threads, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

#include "gemm_profile.cuh"





