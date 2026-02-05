#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

# define CEIL_DIV(M,N) (((M) + (N)-1)/(N))

constexpr int BLOCK_SIZE = 32;

// sgemm_shared_memory_block
__global__ void kernel3 (int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
const uint threadRow = threadIdx.x / BLOCK_SIZE;
const uint threadCol = threadIdx.x % BLOCK_SIZE;

const uint cRow = blockIdx.x;
const uint cCol = blockIdx.y;

__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
__shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

A += cRow * BLOCK_SIZE * K;            
B += cCol * BLOCK_SIZE;                
C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

float tmp = 0.0;

for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE){

    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];

}

__syncthreads();

}

C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}


void matmul(cudaStream_t stream, int M, int N, int K, float alpha, float *A,  float *B, float beta, float *C) {
    dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocks((M + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel3<<<blocks, threads, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}


#include "gemm_profile.cuh"

