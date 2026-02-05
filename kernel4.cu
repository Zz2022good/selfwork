#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <assert.h>

# define CEIL_DIV(M,N) (((M) + (N)-1)/(N))

template <const int BM, const int BN, const int BK,const int TM>
__global__ void kernel4(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){


const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;  // flip x and y to get 30% performance


const uint threadRow = threadIdx.x / BN;
const uint threadCol = threadIdx.x % BN;

__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

A += cRow * BM *K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;

assert(BM*BK == blockDim.x);
assert(BN * BK == blockDim.x);
const uint innerColA = threadIdx.x % BK;
const uint innerRowA = threadIdx.x / BK;
const uint innerColB = threadIdx.x % BN; 
const uint innerRowB = threadIdx.x / BN;

float threadResults[TM] = {0.0};

for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  
  A += BK;
  B += BK * N;

  
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    float Btmp = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}

for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
}
}


void matmul(cudaStream_t stream, int M, int N, int K, float alpha, float *A,  float *B, float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
    dim3 threads((BM * BN) / TM);
    dim3 blocks((N + (BN-1))/BN, (M + (BM -1))/BM);
    kernel4<BM,BN,BK,TM><<<blocks, threads, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
}

#include "gemm_profile.cuh"

