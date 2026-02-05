#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <helper_string.h>

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void matmul(cudaStream_t stream, int M, int N, int K,
            float alpha, float *A, float *B, float beta, float *C) {
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasSetStream(handle, stream);
    runCublasTF32(handle, M, N, K, alpha, A, B, beta, C);
}


#include "gemm_profile.cuh"


// 1.25 TFLOP/s
    

