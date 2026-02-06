In this assignemnt, I'm replicating the kernels and calculations from the aticle by Siboehm, and benchmarking their performance on a H100 GPU using Modal.
in every kernel file, I included "gemm_profile.cuh". The harness & benchmarking profile (gemm_profile.cuh) is provided in lecture. 
The outputs are aso on my Modal Notebook. 
1. Kernel1 is a naive kernel, each thread computes one output element. it achieved 250 GFLOP/s and average kernel time is 275751.53 us.
2. kernel2 is global memory colaescing, threads can access consecutive memory in the same warp. A warp is then assigned to a warp scheduler. it achieved 2330 GFLOP/s and avarage kernel time is 29487.81 us.
3. kernel3 is shared memeory blocking,it loads tiles of A and B into shared memoryonce and reuse them. it achieved 4560 GfLOP/s and average kernel time is 15087.81 us.
4. kernel 4 is 1D blocking, it achieved 8440 GFLOP/s and average kernel time is 8146.36 us.
5. kernel 5 is 2D blocking, it extends thread tiling to 2D and threads compute a sub-block of C. it achieved 12700 GFLOP/s and average kernel time is 5411.41 us.
6. kernel 6 is it achieved 14450 GFLOP/s and average kernel time is 4756.39 us.
9. kernel 9 is autotuned, it achieved 15340 GFLOP/s and average kernel time is 4482.06 us.
10. kernel 10 is warptiling, it achieved 14940 GFLOP/s and average kernel time is 4600.81 us. 

# Notice I've changed M, N,K in main() from 1024 to 4096 for measureing. I'm using large matrices, because otherwise we can't see a major difference in term of prformance between kernel 4,5,6,9,10. 

