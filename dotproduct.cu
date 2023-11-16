#include <cuda_runtime.h>

__global__ void dotProductKernel(const float *a, const float *b, float *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        atomicAdd(c, a[index] * b[index]);
    }
}

void dotProduct(const float *a, const float *b, float *c, int N, int const & threads_per_block) {
    int threadsPerBlock = threads_per_block;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);
    cudaDeviceSynchronize();

}


__global__ void dotProductKernel2(const float *a, const float *b, float *c, int N) {
    extern __shared__ float temp[];  // block shared memory
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // int warp_tid = threadIdx.x % 32;

    // each block sum to shared storage (warp sized)
    // init storage


    // summing to warp wide temp array. 
    // do we need to synchronize during the addtion?  will there be race condition?


    // now reduce across blocks - atomic add?  add and syncthread?
}

void dotProduct2(const float *a, const float *b, float *c, int N, int const & threads_per_block) {
    int threadsPerBlock = threads_per_block;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dotProductKernel<<<blocksPerGrid, threadsPerBlock, 32>>>(a, b, c, N);
    cudaDeviceSynchronize();

}

