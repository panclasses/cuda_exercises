#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel function to print "Hello World" from the GPU
__global__ void helloFromGPU()
{
    printf("Hello World from GPU! thread %d in block %d of size %d\n", threadIdx.x, blockIdx.x, blockDim.x);
}

// Function to launch the kernel
void launchKernel(int const & blocks, int const & threads_per_block)
{
    helloFromGPU<<<blocks, threads_per_block>>>();
    cudaDeviceSynchronize();
}