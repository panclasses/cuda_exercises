#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "CLI11.hpp"
#include <chrono>
#include <ctime>


// Declaration of the function defined in dotProduct.cu
void dotProduct(const float *a, const float *b, float *c, int N, int const & threads_per_block);
void dotProduct2(const float *a, const float *b, float *c, int N, int const & threads_per_block);

int main(int argc, char* argv[]) {
    CLI::App app{"Cuda Hello World"};

    int threads_per_block = 256;
    int N = 100000; // Size of the vectors
    app.add_option("-t,--threads-per-block", threads_per_block, "Number of threads per block");
    app.add_option("-n,--count", N, "element count");

    CLI11_PARSE(app, argc, argv);

    size_t size = N * sizeof(float);

    // Allocate host memory
    std::vector<float> h_a(N, 1.0f); // Initialize with 1.0
    std::vector<float> h_b(N, 2.0f); // Initialize with 2.0
    float h_c = 0.0f;

    // Call the function that launches the kernel
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, sizeof(float));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << " allocate took " << duration.count() << " us." << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);

    // Call the dotProduct function
    dotProduct(d_a, d_b, d_c, N, threads_per_block  );

    // Copy the result back to the host
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << " dot product took " << duration.count() << " us." << std::endl;

    std::cout << "Dot Product: " << h_c << std::endl;

    start_time = std::chrono::high_resolution_clock::now();

    h_c = 0.0f;

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);

    // Call the dotProduct function
    dotProduct2(d_a, d_b, d_c, N, threads_per_block);

    // Copy the result back to the host
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << " dot product 2 took " << duration.count() << " us." << std::endl;

    std::cout << "Dot Product 2: " << h_c << std::endl;


    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
