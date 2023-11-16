#include <iostream>
#include "CLI11.hpp"
#include <chrono>
#include <ctime>


// Declaration of the function to launch the kernel
void launchKernel(int const & blocks, int const & threads_per_block);

int main(int argc, char* argv[]) {
    CLI::App app{"Cuda Hello World"};

    int blocks = 1, threads_per_block = 10;
    app.add_option("-b,--blocks", blocks, "Number of blocks");
    app.add_option("-t,--threads-per-block", threads_per_block, "Number of threads per block");

    CLI11_PARSE(app, argc, argv);




    // Print from the host (CPU)
    std::cout << "Hello World from CPU!\n";

    // Call the function that launches the kernel
    auto start_time = std::chrono::high_resolution_clock::now();
    
    launchKernel(blocks, threads_per_block);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << " hello world took " << duration.count() << " us." << std::endl;

    return 0;
}
