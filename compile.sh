#!/bin/bash

# Set the NVCC compiler
NVCC=nvcc

# Set the C++ standard version
CPP_STANDARD=c++11

# Specify your CUDA and C++ source files
# CUDA_SOURCE="kernel.cu"
# CPP_SOURCE="main.cpp"

# Name of the output executable
# OUTPUT="my_cuda_app"

# Compile and link the sources
$NVCC -std=$CPP_STANDARD -o helloworld helloworld.cpp hellokernel.cu

$NVCC -std=$CPP_STANDARD -o dotproduct dotproduct.cpp dotproduct.cu

$NVCC -std=$CPP_STANDARD -o prefixsum prefixsum.cpp prefixsum.cu


# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Run with ./$OUTPUT"
else
    echo "Compilation failed."
fi
