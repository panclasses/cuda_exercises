#!/bin/bash

#SBATCH --job-name=my_job              # Job name
#SBATCH --output=out_%j.log            # Output log file name (%j expands to jobId)
#SBATCH --error=err_%j.log             # Error log file name
#SBATCH --nodes=2                      # Request 2 nodes
#SBATCH --ntasks=16                    # Total number of tasks (24 tasks/node)
#SBATCH --cpus-per-task=2              # Number of CPU cores per task
#SBATCH --time=02:00:00                # Wall time (2 hours)
#SBATCH --partition=overflow           # Specify partition or queue   -overflow has GPU
#SBATCH --mail-type=ALL                # Send email on start, end, and abort
#SBATCH --mail-user=your_email@example.com # Where to send email notifications

# Load necessary modules (depends on your system setup)
# module load mpi/openmpi-4.0
scl enable devtoolset-10 bash

# Run your program
srun ./build/helloworld
