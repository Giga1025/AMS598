#!/bin/bash
#
#SBATCH --job-name=twins_climate
#SBATCH --output=twins_code.log
#SBATCH --error=twins_code.err
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=01:00:00       # 1 hour (adjust based on dataset size)
#SBATCH -p short-40core       # Partition

# Load required modules
module load mpi4py/latest

# Run the twins code with MPI
mpirun -n 10 python twins_code.py
