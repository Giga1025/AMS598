#!/bin/bash

#SBATCH --job-name=pagerank
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/pagerank_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/pagerank_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=short-40core
#SBATCH --mem=16G

module load python/3.11.2
module load openmpi/gcc/64/4.1.5
module load mpi4py/latest

mpirun -np 5 python3 /gpfs/home/ybhuma/AMS598/page_ranking.py