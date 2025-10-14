#!/bin/bash

#SBATCH --job-name=pagerank

#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/pagerank_%j.out

#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/pagerank_%j.err

#SBATCH --nodes=1

#SBATCH --ntasks=5

#SBATCH --cpus-per-task=1

#SBATCH --time=00:10:00

#SBATCH --partition=short-40core




module load mvapich2/gcc/64/2.2rc1
module load mpi4py/latest

mpirun -np 5 python /gpfs/projects/AMS598/projects2025_submission/project2_submission/Bhuma_Yaswanthreddy/page_ranking.py
