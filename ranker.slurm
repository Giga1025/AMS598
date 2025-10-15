#!/bin/bash

#SBATCH --job-name=pagerank
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs_page/pagerank_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs_page/pagerank_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=short-40core

module load python/3.11.2
module load mvapich2/gcc/64/2.2rc1
module load mpi4py/latest

srun python3 /gpfs/home/ybhuma/AMS598/page_ranking.py
