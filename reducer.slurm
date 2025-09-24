#!/bin/bash

#SBATCH --job-name=reducer_job
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/output_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/error_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short-40core

module load python/3.11.2

python map_reduce.py --stage reducer
