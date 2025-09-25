#!/bin/bash
#SBATCH --job-name=final_merge
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/final_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/final_%j.err
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=short-28core

module load python/3.9.7

SCRIPT_PATH="/gpfs/home/ybhuma/AMS598/map_reduce_4.py"

python $SCRIPT_PATH --stage final

