#!/bin/bash

#SBATCH --job-name=mapper_job
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/mapper_%A_%a.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/mapper_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=short-28core

module load python/3.9.7

# Define variables
DATA_DIR="/gpfs/projects/AMS598/projects2025_data/project1_data"
SCRIPT_PATH="/gpfs/home/ybhuma/AMS598/map_reduce_4.py"

# Array index (0-3)
ID=${SLURM_ARRAY_TASK_ID}

# Assign files to each mapper (4 per mapper)
case $ID in
  0) FILES="data1.txt data2.txt data3.txt data4.txt" ;;
  1) FILES="data5.txt data6.txt data7.txt data8.txt" ;;
  2) FILES="data9.txt data10.txt data11.txt data12.txt" ;;
  3) FILES="data13.txt data14.txt data15.txt data16.txt" ;;
esac

echo "Mapper $ID processing: $FILES"
python $SCRIPT_PATH --stage mapper --id $ID --files $FILES

