#!/bin/bash
#SBATCH --job-name=reducer_job
#SBATCH --output=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/reducer_%j.out
#SBATCH --error=/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/logs/reducer_%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:04:00
#SBATCH --partition=short-28core

module load python/3.11.2

SCRIPT_PATH="/gpfs/home/ybhuma/AMS598/map_reduce_4.py"
MAP_OUT_DIR="/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/map_outputs"

# Get all map output files and split into 4 groups
ALL_FILES=($(ls $MAP_OUT_DIR/map_*.txt))
TOTAL=${#ALL_FILES[@]}
CHUNK=$((TOTAL / 4))

# Get reducer ID from SLURM_PROCID or manually loop it
for i in {0..3}; do
  START=$((i * CHUNK))
  END=$((START + CHUNK))
  REDUCE_FILES="${ALL_FILES[@]:$START:$CHUNK}"

  echo "Reducer $i processing files: ${REDUCE_FILES[@]}"
  python $SCRIPT_PATH --stage reducer --id $i --files ${REDUCE_FILES[@]}
done



