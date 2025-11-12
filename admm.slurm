#!/bin/bash
#
#SBATCH --job-name=project3_admm
#SBATCH --output=project3.log
#SBATCH --error=project3.err
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:30:00       # 30 minutes
#SBATCH -p short-28core       # Partition

module purge
module load python/3.9.7
module load openmpi/4.1.5 
module load mpi4py/latest
mpirun -n 10 python admm.py 
