#!/bin/bash

#SBATCH --job-name=eecs592_hw2_sync # Job name
#SBATCH --nodes=1               	# Number of nodes
#SBATCH --ntasks-per-node=40    	# CPUs per node (MAX=40 for CPU nodes and 80 for GPU)
#SBATCH --output=out-%j-%N.log  	# Standard output (log file)
#SBATCH --partition=compute     	# Partition/Queue
#SBATCH --time=0:15:00          	# Maximum walltime of 5 minutes

module purge
module use /cm/shared/modulefiles_local
module load shared
module load slurm
module load gcc
module list




for N in 2 4 8 16 32 40
do
    echo -
    export OMP_NUM_THREADS=$N
    echo $OMP_NUM_THREADS
    ./matmult_p 4096 final_3.1.csv
done