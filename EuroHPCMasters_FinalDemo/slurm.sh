#!/bin/bash
#SBATCH -A bsc99
#SBATCH --qos=gp_bench
#SBATCH --job-name=LPLCE5PT
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

module purge
module load oneapi/2023.2.0
module load hdf5
module load python/3.12.1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

#compile
mpiicx laplace_fdm_5pt.c -o lf5.exe

#execute
srun ./lf5.exe

#create plots
python multiple_plots.py
