#!/bin/bash

#SBATCH -J BrainDiffusion-mpi                 # Job name
#SBATCH -o log-mpi              # Name of stdout output file (%j expands to jobId)
#SBATCH -p rtx                              # Queue name
#SBATCH -N 1                                # Total number of nodes requested (56 cores/node)
#SBATCH -n 4                                # Total number of mpi tasks requested
#SBATCH -t 02:00:00                         # Run time (hh:mm:ss)
#SBATCH -A BCS23001

source ~/.bashrc
conda activate

mpiexec -n 4 python example_mpi.py 
