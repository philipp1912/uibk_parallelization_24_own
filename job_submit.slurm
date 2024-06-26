#!/bin/bash

#use sbatch ... to submit the job



# Execute job in the partition "lva" unless you have special requirements.
#SBATCH --partition=lva

# Name your job to be able to identify it later
#SBATCH --job-name=test

# Redirect output stream to this file
#SBATCH --output=code.log

# Number of nodes, to be used.
#SBATCH --nodes=4

# Maximum number of tasks (=processes) to start per node
#SBATCH --ntasks-per-node=12

# Memory used per core (1 node, 2 cpus with 6 cores each, so 12 cores per node)
#SBATCH --mem-per-cpu=1G

# Enforce exclusive node allocation, do not share with other jobs


module purge
module load gcc/12.2.0-gcc-8.5.0-p4pe45v
module load openmpi/3.1.6-gcc-12.2.0-d2gmn55

/usr/bin/time -v mpirun -n $SLURM_JOB_NUM_NODES ./Build_Release/apps/run_full_code_parallel
