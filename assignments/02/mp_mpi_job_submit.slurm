#!/bin/bash

#use sbatch ... to submit the job



# Execute job in the partition "lva" unless you have special requirements.
#SBATCH --partition=lva

# Name your job to be able to identify it later
#SBATCH --job-name=test

# Redirect output stream to this file
#SBATCH --output=mandelbrot_mpi_mp.log

# Number of nodes, to be used.
#SBATCH --nodes=4


# In this example we allocate ressources for 20 hybrid MPI+OpenMP tasks,
# placing exactly 4 tasks on each of 5 separate nodes like this:
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=4
#SBATCH --cpus-per-task=2

# Memory used per core (1 node, 2 cpus with 6 cores each, so 12 cores per node)
#SBATCH --mem-per-cpu=1G

# Enforce exclusive node allocation, do not share with other jobs
#SBATCH --exclusive

module purge
module load gcc/12.2.0-gcc-8.5.0-p4pe45v
module load openmpi/3.1.6-gcc-12.2.0-d2gmn55

# let Slurm take care of both levels of parallelism
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
srun --export=ALL --mpi=pmi2 ./build/mandelbrot_mpi_mp 61440 34560