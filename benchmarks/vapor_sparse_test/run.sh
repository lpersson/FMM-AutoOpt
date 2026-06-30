#!/bin/bash

#SBATCH -n 1 
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 02:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_USE_GPU_BUFFER_OPS=1

module purge
#source /path/to/gromacsFMM/bin/GMXRC

# For exact reproduction of the reference output, use processed_structures/frame*.gro
# instead of input/frame*.gro, because those files include the generated velocities.

python fmm_autoopt.py \
	-i input/frame*.gro \
	--top input/topol.top \
	--mdp FMM_soln.mdp \
	--ref 0 20 \
	--openboundary 0 \
	--sparse test \
        --maxerr 0.02 
