#!/bin/bash
#SBATCH --partition=edu5 // gpu partition?
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
module load cuda
srun ./naive
srun ./conflict
srun ./coalesced

