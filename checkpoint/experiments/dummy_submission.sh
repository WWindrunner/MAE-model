#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --job-name=mae
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=general-gpu
#SBATCH --time=12:00:00

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_%t_log.out --error /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_%t_log.err /home/qiy17007/miniconda3/envs/mae/bin/python -u -m submitit.core._submit /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j
