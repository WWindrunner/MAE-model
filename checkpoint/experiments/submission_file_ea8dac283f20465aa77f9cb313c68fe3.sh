#!/bin/bash

# Parameters
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=10
#SBATCH --error=/shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_0_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=mae
#SBATCH --mem=320GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_0_log.out
#SBATCH --partition=learnfair
#SBATCH --signal=USR2@120
#SBATCH --time=4320
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_%t_log.out --error /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j/%j_%t_log.err /home/qiy17007/miniconda3/envs/mae/bin/python -u -m submitit.core._submit /shared/stormcenter/Qing_Y/GAN_ChangeDetection/MAE/mae/checkpoint/experiments/%j
