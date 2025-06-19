#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH -p gpu

#SBATCH --output=/home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/mae_initial-%j.out

set -e

cd /home/uwm/maopuxu

source /home/uwm/maopuxu/miniconda3/bin/activate

conda activate mae_test

python3 /home/uwm/maopuxu/MAE_Topography_Reconstruction/MAE-Topography/mae/main_pretrain.py --input_size 256 --data_path /tank/data/SFS/xinyis/shared/data/mae/one_meter_samples/OH_train_256 --output_dir /tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/256 --log_dir /tank/data/SFS/xinyis/shared/data/mae/output_dir_one_meter/256
