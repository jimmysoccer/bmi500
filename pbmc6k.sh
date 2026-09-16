#!/bin/bash -l
#SBATCH -J pbmc6k_profile
#SBATCH -p overflow
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=./slurm_outputs/pbmc6k_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bmi500

cd ~/bmi500

/usr/bin/time -v python scanpy_pbmc.py \
    --data-dir data \
    --data-set pbmc6k \
    --out-dir data \
    --num-threads 1