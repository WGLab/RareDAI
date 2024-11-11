#!/bin/bash
#SBATCH --job-name=rareDAI_ft
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
module load CUDA/12.1.1

python RareDAI_finetuning.py
#####################sbatch -p gpu-xe9680q run_RareDAI.sh################