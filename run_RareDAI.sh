#!/bin/bash
#SBATCH --job-name=rareDAI_ft
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nguyenqm@chop.edu
module load CUDA/12.1.1

python RareDAI_finetuning.py
#####################sbatch -p gpu-xe9680q run_RareDAI.sh################
#####################sbatch -p gpuq run_RareDAI.sh################