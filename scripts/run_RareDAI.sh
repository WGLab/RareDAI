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
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 -i <input_file> -o <output_file> -model_path <model_path>"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift ;;
        -o) output="$2"; shift ;;
        -model_path) model_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required arguments are provided
if [[ -z "$input" || -z "$output"  || -z "$model_path" ]]; then
    echo "Error: -i, -o, -model_path are required arguments."
    usage
fi
python RareDAI_finetuning.py
#####################sbatch -p gpu-xe9680q run_RareDAI.sh################
#####################sbatch -p gpuq run_RareDAI.sh################

#tokenizer_name = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-8B-Instruct" # Please provide the directory to the foundation Llama 3.1 model
