#!/bin/bash
#SBATCH --job-name=synthetic_COT
module load CUDA/12.1.1
#!/bin/bash

usage() {
    echo "Usage: $0 -train_dir <train_file> -o <output_file> -model_path <model_path> [optional flags: -val_dir <val_file> -hpo -icd -cot -summary -lora -qlora]"
    exit 1
}

# Initialize variables
train_dir=""
val_dir=""
output=""
model_path=""
extra_args=()

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -train_dir) train_dir="$2"; shift ;;
        -val_dir) val_dir="$2"; shift ;;
        -o) output="$2"; shift ;;
        -model_path) model_path="$2"; shift ;;
        -hpo| -icd| -cot| -summary| -lora| -qlora)
            extra_args+=("$1")
            ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check required arguments
if [[ -z "$train_dir" || -z "$output" || -z "$model_path" ]]; then
    echo "Error: -train_dir, -o, and -model_path are required."
    usage
fi

# Build the command
cmd="python RareDAI_finetuning.py --train_dir \"$train_dir\" --output \"$output\" --model_path \"$model_path\""

# Optional arguments
[[ -n "$val_dir" ]] && cmd="$cmd --val_dir \"$val_dir\""

# Add boolean flags
for flag in "${extra_args[@]}"; do
    cmd="$cmd $flag"
done

# Run the final command
eval $cmd

#sbatch -p gpu-xe9680q --gres=gpu:h100:2 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_summary.sh -i /home/nguyenqm/projects/github/RareDAI/gene_training_data.json -o /home/nguyenqm/projects/github/RareDAI/gene_training_data_summary.json -model_path /mnt/isilon/wang_lab/shared/Llama3_3/Llama-3.3-70B-Instruct/'
#sbatch -p gpu-xe9680q --gres=gpu:h100:2 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_summary.sh -i /home/nguyenqm/projects/github/RareDAI/gene_val_data.json -o /home/nguyenqm/projects/github/RareDAI/gene_val_data_summary.json -model_path /mnt/isilon/wang_lab/shared/Llama3_3/Llama-3.3-70B-Instruct/"

