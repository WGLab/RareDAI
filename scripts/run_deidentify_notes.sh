#!/bin/bash
#SBATCH --job-name=synthetic_COT
module load CUDA/12.1.1
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 -i <input_file>"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required arguments are provided
if [[ -z "$input" ]]; then
    echo "Error: -i and -o are required arguments."
    usage
fi

python deidentifying_notes.py -i "$input"

#sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_training_data_summary_cot_ICD"
#sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_val_data_summary_cot_ICD"
#sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_training_data_summary_cot_PnICD"
#sbatch -p gpu-xe9680q --gres=gpu:h100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_val_data_summary_cot_PnICD"
#sbatch -p gpuq --gres=gpu:a100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_training_data_summary_cot"
#sbatch -p gpuq --gres=gpu:a100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_val_data_summary_cot"
#sbatch -p gpuq --gres=gpu:a100:1 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_deidentify_notes.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_test_data_summary"
