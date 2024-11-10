#!/bin/bash
#SBATCH --job-name=synthetic_COT
module load CUDA/12.1.1
# Display usage if the required arguments are not provided
usage() {
    echo "Usage: $0 -i <input_file> -o <output_file>"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift ;;
        -o) output="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required arguments are provided
if [[ -z "$input" || -z "$output" ]]; then
    echo "Error: -i and -o are required arguments."
    usage
fi

python generate_syntheticCOT.py -i "$input" -o "$output"

#sbatch -p gpuq --gres=gpu:a100:4 --cpus-per-gpu=3 --mem-per-cpu=50G --time=2-00:00:00 --profile=all --export=ALL --wrap="bash run_syntheticCOT.sh -i /home/nguyenqm/projects/Genetic_Testing/gene_training_data.json -o /home/nguyenqm/projects/github/rareDAI/gene_training_data_cot.json"
