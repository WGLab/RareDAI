#!/bin/bash
#SBATCH --job-name=synthetic_COT
module load CUDA/12.1.1
# Display usage if the required arguments are not provided
# Display usage if the required arguments are not provided
usage() {
    #echo "Usage: $0 -i <input_file> -o <output_file> [-hpo] [-icd]"
    echo "Usage: $0 -i <input_file> -o <output_file> [-hpo] [-icd] [-summary]"
    exit 1
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) input="$2"; shift ;;
        -o) output="$2"; shift ;;
        -hpo) hpo=true ;;
        -icd) icd=true ;;
        -summary) summary=true ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# # Check if required arguments are provided
# if [[ -z "$input" || -z "$output" ]]; then
#     echo "Error: -i and -o are required arguments."
#     usage
# fi

# Construct the python command dynamically
cmd="python generate_syntheticCOT.py -i \"$input\" -o \"$output\""
if [[ -n "$hpo" ]]; then
    cmd+=" -hpo"
fi
if [[ -n "$icd" ]]; then
    cmd+=" -icd"
fi
if [[ -n "$summary" ]]; then
    cmd+=" -summary"
fi
# Run the python command
echo "Executing: $cmd"
eval $cmd

#sbatch -p gpu-xe9680q --gres=gpu:h100:2 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_syntheticCOT.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_training_data_summary.json -o ... -hpo -icd -summary"
#sbatch -p gpuq --gres=gpu:a100:2 --cpus-per-gpu=3 --mem-per-cpu=50G --time=3-00:00:00 --profile=all --export=ALL --wrap="bash run_syntheticCOT.sh -i /home/nguyenqm/projects/github/RareDAI/summary/gene_val_data_summary.json -o ... -hpo -icd -summary"
