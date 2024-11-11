# RareDAI
RareDAI is an advanced LLM technique, fine-tuned on LLama 3.1 models, designed to support genetic counselors and patients in choosing the most appropriate molecular genetic tests, such as gene panels or WES/WGS, through clear and comprehensive explanations. The model in the paper was fine-tuned using data from the Children’s Hospital of Philadelphia (CHOP). Due to the presence of protected health information, we cannot publicly release the model; however, we have provided guidelines for adapting or fine-tuning LLMs on in-house data. The model accepts clinical notes and Phecodes (converted from ICD-10) as input. You can fine-tune your own model with additional details (such as phenotypes HPO, demographics, etc); however, we recommend that the additional information may be only useful if they are concise and not redundant or irrelevant. This process is elaborated in the subsequent [section](## Fine-tuning). 

RareAI is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

## Installation
We need to install the required packages for model fine-tuning and inference. 
```
conda create -n raredai python=3.11
conda activate raredai
conda install pandas numpy scikit-learn matplotlib seaborn requests
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.1" cuda-toolkit
conda install -c conda-forge jupyter
conda install intel-openmp blas
conda install mpi4py
pip install transformers datasets
pip install fastobo sentencepiece einops protobuf
pip install evaluate sacrebleu scipy accelerate deepspeed
pip install git+https://github.com/huggingface/peft.git
# PLEASE LOAD CUDA MODE IN YOUR ENVIRONMENT BEFORE INSTALL FLASH ATTENTION PACKAGE. FOR EXAMPLE BELOW:
module load CUDA/12.1.1
pip install flash-attn --no-build-isolation
pip install xformers
pip install bitsandbytes
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=raredai
```

In the command above, we utilize the accelerate package for model sharding. PEFT package is used for efficient fine-tuning like LORA. bitsandbytes package is used for model quantization. Please pip uninstalll package and pip install package if you encounter any running issues.

## Set Up Model, Input, and Output directories
1. Models:
    - To use LLaMA 3.1 8B model, please apply for access first and download it into the local drive. [Download here](https://www.llama.com/llama-downloads/)
    - Save model in the Llama3_1/Meta-Llama-3.1-8B-Instruct (it should contain the tokenizer and model weights)
    - Download the updated fine-tuning in the release section on GitHub (Latest version: v1.0.0)
    - Save model weights in the ./model/
2. Input:
    - Input files should be json files including "input", "icd", "mrn" for inference and additionally "output" if fine-tuning.
    - Input file can be either a single json file or a whole directory containing all input json files
3. Download additional Database
    - Phecode ICD10: download phecode_definitions1.2.csv and Phecode_map_v1_2_icd10_beta.csv from the [link](https://phewascatalog.org/phecodes_icd10).

## Fine-tuning
The fine-tuning process is divided into three stages:
1. Data collection: please process and clean your own data before fine-tuning/inference. You can refer our paper to see how our notes are selected. Save all the features (input, icd, mrn, phenotypes, output) in the JSON file for each patient.
    - Make sure your ICD10 are converted to Phecodes and separated by "|". You can use some codes in our script to process the data.
    - Example: Phecode A | Phecode B | Phecode C
2. Generate synthetic CoT for training and validation datasets.
    - Please modify the necessary SLURM arguments in [run_syntheticCOT.sh](https://github.com/WGLab/RareDAI/blob/main/run_syntheticCOT.sh) to run [generate_syntheticCOT.py](https://github.com/WGLab/RareDAI/blob/main/generate_syntheticCOT.py)
    - You should provide the directory where your training/validation JSON-formatted files are located as the input and the file directory of the output in which you want to save the synthetic data. Each of your JSON data should have an additional "cot" key (except testing data). Make sure your input file has the correct keys like mentioned above.
    - Please replace the Python script at line 17 with the foundation Llama 3.1 directory.
3. Fine-tune the model with generated synthetic CoT from stage 2.
    - Please modify the necessary SLURM arguments in [run_RareDAI.sh](https://github.com/WGLab/RareDAI/blob/main/run_RareDAI.sh) to run [RareDAI_finetuning.py](https://github.com/WGLab/RareDAI/blob/main/RareDAI_finetuning.py)
    - Please replace the Python script at line 24, 83, 84, 93 with the corresponding directory.
* If you're encountering CUDA memory issues, it’s likely due to large input texts exceeding your system’s capacity for model training. To address this issue, consider either increasing the number of GPUs and CPUs or adjusting training parameters by reducing the batch size or increasing gradient accumulation steps.

## Inference
Please fine-tune your own model first. Please follow the inference section of the [inference.py](https://github.com/WGLab/RareDAI/blob/main/inference.py) to run your model.

Please use the following command:
```
python inference.py -i your_input_folder_directory -o your_output_folder_directory
```

## Developers:
Quan Minh Nguyen - Bioengineering PhD student at the University of Pennsylvania

Dr. Kai Wang - Professor of Pathology and Laboratory Medicine at the University of Pennsylvania and Children's Hospital of Philadelphia

