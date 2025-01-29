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

Our top performing Llama models are:
1) Fine-tuned 8B (16bit) with CoT + ICD10 on summarized clinical notes.
2) Fine-tuned 70B (4bit QLoRA) with CoT + ICD10 + Phenotypes on raw clinical notes.
3) Fine-tuned 8B (16bit) with CoT + ICD10 on raw clinical notes.

## Set Up Model, Input, and Output directories
1. Models:
    - To use LLaMA 3.1 8B (or 70B) model, please apply for access first and download it into the local drive. [Download here](https://www.llama.com/llama-downloads/)
    - Save model in the Llama3_1/Meta-Llama-3.1-8B-Instruct (it should contain the tokenizer and model weights)
2. Input:
    - Input files should be json files including "input", "hpo", "icd", "mrn" for inference and additionally "output" if fine-tuning. You will need to generate synthetic summary ("summary") and Chain-of-Thought (CoT).
    - Input file can be either a single json file or a whole directory containing all input json files
3. Download additional Database
    - Phecode ICD10: download phecode_definitions1.2.csv and Phecode_map_v1_2_icd10_beta.csv from the [link](https://phewascatalog.org/phecodes_icd10). We also provide these databases in this GitHub.

Due to the protected health information, we cannot be shared our fine-tuned models publicly. However, we provide the codes and detailed instructions so that you should be able to replicate our processes with minimal efforts.

## Fine-tuning
The fine-tuning process is divided into four stages:
1. Data collection: please process and clean your own data before fine-tuning/inference. You can refer our paper to see how our notes are selected. Save all the features (input, icd, mrn, phenotypes, output) in the JSON file for each patient.
    - Make sure your ICD10 are converted to Phecodes and separated by "|". You can use some codes in our script to process the data.
    - Example: Intestinal infection | Fractures | joint disorders and dislocations; trauma-related
2. Generate summary for training and validation datasets (only required if you want to fine-tune models with summary note).
    - Please modify the necessary SLURM arguments in [run_summary.sh](https://github.com/WGLab/RareDAI/blob/main/run_summary.sh) to run [generate_summary.py](https://github.com/WGLab/RareDAI/blob/main/generate_summary.py)
    - You should provide the directory where your input JSON-formatted files are located and the file directory of the output in which you want to save the synthetic data. Each of your resulting JSON data should have an additional "summary" key (including testing data). Make sure your input file has the correct keys like mentioned above.
    - Please replace line 17 in the Python script with the foundation Llama 3.1 70B directory.

Up to this point, you should split your own data into train:validation:test (ratio of 6:2:2). You only need to generate synthetic CoT for your train and validation data. You can fine-tune model either on raw clinical notes (data_point['input']) OR summary notes you generated above (data_point['summary]). 

3. Generate synthetic CoT for training and validation datasets.
    - Please modify the necessary SLURM arguments in [run_syntheticCOT.sh](https://github.com/WGLab/RareDAI/blob/main/run_syntheticCOT.sh) to run [generate_syntheticCOT.py](https://github.com/WGLab/RareDAI/blob/main/generate_syntheticCOT.py)
    - You should provide the directory where your training/validation JSON-formatted files are located as the input and the file directory of the output in which you want to save the synthetic data. Each of your resulting JSON data should have an additional "cot" key (except testing data). Make sure your input file has the correct keys like mentioned above.
    - Please replace line 18 in the Python script with the foundation Llama 3.1 70B directory.
    - Make sure to include phenotypes or phecodes (converted ICD10) in your prompt accordingly. If you want to include just ICD-10 (like our top performing 8B model) then use line 57. If you want to include both features (like our top performing 70B) then use 55.
    - If you want to fine-tune the models on raw clinical notes, indicate "input" in your data_point['input'] in line 55-57.
    - If you want to fine-tune the models on summary clinical notes, indicate "summary" in your data_point['summary'] in the line 55-57.
4. Fine-tune the model with generated synthetic CoT from stage 3.
    - Please modify the necessary SLURM arguments in [run_RareDAI.sh](https://github.com/WGLab/RareDAI/blob/main/run_RareDAI.sh) to run [RareDAI_finetuning.py](https://github.com/WGLab/RareDAI/blob/main/RareDAI_finetuning.py)
    - Please replace the Python script at line 24, 96, 97, 106, 120 with the corresponding directory.
    - Similarly to step 3, make sure to include phenotypes and phecodes accordingly to your model of interest (change the line 76-77) and adjust data_point['input'] or data_point['summary'] correctly.

* If you want to fine-tune 70B model, you may need to use QLoRA 4bit + PEFT to reduce the size of the models even though the higher precisions and full-parameter may achieve better results. You should uncomment quantization_config (line 108-114) and QLoRA setup (line 151-165).

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

