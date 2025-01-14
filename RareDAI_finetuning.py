import os
from datasets import load_dataset
import datasets
import torch
from tokenizers import AddedToken, pre_tokenizers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
torch.backends.cuda.matmul.allow_tf32 = True
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm.auto import tqdm
import gc, json
gc.collect()
torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_name = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-8B-Instruct" # Please provide the directory to the foundation Llama 3.1 model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"
def tokenize(prompt, add_eos_token=True, tokenizer = tokenizer):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    CUTOFF_LEN = 11000 ## Maximum token length for a single input text (roughly 8192 words)
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
def generate_prompt(data_point):
    assert "input" in data_point, "The given data is missing 'input' keys." 
    assert "icd" in data_point, "The given data is missing 'icd' keys."  
    assert "cot" in data_point, "The given data is missing 'cot' keys." # this is generated from Stage 2
    assert "output" in data_point, "The given data is missing 'output' keys." 
    instruction = "You are a genetic counselor adhering to the standards and guidelines set by the American College of Medical Genetics (ACMG). Given the following definitions, answer the question concisely and don't make up any random answer.\ngene panel: look for variants in more than one gene. This type of test is often used to pinpoint a diagnosis when a person has symptoms that may fit a wide array of conditions, or when the suspected condition can be caused by variants in many genes. Gene panel is suitable for any patients with distinctive clinical features, family history of a specific disorder, or indicative biochemistry, X ray, or complementary assays in a fast manner and less expensive than exome sequencing.\ngenome sequencing: analyze the bulk of an individual’s DNA to find genetic variations. Whole exome or whole genome sequencing is typically used when single gene or panel testing has not provided a diagnosis, or when the suspected condition or genetic cause is unclear. Genome sequencing is often more accurate and applicable for patients with multiple nonspecific concerns but takes longer to be done."
    #question = "What genetic testing do you recommend to the patient in the following input? Please give a detailed explanation and return 'gene panel' or 'genome sequencing' as the final response.\nInput: "
    question = """What genetic testing do you recommend to the patient in the following input? Please give a detailed explanation and return 'gene panel' or 'genome sequencing' as the final response. You should rely on the given questions below to build your logical answer.\n
    1. Is the patient presenting with congenital abnormalities or developmental disorders? According to ACMG guidelines, exome or genome sequencing should be considered as a first or second-tier test for these conditions to provide a comprehensive diagnostic approach.
    2. Does the patient’s condition or suspected genetic disorder involve multiple genes or a complex phenotype that cannot be confidently explained by a single-gene or targeted panel approach? If yes, ACMG guidelines support the use of exome or genome sequencing for broader evaluation and potential reanalysis over time.
    3. Does the patient have distinctive clinical features, biochemical findings, or imaging results that suggest a particular genetic condition or set of conditions? If yes, a targeted gene panel may be the most efficient approach. If no, exome or genome sequencing might be more appropriate to uncover a broader range of potential genetic causes.
    4. Does the patient have a high likelihood of a specific genetic disorder based on family history? If yes, ACMG guidelines recommend starting with targeted testing or a gene panel to efficiently identify causative variants.
    5. Has the patient undergone prior genetic testing or any diagnostic tools, and were the results inconclusive or insufficient for diagnosis? If yes, exome or genome sequencing should be considered to provide a broader diagnostic perspective. If no, a gene panel may be the first step, especially if the phenotype suggests a specific set of conditions.
    6. Is the patient in an urgent care setting, such as the NICU or ICU, requiring rapid results for clinical management?  If yes, rapid exome or genome sequencing is preferred for its ability to quickly evaluate most protein-coding genes at once.
    7. Are there cost or accessibility concerns that might limit the use of exome or genome sequencing as a first-tier test? If yes, ACMG guidelines suggest that a targeted approach, such as a gene panel, may be a pragmatic starting point while considering sequencing as a follow-up.
    Input:\n"""
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {model_answer}<|eot_id|><|end_of_text|>"""
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + "\n|==|ICD-10 Diagnosis|==|\n" + data_point['icd'] + "\n|==|Note|==|\n" + data_point['input'],
                                model_answer = "|==|Explanation|==|\n" + data_point['cot'].strip() + "\n|==|Response|==|\n" + data_point['output']
                                )
    return prompt
def generate_and_tokenize_prompt(data_point): ## formulate the input text template and tokenize to numbers
    full_prompt = generate_prompt(data_point) # if just use raw text as input => for pretraining
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt
def defining_args():
    return f"""
    Any notes here to differentiate the different runs. Not important!
    8B full params
    7 questions during CoT generation with ACMG guidelines
    Full VERSION
    include 7 questions during finetuning
    """
def main():
    """
    Set training parameters and train model
    """
    train_data = load_dataset("json", data_files="/home/nguyenqm/projects/github/RareDAI/gene_training_data_cot.json", split = 'train') # Please provide the directory to your training data with synthetic CoT
    val_data = load_dataset("json", data_files="/home/nguyenqm/projects/github/RareDAI/gene_training_data_cot.json", split = 'train') # Please provide the directory to your validation data with synthetic CoT
    print(generate_prompt(train_data[0]))

    train_data = (
        train_data.map(generate_and_tokenize_prompt)
    )
    val_data = (
        val_data.map(generate_and_tokenize_prompt)
    )
    model_name = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-8B-Instruct" # Please provide the directory to the foundation Llama 3.1 model
    model=AutoModelForCausalLM.from_pretrained(model_name,do_sample=True, #quantization_config=quantization_config,
                                            attn_implementation="flash_attention_2",
                                            torch_dtype=torch.bfloat16, device_map = 'auto')
    model.resize_token_embeddings(len(tokenizer)) ## go along with tokenizer.pad_token is None
    model.config.pad_token_id = tokenizer.pad_token_id
    #out_dir = os.getcwd() + '/model/RareDAI/'
    out_dir = os.getcwd() + '/model/RareDAI_ACMGquestionsCOT_written/'
    os.makedirs(out_dir, exist_ok=True)
    out_dir_model = out_dir + '/model'
    with open(out_dir + '/params.txt', 'w') as f:
        f.write(defining_args())
    os.makedirs(out_dir_model, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=out_dir,
        warmup_ratio=0.3,
        optim="adamw_torch_fused",# use fused adamw optimizer, default parameters
        per_device_train_batch_size=2, #1
        gradient_accumulation_steps=20, #4
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=1000,
        bf16=True,    # mixed precision
        tf32=True,
        do_eval=True,
        eval_strategy = 'steps',
        per_device_eval_batch_size=1,
        eval_steps = 10,
        eval_accumulation_steps = 1,
        #per_device_eval_batch_size=10,
        weight_decay=0.01,
        save_total_limit=2,
        push_to_hub=False,
        num_train_epochs=10,
    )
    # LORA_R = 128 #128
    # LORA_ALPHA = 256 #256
    # LORA_DROPOUT= 0.05
    # LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj","lm_head"]
    # model = prepare_model_for_kbit_training(model)
    # config = LoraConfig(
    #     r=LORA_R,
    #     lora_alpha=LORA_ALPHA,
    #     target_modules=LORA_TARGET_MODULES,
    #     lora_dropout=LORA_DROPOUT,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model.gradient_checkpointing_enable()
    # model = get_peft_model(model, config)
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    #try:
    trainer.train()
    trainer.save_model(out_dir_model)
    # except:
    #     print(os.system("nvidia-smi"))
if __name__ == "__main__":
    main()
