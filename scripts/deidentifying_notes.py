import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os, sys, re, torch, json, glob, argparse, gc, ast
from itertools import chain
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="Deidentifying Clinical Notes",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input file")
args = parser.parse_args()
model_id = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-8B-Instruct" # Please replace with your Llama 70B Tokenizer folder
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"
model.config.pad_token_id = tokenizer.pad_token_id # setting pad token id for model
def generate_prompt(data_point):
    instruction = (
        "De-identify this clinical note by replacing only the following personal identifiable information (PII) "
        "such as patient names, doctor names, hospital names, addresses, phone numbers, emails, and dates "
        "with appropriate placeholders (e.g., [PATIENT_NAME], [DOCTOR_NAME], [HOSPITAL_NAME]). "
        "Please do not deidentify family history, chief complaints, signs, symptoms, conditions, medical phenotypes, treatments, diagnoses, or medical test results (from patients or their family). Keep them unchanged so doctors can review the original information. "
        "Keep every other word exactly the same. Do not rephrase or summarize."
    )    
    question = f"""Clinical Note:\n""" + data_point['cot']
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question)
    return prompt
def generate_output(data_point):
    prompt = generate_prompt(data_point)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    #model.to(device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=7000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8, #higher temperature => generate more creative answers but the responses may change
            top_p=0.8, #higher top_p => less likely to generate random answer not in the text
    )
    response = generation_output[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True).strip()
    if len(input_ids[0]) > 12000:
        print("WARNING: Your text input has more than the predefined maximum 12000 tokens. The results may be defective.")
    return(output)

def deidentifying_data(data_dir):
    with open(f'{data_dir}_deidentified.json', 'r') as f:
        summary_data = json.load(f)
    for index, data in tqdm(enumerate(summary_data), total=len(summary_data), bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} elapsed, {remaining} left, {rate_fmt}]"):
        summary_data[index]['cot_deidentified'] = generate_output(data)
    with open(f'{data_dir}_deidentified.json', 'w') as f:
        json.dump(summary_data, f)
    return summary_data
def main():
    data = deidentifying_data(args.input)
    print(data[20]['summary'])
    print()
    print(data[20]['summary_deidentified'])
if __name__ == "__main__":
    main()

