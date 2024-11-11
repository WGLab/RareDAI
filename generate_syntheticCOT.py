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
parser = argparse.ArgumentParser(description="Genetic Testing Recommender (Gene Panel or WES/WGS)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input file without COT")
parser.add_argument("-o", "--output", required = True, help="directory to output file with generated COT")
args = parser.parse_args()
model_id = "" # Please replace with your Llama 70B Tokenizer folder
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
    assert "input" in data_point, "The given data is missing 'input' keys." 
    assert "icd" in data_point, "The given data is missing 'icd' keys."  
    #assert "phenotypes" in data, "The given data is missing 'phenotypes' keys." 
    assert "output" in data_point, "The given data is missing 'output' keys." 
    if data_point['output'] == 'gene panel':
        label1 = 'gene panel'
        label2 = 'genome sequencing'
    else:
        label1 = 'genome sequencing'
        label2 = 'gene panel'
    instruction = "You are a genetic counselor. Based on the provided definitions, explain why the recommended test is the most suitable choice for this patient. Be concise, accurate, and avoid fabricating any details.\ngene panel: look for variants in more than one gene. This type of test is often used to pinpoint a diagnosis when a person has symptoms that may fit a wide array of conditions, or when the suspected condition can be caused by variants in many genes. Gene panel is suitable for any patients with distinctive clinical features, family history of a specific disorder, or indicative biochemistry, X ray, or complementary assays in a fast manner and less expensive than exome sequencing.\ngenome sequencing: analyze the bulk of an individual’s DNA to find genetic variations. Whole exome or whole genome sequencing is typically used when single gene or panel testing has not provided a diagnosis, or when the suspected condition or genetic cause is unclear. Genome sequencing is often more accurate and applicable for patients with multiple nonspecific concerns but takes longer to be done."
    question = f"""Explain why the recommended test, {data_point['output']}, is the best choice for the patient described in the following input. Build a detailed, expert, and logical step-by-step reasoning explanation for why {data_point['output']} is strongly preferable to {label2} in this case to convince our physicians and patients using all the given ICD-10 diagnoses and clinical note. Focus on whether the patient has distinct clinical features and a medical history or family history suggesting specific conditions (favoring a gene panel) or presents with unclear symptoms needing comprehensive investigation (favoring genome sequencing). Input:\n"""
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """
    prompt = base_prompt.format(system_prompt = instruction,
                                #user_prompt = question + "|==|Phenotypes|==|\n" + data_point['hpo'] + "\n|==|ICD-10 Diagnosis|==|\n" + data_point['icd'] + "\n|==|Note|==|\n" + data_point['input'],
                                user_prompt = question + "\n|==|ICD-10 Diagnosis|==|\n" + data_point['icd'] + "\n|==|Note|==|\n" + data_point['input'],
                                )
    return prompt
def read_text(input_file):
    if os.path.isfile(input_file):
        input_list=[input_file]
    else:
        input_list = glob.glob(input_file + "/*")
    input_dict = {}
    for f in input_list:
        file_name = f.split('/')[-1]#[:-4]
        file_name = file_name.split('.')[0]
        with open(f, 'r') as r:
            data = json.load(r)
        if isinstance(data, list): # all patients in one file
            for pat_data in data:
                # Assert that the variable is a dictionary
                assert isinstance(pat_data, dict), "The given data is not a dictionary."  
                # Assert that both 'input', 'icd', 'mrn' keys are present
                assert "input" in pat_data, "The given data is missing 'input' key."
                assert "icd" in pat_data, "The given data is missing 'icd' key."
                assert "mrn" in pat_data, "The given data is missing 'mrn' key."
                input_dict[file_name + "_" + pat_data['mrn']] = pat_data
        elif isinstance(data, dict):
            # Assert that both 'input' and 'icd' keys are present
            assert "input" in data, "The given data is missing 'input' key."
            assert "icd" in data, "The given data is missing 'icd' key."
            input_dict[file_name] = data
        else:
            raise ValueError('The given data is in the wrong format. It has to be in either a LIST of dictionary objects (for list of patients) OR a DICTIONARY object (for one patient).')
    return(input_dict)
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
            max_new_tokens=12000,
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
def main():
    input_dict = read_text(args.input)
    train_cot = []
    for file_name, text in tqdm(input_dict.items()):
        print(file_name)
        pred = generate_output(text)
        text['cot'] = pred
        train_cot.append(text)
    with open(args.output, 'w' ) as f:
        json.dump(train_cot, f)
    
if __name__ == "__main__":
    main()
