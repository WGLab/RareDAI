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
parser.add_argument("-i", "--input", required = True, help="directory to input folder")
parser.add_argument("-o", "--output", required = True, help="directory to output folder")
parser.add_argument("-model_dir", "--model_dir", required = False, help="directory to model folder")
args = parser.parse_args()
#Tokenizer
tokenizer_id = '' # replace your Llama 3.1 8B directory here
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

def generate_prompt(data_point):
    if "icd" not in data_point.keys() or len(data_point['icd']) == 0:
        icd_text = "Please refer the note for more detailed information."
    else:
        icd_text = data_point['icd']
    instruction = "You are a genetic counselor! Given the following definitions, answer the question concisely and don't make up any random answer.\ngene panel: look for variants in more than one gene. This type of test is often used to pinpoint a diagnosis when a person has symptoms that may fit a wide array of conditions, or when the suspected condition can be caused by variants in many genes. Gene panel is suitable for any patients with distinctive clinical features, family history of a specific disorder, or indicative biochemistry, X ray, or complementary assays in a fast manner and less expensive than exome sequencing.\ngenome sequencing: analyze the bulk of an individual’s DNA to find genetic variations. Whole exome or whole genome sequencing is typically used when single gene or panel testing has not provided a diagnosis, or when the suspected condition or genetic cause is unclear. Genome sequencing is often more accurate and applicable for patients with multiple nonspecific concerns but takes longer to be done."
    question = "What genetic testing do you recommend to the patient in the following input? Please give a detailed explanation and return 'gene panel' or 'genome sequencing' as the final response.\nInput: "
    prompt = question + "\n|==|ICD-10 Diagnosis|==|\n" + icd_text + "\n|==|Note|==|\n" + data_point['input']
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = prompt,
                                )
    return prompt

def generate_output(model, data_point):
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
            max_new_tokens=11000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1, #higher temperature => generate more creative answers but the responses may change
            top_p=0.8, #higher top_p => less likely to generate random answer not in the text
    )

    response = generation_output[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True).strip()
    if len(input_ids[0]) > 11000:
        print("WARNING: Your text input has more than the predefined maximum 11000 tokens. The results may be defective.")
    return(output)
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
def main():
    ##set up model
    #Model
    if args.model_dir:
        model_id = args.model_dir + '/model/'
    else:
        model_id = os.getcwd() + '/model/RareDAI/model/'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    model.resize_token_embeddings(len(tokenizer)) ## go along with tokenizer.pad_token is None
    model.config.pad_token_id = tokenizer.pad_token_id # setting pad token id for model
    model.eval()
    print('start rareDAI')
    input_dict = read_text(args.input)
    for file_name, text in tqdm(input_dict.items()):
        print(file_name)
        # generate raw response
        raw_output = generate_output(model, text)
        # clean up response
        # save output
        os.makedirs(args.output, exist_ok=True)
        output_name = args.output+"/"+file_name+"_rareDAI.txt"
        with open(output_name, 'w') as f:
            f.write(raw_output)
        print(raw_output)
if __name__ == "__main__":
    main()