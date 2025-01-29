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
parser = argparse.ArgumentParser(description="Chain of Thought Synthetic Generator",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input file without COT")
parser.add_argument("-o", "--output", required = True, help="directory to output file with generated COT")
args = parser.parse_args()
model_id = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-70B-Instruct" # Please replace with your Llama 70B Tokenizer folder
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
    assert "output" in data_point, "The given data is missing 'output' keys." 
    if data_point['output'] == 'gene panel':
        label1 = 'gene panel'
        label2 = 'genome sequencing'
    else:
        label1 = 'genome sequencing'
        label2 = 'gene panel'
    instruction = "You are a genetic counselor adhering to the standards and guidelines set by the American College of Medical Genetics (ACMG). Based on the provided definitions, explain why the recommended test is the most suitable choice for this patient. Be concise, accurate, and avoid fabricating any details.\ngene panel: look for variants in more than one gene. This type of test is often used to pinpoint a diagnosis when a person has symptoms that may fit a wide array of conditions, or when the suspected condition can be caused by variants in many genes. Gene panel is suitable for any patients with distinctive clinical features, family history of a specific disorder, or indicative biochemistry, X ray, or complementary assays in a fast manner and less expensive than exome sequencing.\ngenome sequencing: analyze the bulk of an individual’s DNA to find genetic variations. Whole exome or whole genome sequencing is typically used when single gene or panel testing has not provided a diagnosis, or when the suspected condition or genetic cause is unclear. Genome sequencing is often more accurate and applicable for patients with multiple nonspecific concerns but takes longer to be done."
    question = f"""Explain why we recommend {data_point['output']} as the best choice for the patient described in the following clinical note. Build a detailed, expert, and logical step-by-step reasoning explanation for why {data_point['output']} is strongly preferable to {label2} in this case to convince our physicians and patients from all the given phenotypes and clinical note. You should rely on the given questions to build your logical answer. 
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
    
    """
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + "\n|==|Phenotypes|==|\n" + data_point['hpo'] + "\n|==|ICD-10 Diagnosis|==|\n" + data_point['icd'] + "\n|==|Note|==|\n" + data_point['summary'], # replace "summary" with "input" if you want to run the full clinical notes
                                #user_prompt = question + "\n|==|Phenotypes|==|\n" + data_point['hpo'] + "\n|==|Note|==|\n" + data_point['input'],
                                #user_prompt = question + "\n|==|ICD-10 Diagnosis|==|\n" + data_point['icd'] + "\n|==|Note|==|\n" + data_point['input'],
                                #user_prompt = question + "\n|==|Note|==|\n" + data_point['input'],
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
                assert "hpo" in pat_data, "The given data is missing 'hpo' key."
                assert "icd" in pat_data, "The given data is missing 'icd' key."
                assert "mrn" in pat_data, "The given data is missing 'mrn' key."
                input_dict[file_name + "_" + pat_data['mrn']] = pat_data
        elif isinstance(data, dict):
            # Assert that both 'input' and 'icd' keys are present
            assert "input" in data, "The given data is missing 'input' key."
            assert "hpo" in pat_data, "The given data is missing 'hpo' key."
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
