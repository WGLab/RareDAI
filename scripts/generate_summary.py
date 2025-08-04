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
parser = argparse.ArgumentParser(description="Clinical Note Summarizer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input file with raw clinical notes")
parser.add_argument("-o", "--output", required = True, help="directory to output file with generated note summary")
parser.add_argument("-model_id", "--model_id", required = False, help="directory to model")
args = parser.parse_args()
#model_id = "/mnt/isilon/wang_lab/shared/Llama3_1/Meta-Llama-3.1-70B-Instruct" # Please replace with your Llama 70B Tokenizer folder
model_id = args.model_id
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
    instruction = "You are a genetic counselor tasked with summarizing medical notes with the highest level of detail. Your primary objective is to capture and convey key information, ensuring the summary closely reflects the original notes. Whenever possible, use the original wording and sentences from the notes to preserve the accuracy and integrity of the information. Do not generate random details."
    question = f"""In order to conduct a comprehensive genetic assessment, please provide as much detail as possible about the following information:
    Patient Demographics: Age, gender, ethnicity, and any relevant demographic factors.
    Presenting Complaints: Describe the primary complaints or concerns, with a timeline of when symptoms started.
    Previous Assessments: Describe any past genetic evaluations or screenings, including the results and any relevant interpretations.
    Family History: Include any known instances of genetic or hereditary conditions in the family, specifying relationships, affected members, and known genetic mutations.
    Medical History: Provide a detailed personal medical history, including known medical conditions, previous diagnoses, and treatments received.
    Symptoms and Signs: List all medical phenotypes, symptoms and physical signs observed, noting their onset, duration, and progression.
    Diagnostic Tests (where available): Share results and findings from all diagnostic tests performed descibed in the note such as: Physical examinations (if available), Blood tests (e.g., CBC, biochemical markers), Biochemical tests (e.g., metabolic panels, enzyme activity tests), Imaging studies (e.g., MRI, CT, ultrasound), Audiological or visual assessments, and Genetic testing results (e.g., karyotyping, exome or genome sequencing).
    Current Assessment: Provide complete ongoing evaluations or preliminary diagnoses. This is the most important information for downstream genetic assessment.
    Environmental and Lifestyle Factors: Describe relevant environmental exposures, lifestyle habits, or occupational risks that may influence genetic predispositions.
    Future Directions: Mention planned tests, referrals, or interventions that may help refine or confirm a diagnosis.
    Additional Details: Add any other context or information that may assist in making an accurate genetic assessment.
    Input:\n"""
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    """
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + "\n|==|Note|==|\n" + data_point['query'] #data_point['input'],
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
        if "txt" in f:
            with open(f, 'r') as r:
                data = r.readlines()
            data = "".join(data)
            input_dict[file_name] ={'input':data}
        else:
            with open(f, 'r') as r:
                data = json.load(r)
            if isinstance(data, list): # all patients in one file
                for pat_data in data:
                    input_dict[pat_data['mrn']] = pat_data
            elif isinstance(data, dict):
                # Assert that both 'input' and 'icd' keys are present
                assert "input" in data, "The given data is missing 'input' key."
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
    summary = []
    for file_name, text in tqdm(input_dict.items()):
        #print(file_name)
        pred = generate_output(text)
        text['summary'] = pred
        summary.append(text)
    with open(args.output, 'w' ) as f:
        json.dump(summary, f)
    
if __name__ == "__main__":
    main()
