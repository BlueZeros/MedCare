import os 
import json 
import jsonlines

import argparse 
from tqdm import tqdm, trange
from subprocess import PIPE, Popen, TimeoutExpired
import tempfile
import re 
from pathlib import Path
import signal
from sympy import sympify
import pandas as pd
import evaluate
from medcare.eval.cblue.post_generate_process import process_generated_results
from medcare.eval.cblue.evaluate import calc_scores, report_error_msg, error_msg, report_score
from medcare.eval.gpt4_evaluations.cmb_eval import cmb_eval

def normalize_frac(x):
    # Pattern to match \frac{a}{b}
    pattern = r'\\frac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize_dfrac(x):
    pattern = r'\\dfrac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize(x):
    if "\\frac" in x and normalize_frac(x):
        a, b = normalize_frac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
        
    elif "\\dfrac" in x and normalize_dfrac(x):
        a, b = normalize_dfrac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
    else:
        try:
            x = sympify(x).evalf()
            return float(x)
        except:
            return x

def acc(pred, target):
    return 1 if pred == target else 0

def rouge(pred, target):
    # compute rouge-1, rouge-2, rouge-l
    pass

def extract_bbox_content(s):
    contents = []
    i = 0
    while i < len(s):
        if s[i:i+7] == '\\boxed{':
            depth = 1
            start = i + 7
            i += 7
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        contents.append(s[start:i])
                i += 1
        else:
            i += 1
    return contents

def extract_answer_content(s):
    match = re.search(r'answer is (.*?)(\.|$)', s, re.IGNORECASE)
    return match.group(1) if match else None

def answer_acc(line):
    pred = line['text']

    pred = re.findall(r'[A-E]', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer']

    return 1 if pred == answer else 0 

def mmedbench_acc(line):
    pred = line['text']

    pred = re.findall(r'[A-E]', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if pred == answer else 0 
def record_acc(line):
    pred = line['text']
    answers = line['additional_info']['answers']
    for answer in answers:
        if answer["text"] in pred:
            return 1
    
    return 0

def mmedbench_en_cot_acc(line):
    pred = re.search(r'the answer is (.*?)(\.|$)', line['text'], re.IGNORECASE)
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if f"{answer}." in pred else 0 

def mmedbench_zh_cot_acc(line):
    pred = re.search(r'答案为(.*?)$', line['text'])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        return 0
    else:
        pred = pred[0]
    
    answer = line['additional_info']['answer_idx']

    return 1 if f"{answer}" in pred else 0 

def cblue_score(args, questions):
    answer_file = "./datas/test/CBLUE_structured.json"
    output_path = args.output_file.rsplit("/", 1)[0] + "results.json"
    dict_pred = process_generated_results(questions)
    
    dict_gt = json.load(
        open(answer_file, "r", encoding="utf-8")
    )

    score_map, success_flag = calc_scores(
        dict_gt, dict_pred, output_path
    )

    if success_flag:
        # turn to 100-score format
        score_map = {key: value * 100 for key, value in score_map.items()}
        report_score(score_map, output_path)

def multiplechoice_acc(line):
    pred = re.search(r'答案为(.*?)$', line['text'].rsplit("\n\n\n", 1)[0])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        # import pdb
        # pdb.set_trace()
        answer = line['additional_info']['answer']
        if "错，为本题正确答案" in line['text'] and f"{answer}错，为本题正确答案" in line['text']:
            return 1
        else:
            all_index = "ABCDE"
            for answer in line['additional_info']['answer']:
                all_index = all_index.replace(answer, "")
                if f"{answer}对" not in line['text']:
                    return 0
            # if len(line['additional_info']['answer']) > 1:
            if True:
                for o_answer in all_index:
                    if f"{o_answer}对" in line['text']:
                        return 0
            return 1

    else:
        pred = pred[0]
    
    all_index = "ABCDE"
    answer_list = line['additional_info']['answer']
    for answer in answer_list:
        all_index = all_index.replace(answer, "")
        if answer not in pred:
            return 0
    # if len(answer_list) > 1:
    if True:
        for o_answer in all_index:
            if f"{o_answer}" in pred:
                return 0
    return 1

def multiplechoice_en_acc(line):
    pred = re.search(r'The answer is (.*?)$', line['text'])
    # pred = re.findall(r'[A-E].', pred)
    if pred == [] or pred is None:
        # import pdb
        # pdb.set_trace()
        return 0
    else:
        pred = pred[0]

    # return 1 if line['additional_info']['answer'] in pred else 0
    
    all_index = "ABCDE"
    answer_list = line['additional_info']['answer']
    for answer in answer_list:
        all_index = all_index.replace(answer, "")
        if f"{answer}" not in pred:
            return 0
        
    if len(answer_list) > 1:
        for o_answer in all_index:
            if f"{o_answer}" in pred:
                return 0
    return 1

def cmb_score(args, questions):
    avg_score, questions = cmb_eval(args, questions)
    # print(avg_score)

METRIC_FUNC_MAPPING = {
    "mmedbench_en": mmedbench_en_cot_acc,
    "PLE_TCM": multiplechoice_acc,
    "PLE_Pharmacy": multiplechoice_acc,
    "ceval": multiplechoice_acc,
    "cmmlu": multiplechoice_acc,
    "CMExam": multiplechoice_acc,
    "CMB": multiplechoice_acc,
    "mmlu": multiplechoice_en_acc,
    "MedQA": multiplechoice_en_acc,
    "MedMCQA": multiplechoice_en_acc,
    "medqa_mainland": multiplechoice_acc,
    "mmedbench_zh": multiplechoice_acc,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # input_file is a jsonl file with the following format:
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    
    total_num = len(questions)
    total_score = 0
    dataset_name  = args.input_file.split("/")[-2]
    if dataset_name in METRIC_FUNC_MAPPING:
        acc_func = METRIC_FUNC_MAPPING[dataset_name]
    else:
        acc_func = None
    wrong_idx = []

    if dataset_name in ["CBLUE"]:
        cblue_score(args, questions)
    elif dataset_name in ["CCTE"]:
        cmb_score(args, questions)
    else:
        if "type" in questions[0]['additional_info']:
            type_total = {}
            type_score = {}
            for line in tqdm(questions, total=total_num):
                class_type = line['additional_info']["type"]
                if class_type not in type_total:
                    type_total[class_type] = 1
                    type_score[class_type] = 0
                else:
                    type_total[class_type] += 1

                scores = acc_func(line)

                if scores is None:
                    type_total[class_type] -= 1
                    wrong_idx.append(line)
                    continue

                type_score[class_type] += scores
                if scores == 0:
                    wrong_idx.append(line)
            
            type_acc = {"type": [], "acc": []}
            for class_type in type_total:
                type_acc["type"].append(class_type)
                type_acc["acc"].append(type_score[class_type] / type_total[class_type])
            
            # import pdb
            # pdb.set_trace()
            avg_acc = sum([type_score[class_type] for class_type in type_total]) / sum([type_total[class_type] for class_type in type_total])
            type_acc["type"].append("AVERAGE")
            type_acc["acc"].append(avg_acc)

            # import pdb
            # pdb.set_trace()
            df = pd.DataFrame(type_acc)
            df.to_csv(os.path.join(args.output_file.rsplit("/", 1)[0], "type_acc.csv"), index=False)
            print(f"Acc in {dataset_name}: {avg_acc}")
            
        else:
            for line in tqdm(questions, total=total_num):
                scores = acc_func(line)
                # print("score: ", scores)
                # import pdb
                # pdb.set_trace()
                if scores is None:
                    total_num -= 1
                    wrong_idx.append(line)
                    continue
                total_score += scores
                if scores == 0:
                    wrong_idx.append(line)
        
            avg_acc = total_score / total_num
            print(f"Acc in {dataset_name}: {avg_acc}")

        with open(args.output_file, 'w') as f:
            json.dump(wrong_idx, f, indent=4, ensure_ascii=False)