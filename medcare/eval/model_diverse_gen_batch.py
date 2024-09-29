import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid


from medcare.conversations import conv_templates
from medcare.model.builder import load_pretrained_model, load_molora_pretrained_model, load_pretrained_orth_model
from medcare.utils import disable_torch_init, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset:
    def __init__(self, questions, batch_size, conv_mode, task_specific_prompt):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)
        self.conv = conv_templates[conv_mode].copy()
        self.task_specific_prompt = task_specific_prompt

    def __getitem__(self, index):
        bz = self.batch_size

        # return question, ansewr, additional info
        questions = []
        prompts = []
        answers = []
        additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                conv = self.conv.copy()

                line = self.questions[i]
                question = line['conversations'][0]['value']
                questions.append(question)
                conv.append_message(conv.roles[0], question+self.task_specific_prompt)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())
                answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                additional_infos.append(line['eval'] if 'eval' in line else None)

        return questions, prompts, answers, additional_infos

    def __len__(self):
        return len(self.questions) // self.batch_size + 1

    def __iter__(self):
        # 返回迭代器对象本身
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            # 返回下一个值并更新索引
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            # 没有更多元素时抛出StopIteration异常
            raise StopIteration


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.question_file.split("/")[-1].split(".")[0] in ["mmedbench_zh", "ceval", "cmmlu", "race_high", "race_middle", "mmedbench_en", "mmlu", "arc", "winogrande"]:
        args.use_logit_bias = True

    # else:
    if "orth" in model_path or "2lora" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space, other_model = load_pretrained_orth_model(model_path, args.model_base, args.lora_name_or_path, only_load=args.only_load)
    elif "molora" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_molora_pretrained_model(model_path, args.model_base, model_name, only_load=args.only_load)
    else:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_pretrained_model(model_path, args.model_base, model_name, infer_args=args)
    tokenizer.padding_side = "left"
    tokenizer_with_prefix_space.padding_side = "left"

    # load args.question_file, which is a csv file
    if args.question_file.endswith(".csv"):
        questions = pd.read_csv(args.question_file)
        questions = convert_to_json(questions)
    elif args.question_file.endswith(".jsonl"):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    else:
        # a json file
        with open(args.question_file, 'r') as f:
            questions = json.load(f)
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # import pdb
    # pdb.set_trace()

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume and os.path.exists(answers_file):
        current_file_num = 0
        with open(answers_file, 'r') as f:
            for line in f:
                current_file_num += 1
        questions = questions[current_file_num:]
        ans_file = open(answers_file, "a", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # data_loader = create_data_loader(questions, tokenizer, model.config)
    model: torch.nn.Module
    model.eval()
    sequence_bias = None
    def get_tokens_as_tuple(word):
        return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])

    task_specific_prompt = ""
    dataset_name = args.question_file.split("/")[-1].split(".")[0]
    if dataset_name in ['mmedbench_en', "mmlu", "MedQA", "MedMCQA"]:
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name in ["mmedbench_zh", "PLE_Pharmacy", "PLE_TCM", "cmmlu", "ceval", "CMB", "CMExam"]:
        task_specific_prompt = "\n\n请在回答的最后用以下格式回答：答案为{answer}。"

    elif dataset_name in ["CCTE", "CBLUE"]:
        pass

    else:
        raise NotImplementedError
    
    if "7b" in args.model_path.lower():
        args.batch_size = 8
    
    elif "14b" in args.model_path.lower():
        args.batch_size = 4

    elif "34b" in args.model_path.lower():
        args.batch_size = 4

    dataset = CustomDataset(questions, batch_size=args.batch_size, conv_mode=args.conv_mode, task_specific_prompt=task_specific_prompt)
    for idx in trange(len(dataset)):
        questions, prompts, answers, additional_infos = dataset[idx]
        if len(questions) == 0:
            break

        stop_str = [tokenizer.eos_token]

        # print("[FIRST INPUT]: ", prompt)
        input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                stop_strings=stop_str,
                eos_token_id=tokenizer("<|im_end|>")["input_ids"][-1],
                sequence_bias=sequence_bias,
                use_cache=True)
        # print(input_ids.shape, output_ids.shape)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        if "The answer is" in prompts[0]:
            cot_prompt = "\nThe answer is "
        elif "答案为" in prompts[0]:
            cot_prompt = "\n答案为"
        
        conv = conv_templates[args.conv_mode].copy()
        cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
        input_ids = tokenizer(cot_prompts, return_tensors='pt', padding=True).input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
        
        if dataset_name not in ["CMExam_cot", "PLE_TCM_cot", "PLE_Pharmacy_cot"]:
            if "E." in prompts[0] or "(E)" in prompts[0]:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D", "E"]}
            else:
                cot_sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
            cot_max_new_tokens = 1
        else:
            cot_sequence_bias = None
            cot_max_new_tokens = 50

        with torch.inference_mode():
            answer_output_ids = model.generate(
            input_ids,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=cot_max_new_tokens,
            eos_token_id=tokenizer("<|im_end|>")["input_ids"][-1],
            sequence_bias=cot_sequence_bias,
            use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != answer_output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        answer_outputs = tokenizer.batch_decode(answer_output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}" for output, answer_output in zip(outputs, answer_outputs)]


        for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
            ans_file.write(json.dumps({"prompt": question,
                                    "text": output,
                                    "solution": answer,
                                    "additional_info": additional_info,
                                    "model_id": model_name,
                                    "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()
        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--logit-score", default=100.0)
    parser.add_argument("--use_logit_bias", action="store_true", default=False)
    parser.add_argument("--only_load", choices=["attn", "ffn", "share", "no_share", "base", "orth"], default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lora_name_or_path", type=str, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args()

    eval_model(args)