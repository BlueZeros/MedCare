

# MEDCARE: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation

<p align="center">
  <img src=".\images\overview.png" width=800px/>
</p>

## News
🔥 [2024/09/20] Our paper is accepted by EMNLP 2024 Findings!
* This is the repo for [MEDCARE: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation](https://arxiv.org/pdf/2406.17484v1)

## HighLight
* Knowledge Comprehension
<p align="center">
  <img src=".\images\performance.png" width=800px/>
</p>
* Low Data Dependency
<p align="center">
  <img src=".\images\data-efficient-1.png" width=600px/>
  <img src=".\images\data-efficient-2.png" width=283px/>
</p>
* Superior Fine-tuing performance & Cross-Lingual Generality
<p align="center">
  <img src=".\images\cross-lingual.png" width=800px/>
</p>

## Quick Start
### Installation
```bash
# create new env
conda create -n medcare python==3.10.8
# install dependency
pip install -e ".[train]"
```

### Preparation
* DATA: You can download the fine-tuning and evaluation datas from [here](https://huggingface.co/datasets/BlueZeros/MedCareData) and put it in the folder.

* MODEL: Our methods support Qwen1.5 and Qwen2 Chat Series, You can obtain the models on [Huggingface](https://huggingface.co/).

### Fine-tuning
* First Stage Fine-tuning
```bash
MODEL_PATH={folder-of-models}
DATA_PATH=./MedCareData

BASE_MODEL={model-name}
DATA_NAME=1-stage
SAVE_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}

python -u -m torch.distributed.run \
    --nproc_per_node 4 \
    --nnodes 1 \
    medcare/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --num_experts 8 --num_experts_per_token 2 \
    --share_expert True --num_share_experts 2 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH}/${BASE_MODEL} \
    --train_data_path ${DATA_PATH}/${DATA_NAME}/train.json \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb
```

* Merge Attention Module
```bash
MODEL_PATH={folder-of-models}
DATA_NAME=1-stage
BASE_MODEL={model-name}
SAVE_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}
TAIA_MODEL_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}-taia

srun -p medai_llm --quotatype=auto --gres=gpu:1 python -m medcare.eval.merge_lora_molora_weights \
    --model_base  ${MODEL_BASE}\
    --model_path ${SAVE_PATH} \
    --save_path ${TAIA_PATH_PATH} \
    --only_load attn
```

*  Second Stage Fine-tuning
```bash
# First satage parameters
DATA_NAME=1-stage
BASE_MODEL={model-name}
SAVE_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}
TAIA_MODEL_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}-taia

# Second satage parameters
DATA_PATH=./MedCareData
LORA_PATH=${SAVE_PATH}
DATA_NAME=2-stage
SAVE_PATH=./checkpoints/${DATA_NAME}-${BASE_MODEL}
BASE_MODEL=${TAIA_MODEL_PATH}

python -u -m torch.distributed.run \
    --nproc_per_node $4 \
    --nnodes 1 \
    medcare/train/train_mem.py \
    --lora_enable True --wrap_ffn_lora False --wrap_attn_lora False --lora_r 16 --lora_alpha 32 \
    --use_orthogonal True \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${BASE_MODEL} \
    --train_data_path ${DATA_PATH}/${DATA_NAME}/train.json \
    --bf16 True \
    --output_dir ${SAVE_PATH} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --lora_name_or_path ${LORA_PATH} \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --lamda_1 1 \
    --lamda_2 0.
```

### Evaluation
```bash
LOGS_BASE_PATH=./log
DATA_PATH=./MedCareData
BASE_MODEL={model-name}

DATASET={filename-wo-suffix-in-data-test-folder}
MODEL_BASE=./checkpoints/1-stage-${BASE_MODEL}-taia
MODEL_PATH=./checkpoints/2-stage-${BASE_MODEL}
LORA_PATH=./checkpoints/1-stage-${BASE_MODEL}
CKPT=2-stage-${BASE_MODEL}

echo "Processing ${DATASET}"
python -m medcare.eval.model_diverse_gen_batch \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file ${DATA_PATH}/test/${DATASET}.json \
    --answers-file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --temperature 0 \
    --conv-mode qwen \
    --resume \
    --lora_name_or_path ${LORA_PATH}

echo "Evaluating ${DATASET}"
python -m medcare.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/infer.jsonl \
    --output_file ${LOGS_BASE_PATH}/${CKPT}/${DATASET}/wrong.jsonl
```

## Citation
If you find MedCare useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{liao2024medcare,
  title={MedCare: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation},
  author={Liao, Yusheng and Jiang, Shuyang and Wang, Yanfeng and Wang, Yu},
  journal={arXiv preprint arXiv:2406.17484},
  year={2024}
}
```