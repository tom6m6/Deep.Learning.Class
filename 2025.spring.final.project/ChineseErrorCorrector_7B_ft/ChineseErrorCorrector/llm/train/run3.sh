#!/bin/bash

# 获取脚本所在的目录（即 train 目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 获取 ChineseErrorCorrector 根目录（向上两级）
BASE_PATH=$(dirname "$SCRIPT_DIR")/..


# NaCGEC
TRAIN_FILE="${BASE_PATH}/data/paper_data/train_nacgec.json"
# NaCGEC数据集
DEV_FILE="${BASE_PATH}/data/paper_data/test_nacgec.json"
# 训练模型路径
MODEL_NAME="${BASE_PATH}/pre_model/merge2"
# LoRA 训练的存储目录
OUTPUT_DIR="${BASE_PATH}/data/paper_data/model_output_stage3"
# 数据 cache 目录
CACHE_DIR="${BASE_PATH}/data/paper_data/cache_dir_3"

# 运行训练脚本
python ${BASE_PATH}/llm/train/run.py \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --model_type auto \
    --model_name $MODEL_NAME \
    --do_train True \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --device cuda \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --logging_steps 50 \
    --max_steps -1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2\
    --warmup_steps 1000 \
    --save_steps 100000 \
    --optimizer adamw_torch \
    --save_strategy steps \
    --eval_steps 100000 \
    --save_total_limit 10 \
    --report_to tensorboard \
    --overwrite_output_dir True \
    --max_eval_samples 1000 \
    --peft_type LORA \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --cache_dir $CACHE_DIR \
    --preprocessing_num_workers 4 \
    --resume_from_checkpoint $OUTPUT_DIR \
    --prompt_template_name qwen \
    --max_seq_length 512 \
    --max_length 512
