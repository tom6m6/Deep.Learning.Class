#!/bin/bash

# 获取脚本所在的目录（即 train 目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 获取 ChineseErrorCorrector 根目录（向上两级）
BASE_PATH=$(dirname "$SCRIPT_DIR")/..


# base model path
BASE_MODEL_PATH="${BASE_PATH}/pre_model/Qwen2.5-7B-Instruct"
# lora weight path
LOARA_PATH="${BASE_PATH}/data/paper_data/model_output_stage1"
# merge model path
OUTPUT_PATH="${BASE_PATH}/pre_model/merge1"


# 运行训练脚本
python ${BASE_PATH}/utils/merge.py \
    --base_model $BASE_MODEL_PATH \
    --lora_path $LOARA_PATH \
    --output_model $OUTPUT_PATH