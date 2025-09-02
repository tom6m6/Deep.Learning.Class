#!/bin/bash

# 获取脚本所在的目录（即 train 目录）
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 获取 ChineseErrorCorrector 根目录（向上两级）
BASE_PATH=$(dirname "$SCRIPT_DIR")/..


# base model path
BASE_MODEL_PATH="${BASE_PATH}/pre_model/merge1"
# lora weight path
LOARA_PATH="${BASE_PATH}/data/paper_data/model_output_stage2"
# merge model path
OUTPUT_PATH="${BASE_PATH}/pre_model/merge2"


# 运行训练脚本
python ${BASE_PATH}/utils/merge.py \
    --base_model $BASE_MODEL_PATH \
    --lora_path $LOARA_PATH \
    --output_model $OUTPUT_PATH