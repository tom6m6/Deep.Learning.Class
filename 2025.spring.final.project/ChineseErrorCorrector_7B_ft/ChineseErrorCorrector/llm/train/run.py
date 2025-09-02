import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

sys.path.append(base_path)
import argparse
from ChineseErrorCorrector.config import TrainConfig, Qwen2TextCorConfig
from loguru import logger
import torch

from ChineseErrorCorrector.llm.train.train_lora import TrainLLM


def main():
    parser = argparse.ArgumentParser()

    # 数据文件路径
    parser.add_argument('--train_file', default=TrainConfig.TRAIN_PATH, type=str, help='Train file')
    parser.add_argument('--dev_file', default=TrainConfig.DEV_PATH, type=str, help='Dev file')

    # 模型类型
    parser.add_argument('--model_type', default='auto', type=str, help='Transformers model type')

    # 预训练模型路径
    parser.add_argument('--model_name', default=Qwen2TextCorConfig.DEFAULT_CKPT_PATH, type=str, help='LLM path')

    # 训练/预测开关
    parser.add_argument('--do_train', default=True, type=bool, help='Whether to run training.')
    parser.add_argument('--do_predict', default=True, type=bool, help='Whether to run predict.')

    # 输出路径
    parser.add_argument('--output_dir', default=TrainConfig.SAVE_PATH, type=str, help='Model output directory')

    # 设备配置
    parser.add_argument('--device_map', default="auto", type=str)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    # 精度控制
    parser.add_argument('--bf16', default=True, type=bool, help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--int8', default=False, type=bool)
    parser.add_argument('--int4', default=False, type=bool)

    # 训练配置
    parser.add_argument('--num_train_epochs', default=5, type=int, help='train llm epoch')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='llm learning rate')
    parser.add_argument('--logging_steps', default=50, type=int, help='currect % logging_steps==0,print log')
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--per_device_train_batch_size', default=8, type=int)
    parser.add_argument('--gradient_checkpointing', default=True, type=bool)
    parser.add_argument('--torch_compile', default=False, type=bool)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--warmup_steps', default=50, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--eval_steps', default=1000, type=int)
    parser.add_argument('--save_total_limit', default=10, type=int)
    parser.add_argument('--max_eval_samples', default=1000, type=int)
    parser.add_argument('--save_strategy', default='steps', type=str)
    parser.add_argument('--optimizer', default='adamw_torch', type=str)
    parser.add_argument('--remove_unused_columns', default=False, type=bool)
    parser.add_argument('--report_to', default='tensorboard', type=str)
    parser.add_argument('--overwrite_output_dir', default=True, type=bool)

    # PEFT（参数高效微调）相关
    parser.add_argument('--qlora', default=False,type=bool)
    parser.add_argument('--peft_type', default='LORA', type=str)
    parser.add_argument('--use_peft', default=True, type=bool)
    parser.add_argument('--lora_target_modules', default=['all'], type=list)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_bias', default='none', type=str)

    # 其他配置
    parser.add_argument('--no_cache', default=False, type=bool)
    parser.add_argument('--dataset_class', default=None, type=object)
    parser.add_argument('--cache_dir', default=TrainConfig.CACHE_PATH, type=str)
    parser.add_argument('--preprocessing_num_workers', default=4, type=int)
    parser.add_argument('--reprocess_input_data', default=True, type=bool)
    parser.add_argument("--resume_from_checkpoint", default=TrainConfig.SAVE_PATH, type=str)

    # Prompt 模板配置
    parser.add_argument('--prompt_template_name', default='qwen', type=str, help='Prompt template name')
    parser.add_argument('--max_seq_length', default=512, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=512, type=int, help='Output max sequence length')

    # 分布式训练配置
    parser.add_argument("--local_rank", type=int, help="Used by dist launchers")

    # 添加参数解析
    args = parser.parse_args()
    logger.info(args)

    if args.do_train:
        model = TrainLLM(args)
        model.train_model(train_data=args.train_file, output_dir=args.output_dir, eval_data=None) # eval_data=args.dev_file None


if __name__ == '__main__':
    main()
