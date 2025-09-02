import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from peft import PeftModel
import torch
import argparse


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据文件路径
    parser.add_argument('--base_model', type=str, help='base train model')
    parser.add_argument('--output_model', type=str, help='merge lora output model')
    parser.add_argument('--lora_path', type=str, help='lora weight')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    apply_lora(args.base_model, args.output_model, args.lora_path)
