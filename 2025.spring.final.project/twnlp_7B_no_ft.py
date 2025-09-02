from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import snapshot_download
import os
import json
import numpy as np
import re
from difflib import SequenceMatcher
from tqdm import tqdm
set_seed(42)

model_path = "model_local/TWNLP-7B"  
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')

prompt = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："

with open("input/ocr_test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for num, item in enumerate(data):
    char_bounding_box_list = item["char_bounding_box_list"]
    src = item["source_text"]
    new_item = dict(item)
    sentences = re.split(r'(?<=[。？?！!])', src)
    sentences = [s for s in sentences if s]
    corrected_text = ""
    for s in sentences:
        messages = [
            {
                "role": "user", 
                "content": prompt + s
            }
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        corrected_text += response.strip()
    
    new_item["predict_text"] = corrected_text
    matcher = SequenceMatcher(None, src, corrected_text)
    bounding_box_list = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            for i in range(i1, i2):
                if i < len(char_bounding_box_list):
                    box = char_bounding_box_list[i]["box"]
                    bounding_box_list.append(box)
        
    new_item["bounding_box_list"] = bounding_box_list
    results.append(new_item)
    if (num+1) % 5 == 0:
        print(f"{num+1} finished.")


with open("output/predict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Correct Finished.")