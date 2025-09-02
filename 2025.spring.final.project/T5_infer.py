import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, AutoTokenizer
from pycorrector.t5.t5_corrector import T5Corrector
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher

if __name__ == "__main__":
    model = T5Corrector('models/T5')
    with open("input/ocr_test_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, item in enumerate(data):
        src = item["source_text"]
        char_bounding_box_list = item["char_bounding_box_list"]
        corrected_result = model.correct(src)
        corrected_text = corrected_result['target']
        new_item = dict(item)
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

    with open("output/predict.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Correct Finished.")