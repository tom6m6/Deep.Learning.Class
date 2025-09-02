import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, AutoTokenizer
from pycorrector.t5.t5_corrector import T5Corrector
import numpy as np
from tqdm import tqdm

class T5CorrectorDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=200):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data:
            self.data.append({
                'source_text': item['source_text'],
                'target_text': item['target_text']
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['source_text']
        target_text = item['target_text']
        # T5模型的输入格式为 "correct: [source_text]"
        source_encoding = self.tokenizer(
            f"correct: {source_text}",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # 忽略pad token的损失
        return {
            'input_ids': source_encoding.input_ids.squeeze(),
            'attention_mask': source_encoding.attention_mask.squeeze(),
            'labels': labels
        }

def train_model( train_file, save_path='models/T5',
    batch_size=32,
    epochs=20,
    learning_rate=5e-5,
    max_length=200,
    device='cuda'
):
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    print("device:", device)
    os.makedirs(save_path, exist_ok=True)
    model = T5ForConditionalGeneration.from_pretrained(
        "model_local/T5",
        trust_remote_code=True,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("model_local/T5", trust_remote_code=True)

    train_dataset = T5CorrectorDataset(train_file, tokenizer, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return model, tokenizer

if __name__ == "__main__":
    train_model(
        train_file='input/train_data.json',
        save_path='models/T5',
        batch_size=32,
        epochs=20,
        learning_rate=5e-5,
        max_length=200,
        device='cuda'
    )