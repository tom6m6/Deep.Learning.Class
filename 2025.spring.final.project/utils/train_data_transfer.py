import json
# 为了微调twnlp-7B, 需要将数据集的格式进行一些转化
with open("input/train_data.json", "r", encoding="utf-8") as f:
    source_data = json.load(f)

converted_data = []
for item in source_data:
    converted_data.append({
        "conversations":[
            {
                "from": "human",
                "value": item["source_text"]
            },
            {
                "from": "gpt",
                "value": item["target_text"]
            }
        ]
    })


with open("input/converted_train_data_list.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

with open("input/converted_train_data_list.json", "r", encoding="utf-8") as f:
    list_data = json.load(f)

with open("input/converted_train_data.json", "w", encoding="utf-8") as f:
    for item in list_data:
        # 将每个对象单独写入一行
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"transfer {len(source_data)} finished")