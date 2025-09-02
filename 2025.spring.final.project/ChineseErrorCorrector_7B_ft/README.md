# 中文拼写和语法纠错

[**🇨🇳中文**](https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/README.md)

<div align="center">
  <a href="https://github.com/TW-NLP/ChineseErrorCorrector">
    <img src="images/image_fx_.jpg" alt="Logo" height="156">
  </a>
</div>



-----------------

## 介绍

支持中文拼写和语法错误纠正，并开源拼写和语法错误的增强工具、大模型训练代码。荣获2024CCL 冠军
🏆，[查看论文](https://aclanthology.org/2024.ccl-3.31/) ，[2023 NLPCC-NaCGEC纠错冠军🏆](https://github.com/TW-NLP/ChineseErrorCorrector?tab=readme-ov-file#nacgec-%E6%95%B0%E6%8D%AE%E9%9B%86)， [2022 FCGEC 纠错冠军🏆](https://github.com/TW-NLP/ChineseErrorCorrector?tab=readme-ov-file#fcgec-%E6%95%B0%E6%8D%AE%E9%9B%86)
，如有帮助，感谢star✨。

## 🔥🔥🔥 新闻

[2025/04/28] 根据[建议](https://github.com/TW-NLP/ChineseErrorCorrector/issues/17)
，我们重新训练纠错模型，并完全开源训练步骤，支持结果复现，[复现教程](https://github.com/TW-NLP/ChineseErrorCorrector/tree/main?tab=readme-ov-file#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E5%A4%8D%E7%8E%B0)

[2025/03/17]
更新批量错误文本的解析，[transformers批量解析](https://github.com/TW-NLP/ChineseErrorCorrector?tab=readme-ov-file#transformers-%E6%89%B9%E9%87%8F%E6%8E%A8%E7%90%86) ;[VLLM批量解析](https://github.com/TW-NLP/ChineseErrorCorrector?tab=readme-ov-file#vllm-%E5%BC%82%E6%AD%A5%E6%89%B9%E9%87%8F%E6%8E%A8%E7%90%86)

[2025/03/10] 模型支持多种推理方式，包括 transformers、VLLM、modelscope。

[2025/02/25]
🎉🎉🎉使用200万纠错数据进行多轮迭代训练，发布了[twnlp/ChineseErrorCorrector2-7B](https://huggingface.co/twnlp/ChineseErrorCorrector2-7B)
，在 [NaCGEC-2023NLPCC官方评测数据集](https://github.com/masr2000/NaCGEC)
上，超越第一名华为10个点，遥遥领先，推荐使用✨✨， [技术详情](https://blog.csdn.net/qq_43765734/article/details/145858955)

[2025/02]
为方便部署，使用38万开源拼写数据，发布了[twnlp/ChineseErrorCorrector-1.5B](https://huggingface.co/twnlp/ChineseErrorCorrector-1.5B)

[2025/01]
使用38万开源拼写数据，基于Qwen2.5训练中文拼写纠错模型，支持语似、形似等错误纠正，发布了[twnlp/ChineseErrorCorrector-7B](https://huggingface.co/twnlp/ChineseErrorCorrector-7B)，[twnlp/ChineseErrorCorrector-32B-LORA](https://huggingface.co/twnlp/ChineseErrorCorrector-32B-LORA/tree/main)

[2024/06]
v0.1.0版本：🎉🎉🎉开源一键语法错误增强工具，该工具可以进行14种语法错误的增强，不同行业可以根据自己的数据进行错误替换，来训练自己的语法和拼写模型。详见[Tag-v0.1.0](https://github.com/TW-NLP/ChineseErrorCorrector/tree/0.1.0)

## 模型列表

| 模型名称                                                                                        | 纠错类型  | 描述                                        |
|:--------------------------------------------------------------------------------------------|:------|:------------------------------------------|
| [twnlp/ChineseErrorCorrector2-7B](https://huggingface.co/twnlp/ChineseErrorCorrector2-7B)   | 语法+拼写 | 使用200万纠错数据进行多轮迭代训练，适用于语法纠错和拼写纠错，效果好，推荐使用。 |
| [twnlp/ChineseErrorCorrector-7B](https://huggingface.co/twnlp/ChineseErrorCorrector-7B)     | 拼写    | 使用38万开源拼写数据，支持语似、形似等拼写错误纠正，拼写纠错效果好。       |
| [twnlp/ChineseErrorCorrector-1.5B](https://huggingface.co/twnlp/ChineseErrorCorrector-1.5B) | 拼写    | 使用38万开源拼写数据，支持语似、形似等拼写错误纠正，拼写纠错效果一般。      |

## 数据集

| 数据集名称                        | 数据链接                                                                                             | 数据量和类别说明                                                                 | 描述                              |
|:-----------------------------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|:--------------------------------|
| ChinseseErrorCorrectData     | [twnlp/ChinseseErrorCorrectData](https://huggingface.co/datasets/twnlp/ChinseseErrorCorrectData) | 200万                                                                     | ChineseErrorCorrector2-7B 训练数据集 |
| CSC（拼写纠错数据集）                 | [twnlp/csc_data](https://huggingface.co/datasets/twnlp/csc_data)                                 | W271K(279,816) Medical(39,303) Lemon(22,259) ECSpell(6,688) CSCD(35,001) | 中文拼写纠错的数据集                      |
| CGC（语法纠错数据集）                 | [twnlp/cgc_data](https://huggingface.co/datasets/twnlp/cgc_data)                                 | CGED(20,449) FCGEC(37,354) MuCGEC(2,467) NaSGEC(7,568)                   | 中文语法纠错的数据集                      |
| Lang8+HSK（百万语料-拼写和语法错误混合数据集） | [twnlp/lang8_hsk](https://huggingface.co/datasets/twnlp/lang8_hsk)                               | 1,568,885                                                                | 中文拼写和语法数据集                      |

## 拼写纠错评测

- 评估指标：F1

| Model Name                           | Model Link                                                                           | Base Model                 | Avg   | SIGHAN-2015(通用) | EC-LAW(法律) | EC-MED(医疗) | EC-ODW(公文) |
|:-------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------|:------|:----------------|:-----------|:-----------|:-----------|
| twnlp/ChineseErrorCorrector-1.5B     | [huggingface](https://huggingface.co/twnlp/ChineseErrorCorrector-1.5B/tree/main)     | Qwen/Qwen2.5-1.5B-Instruct | 0.459 | 0.346           | 0.517      | 0.433      | 0.540      |
| twnlp/ChineseErrorCorrector-7B       | [huggingface](https://huggingface.co/twnlp/ChineseErrorCorrector-7B/tree/main)       | Qwen/Qwen2.5-7B-Instruct   | 0.712 | 0.592           | 0.787      | 0.677      | 0.793      |
| twnlp/ChineseErrorCorrector-32B-LORA | [huggingface](https://huggingface.co/twnlp/ChineseErrorCorrector-32B-LORA/tree/main) | Qwen/Qwen2.5-32B-Instruct  | 0.757 | 0.594           | 0.776      | 0.794      | 0.864      |

## 文本纠错评测(双冠军 🏆)

### NaCGEC 数据集

- 评估工具：ChERRANT  [评测工具](https://github.com/HillZhang1999/MuCGEC)
- 评估数据：[NaCGEC](https://github.com/masr2000/NaCGEC)
- 评估指标：F1-0.5

🏆
| Model Name | Model Link | Prec | Rec | F0.5 |
|:-----------------|:---------------------------------------------------------------|:-----------|:------------|:-------|
| twnlp/ChineseErrorCorrector2-7B | [huggingface](https://huggingface.co/twnlp/ChineseErrorCorrector2-7B) ； [modelspose(国内下载)](https://www.modelscope.cn/models/tiannlp/ChineseErrorCorrector2-7B) | 0.5686 | 0.57 | 0.5689 |
| HW_TSC_nlpcc2023_cgec(华为) | 未开源 | 0.5095 | 0.3129 | 0.4526 |
| 鱼饼啾啾Plus(北京大学) | 未开源 | 0.5708 | 0.1294 | 0.3394 |
| CUHK_SU(香港中文大学) | 未开源 | 0.3882 | 0.1558 | 0.2990 |

### FCGEC 数据集

- 评估指标：binary_f1

[评测🏆](https://codalab.lisn.upsaclay.fr/competitions/8020#results)

## 使用

### 🤗 transformers

```shell
pip install transformers
```

```shell
from transformers import AutoModelForCausalLM, AutoTokenizer,set_seed
set_seed(42)

model_name = "twnlp/ChineseErrorCorrector2-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

prompt = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："
text_input = "对待每一项工作都要一丝不够。"
messages = [
    {"role": "user", "content": prompt + text_input}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```

### VLLM

```shell
pip install transformers
pip install vllm==0.3.3
```

```shell
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("twnlp/ChineseErrorCorrector2-7B")

# Pass the default decoding hyperparameters of twnlp/ChineseErrorCorrector2-7B
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(seed=42,max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="twnlp/ChineseErrorCorrector2-7B")

# Prepare your prompts
text_input = "对待每一项工作都要一丝不够。"
messages = [
    {"role": "user", "content": "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："+text_input}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}") 
```

### VLLM 异步批量推理

- Clone the repo

``` sh
git clone https://github.com/TW-NLP/ChineseErrorCorrector
cd ChineseErrorCorrector
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n zh_correct -y python=3.10
conda activate zh_correct
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

```sh
# 修改config.py
#（1）根据不同的模型，修改的DEFAULT_CKPT_PATH，默认为ChineseErrorCorrector2-7B(将模型下载，放在ChineseErrorCorrector/pre_model/ChineseErrorCorrector2-7B)
#（2）将Qwen2TextCorConfig的USE_VLLM = True

#批量预测
python main.py
```

### Transformers 批量推理

- Clone the repo

``` sh
git clone https://github.com/TW-NLP/ChineseErrorCorrector
cd ChineseErrorCorrector
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n zh_correct -y python=3.10
conda activate zh_correct
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

``` sh
# 修改config.py
#（1）根据不同的模型，修改的DEFAULT_CKPT_PATH，默认为ChineseErrorCorrector2-7B
#（2）将Qwen2TextCorConfig的USE_VLLM = False

#批量预测
python main.py

#输出：
'''
[{'source': '对待每一项工作都要一丝不够。', 'target': '对待每一项工作都要一丝不苟。', 'errors': [('够', '苟', 12)]}, {'source': '大约半个小时左右', 'target': '大约半个小时', 'errors': [('左右', '', 6)]}]
'''

```

### 🤖 modelscope

```shell
pip install modelscope
```

```shell
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "tiannlp/ChineseErrorCorrector2-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："
text_input = "对待每一项工作都要一丝不够。"
messages = [
    {"role": "user", "content": prompt + text_input}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```

## 实验结果复现

### 环境准备

- Clone the repo

``` sh
git clone https://github.com/TW-NLP/ChineseErrorCorrector
cd ChineseErrorCorrector
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n zh_correct -y python=3.10
conda activate zh_correct
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 数据和模型的准备

1、下载训练数据集：[twnlp/ChinseseErrorCorrectData](https://huggingface.co/datasets/twnlp/ChinseseErrorCorrectData) ,放在
`/data/paper_data` 中。

2、下载Qwen2.5-7B-Instruct：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) ,放在`/pre_model`中

### 模型训练与合并

``` sh

# Lang8+HSK 训练
bash ./llm/train/run1.sh
bash ./llm/train/merge1.sh

# CGC+CSC 数据集训练
bash ./llm/train/run2.sh
bash ./llm/train/merge2.sh

# Nacgec 数据集训练
bash ./llm/train/run3.sh
bash ./llm/train/merge3.sh
``` 

## Citation

If this work is helpful, please kindly cite as:

```bibtex

@inproceedings{wei2024中小学作文语法错误检测,
  title={中小学作文语法错误检测, 病句改写与流畅性评级的自动化方法研究},
  author={Wei, Tian},
  booktitle={Proceedings of the 23rd Chinese National Conference on Computational Linguistics (Volume 3: Evaluations)},
  pages={278--284},
  year={2024}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=TW-NLP/ChineseErrorCorrector&type=Date)
