import torch

from ChineseErrorCorrector.config import DEVICE, DEVICE_COUNT, Qwen2TextCorConfig
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel, \
    AutoModelForCausalLM, set_seed


class HFTextCorrectInfer(object):

    def __init__(self, ):
        set_seed(42)
        self.prompt_prefix = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："
        if DEVICE == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(
                Qwen2TextCorConfig.DEFAULT_CKPT_PATH,
                # cpu不支持 float16,改用bfloat16
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                Qwen2TextCorConfig.DEFAULT_CKPT_PATH,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE
            )
        self.tokenizer = AutoTokenizer.from_pretrained(Qwen2TextCorConfig.DEFAULT_CKPT_PATH, trust_remote_code=True,
                                                       padding_side='left')

    def infer(self, input_list):
        all_outputs = []

        # 批量处理整个batch
        messages = [
            [{"role": "user", "content": self.prompt_prefix + error}]

            for error in input_list
        ]

        # 批量编码所有输入
        input_texts = [
            self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        inputs = self.tokenizer.batch_encode_plus(
            input_texts,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # 批量生成输出
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024
        )

        # 批量解码输出
        for i, output in enumerate(outputs):
            prompt_len = len(inputs.input_ids[i])
            gen_text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            gen_text = gen_text.strip()
            all_outputs.append(gen_text)
        return all_outputs


if __name__ == '__main__':
    pass
