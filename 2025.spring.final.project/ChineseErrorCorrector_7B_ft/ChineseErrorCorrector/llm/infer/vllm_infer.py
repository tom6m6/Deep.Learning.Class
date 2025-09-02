import torch
import uuid

from ChineseErrorCorrector.config import DEVICE, DEVICE_COUNT, Qwen2TextCorConfig
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel, \
    AutoModelForCausalLM, set_seed
from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
import time
import copy

from ChineseErrorCorrector.utils.correct_tools import torch_gc



class VLLMTextCorrectInfer(object):

    def __init__(self, ):
        set_seed(42)
        self.prompt_prefix = "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为："

        if DEVICE == 'cpu':
            pass
        else:
            device = torch.device(DEVICE)
            capability = torch.cuda.get_device_capability(device)
            # T4 算力为7.5 无法使用BF16，改为float16
            if capability[0] < 8:
                model_args = AsyncEngineArgs(Qwen2TextCorConfig.DEFAULT_CKPT_PATH,
                                             tensor_parallel_size=DEVICE_COUNT,
                                             dtype='float16',
                                             trust_remote_code=True
                                             , gpu_memory_utilization=Qwen2TextCorConfig.GPU_MEMARY,
                                             max_model_len=Qwen2TextCorConfig.MAX_LENGTH)
            else:
                model_args = AsyncEngineArgs(Qwen2TextCorConfig.DEFAULT_CKPT_PATH,
                                             tensor_parallel_size=DEVICE_COUNT,
                                             trust_remote_code=True
                                             , gpu_memory_utilization=Qwen2TextCorConfig.GPU_MEMARY,
                                             max_model_len=Qwen2TextCorConfig.MAX_LENGTH)
            self.model = AsyncLLMEngine.from_engine_args(model_args)
            self.tokenizer = AutoTokenizer.from_pretrained(Qwen2TextCorConfig.DEFAULT_CKPT_PATH, trust_remote_code=True)

    async def generate(self, query):
        """
        非流式输出
        :param query:
        :return:
        """
        try:
            start_time = time.time()
            request_id = str(uuid.uuid4())
            messages = [
                {"role": "user", "content": self.prompt_prefix + query}
            ]

            # prompt中的assistant 改为 system

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            )
            prompt_token_ids = inputs['input_ids'].tolist()[0]

            sampling = SamplingParams(
                use_beam_search=False,
                seed=42,
                max_tokens=Qwen2TextCorConfig.MAX_LENGTH
            )
            inputs = {'prompt': query, "prompt_token_ids": prompt_token_ids}
            generator = self.model.generate(
                **inputs,
                sampling_params=sampling,
                request_id=request_id
            )

            result_list = []
            result_list_tokens = []

            async for request_i in generator:
                result_list.append(str(request_i.outputs[0].text))
                result_list_tokens.append(request_i.outputs[0].token_ids)

            print(f'Token的生成速度：{len(result_list_tokens[-1]) / (time.time() - start_time)}')

            return str(result_list[-1])
        except Exception as err:
            print(err)
            if 'CUDA out of memory' in err.args[0]:
                raise MemoryError(err)

        if DEVICE != "cpu":
            torch_gc()

    async def infer(self, req_json):

        # 大模型的纠错推理
        prompts, querys = await self.parse_request(req_json)

        result = [await self.generate(query_i) for query_i in prompts]

        return result

    async def infer_sentence(self, user_inputs):
        result = [await self.generate(query_i) for query_i in user_inputs]
        return result

    async def parse_request(self, req_json):
        """
                {
          "contents": [
                "少先队员因该为老人让坐。",
                "我的明子叫小明"
          ]

        }
        :param req_json:
        :return:
        """

        req_json_copy = copy.deepcopy(req_json)

        query_list = req_json_copy.get('prompt')

        prompt = []
        # 增加文本纠错的prompt
        for pmt_i in query_list:
            prompt_sys = {}
            prompt_user = {}
            prompt_user['role'] = "user"
            prompt_user['content'] = self.prompt_prefix + pmt_i

            prompt.append([prompt_sys, prompt_user])

        return prompt, query_list


if __name__ == '__main__':
    pass
