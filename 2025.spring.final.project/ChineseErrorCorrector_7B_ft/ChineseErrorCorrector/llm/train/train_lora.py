# -*- coding: utf-8 -*-
import math
import os
import torch
from transformers import (
    AutoConfig,
    BloomTokenizerFast,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TextIteratorStreamer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers.trainer import TRAINING_ARGS_NAME

from ChineseErrorCorrector.utils.llm_dataloader import GptSupervisedDataset, IGNORE_INDEX
from loguru import logger

MODEL_CLASSES = {
    "llama": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


class TrainLLM:
    def __init__(self, args):

        """
        Initializes a GptModel model.

        Args:
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
        """
        self.args = args
        model_type = args.model_type

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp = self.world_size != 1
        self.results = {}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.config = config_class.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        self.torch_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        self.model = model_class.from_pretrained(
            args.model_name,
            config=self.config,
            torch_dtype=self.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.int4,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            ) if args.qlora else None,
        )

        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name, trust_remote_code=True)
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "</s>"  # eos token is required for SFT
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.architectures[0] == "Qwen2ForCausalLM":
            self.tokenizer.padding_side = "left"

    def find_all_linear_names(self, int4=False, int8=False):
        cls = torch.nn.Linear
        if int4 or int8:
            import bitsandbytes as bnb
            if int4:
                cls = bnb.nn.Linear4bit
            elif int8:
                cls = bnb.nn.Linear8bitLt
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                # last layer is not add to lora_module_names
                if 'lm_head' in name:
                    continue
                if 'output_layer' in name:
                    continue
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)

    def load_and_cache_examples(
            self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a LlamaDataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = self.tokenizer
        args = self.args
        mode = "dev" if evaluate else "train"
        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(tokenizer, args, data, mode)
        else:
            return GptSupervisedDataset(tokenizer, args, data, mode)

    def save_model(
            self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
    ):
        """Save the model and the tokenizer."""
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model:
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

    def train_model(
            self,
            train_data,
            output_dir=None,
            eval_data=None
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: json file path or Pandas DataFrame containing 1 columns - `conversations`.
                format: {"conversations": [{"from": "human", "value": "你是一个文本纠错专家，纠正输入句子中的语法错误，并输出正确的句子，输入句子为：这是不是太不公平!"}, {"from": "gpt", "value": "这是不是太不公平？"}]}
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            eval_data (optional): A DataFrame against which evaluation will be performed. If it is not passed, evaluation will be skipped.
        Returns:
            global_step: Number of global steps trained
            training_details: Training progress scores 
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir
        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        # Setup train args
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.args.logging_steps,
            max_steps=self.args.max_steps,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_train_batch_size,
            gradient_checkpointing=self.args.gradient_checkpointing,
            torch_compile=self.args.torch_compile,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            optim=self.args.optimizer,
            save_strategy=self.args.save_strategy,
            # evaluation_strategy='steps' if eval_data is not None else 'no',
            eval_steps=self.args.eval_steps if eval_data is not None else None,
            load_best_model_at_end=True if eval_data is not None else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            save_total_limit=self.args.save_total_limit,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            remove_unused_columns=self.args.remove_unused_columns,
            report_to=self.args.report_to,
            overwrite_output_dir=self.args.overwrite_output_dir,
            no_cuda=True if self.args.device == "cpu" else False
        )

        if 'all' in self.args.lora_target_modules:
            self.args.lora_target_modules = self.find_all_linear_names(self.args.int4, self.args.int8)
        # setup peft
        if self.args.use_peft:
            if self.args.int8 or self.args.int4:
                self.model = prepare_model_for_kbit_training(self.model, self.args.gradient_checkpointing)

            peft_type = self.args.peft_type.upper()
            # add peft config
            if peft_type == 'LORA':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )
            elif peft_type == 'ADALORA':
                from peft import AdaLoraConfig
                peft_config = AdaLoraConfig(
                    init_r=self.args.adalora_init_r,
                    r=self.args.lora_r,
                    beta1=self.args.lora_beta,
                    beta2=self.args.lora_beta,
                    tinit=self.args.adalora_tinit,
                    tfinal=self.args.adalora_tfinal,
                    deltaT=self.args.adalora_delta_t,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                )
            elif peft_type == 'PROMPT_TUNING':
                from peft import PromptTuningConfig

                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                )
            elif peft_type == 'P_TUNING':
                from peft import PromptEncoderConfig

                peft_config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size
                )
            elif peft_type == 'PREFIX_TUNING':
                from peft import PrefixTuningConfig

                peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.args.num_virtual_tokens,
                    encoder_hidden_size=self.args.prompt_encoder_hidden_size,
                    prefix_projection=True,
                )
                self.model.gradient_checkpointing_disable()
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    bias=self.args.lora_bias,
                )

            if isinstance(self.model, PeftModel):
                self.model = self.model.merge_and_unload()
            self.model = get_peft_model(self.model, peft_config)
            # Set data type to float32
            for param in filter(lambda p: p.requires_grad, self.model.parameters()):
                param.data = param.data.to(torch.float32)

            self.model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        else:
            self.model = self.model.float()
        os.makedirs(output_dir, exist_ok=True)

        # load dataset
        train_dataset = self.load_and_cache_examples(train_data)

        eval_dataset = None
        if eval_data is not None:
            eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        # Update model train config
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model.config.use_cache = True
        self.model.enable_input_require_grads()
        if not self.ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        # Initialize our Trainer
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX)
        trainer = SavePeftModelTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_data is not None else None,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Training
        logger.info("*** Training ***")
        # 断点继续训练

        (global_step, training_loss, metrics) = trainer.train()
        self.model.config.use_cache = True  # enable cache after training
        # trainer.save_state()
        self.save_model(model=self.model)

        if eval_data is not None:
            logger.info("*** Evaluate ***")
            if self.args.fp16:
                self.model.half()
            metrics = trainer.evaluate(metric_key_prefix="eval")
            metrics['eval_samples'] = len(eval_dataset)
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity
            self.results.update(metrics)

        return global_step, training_loss


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
