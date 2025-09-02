from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import snapshot_download

set_seed(42)
snapshot_download(
    repo_id = "shibing624/mengzi-t5-base-chinese-correction",
    cache_dir = None,
    local_dir = "model_local/T5",
    force_download = False,
    ignore_patterns=["*.ckpt", "*.bin"]
)

snapshot_download(
    repo_id = "twnlp/ChineseErrorCorrector2-7B",
    cache_dir=None,
    local_dir="model_local/TWNLP-7B",
    force_download = False,
    ignore_patterns = ["*.ckpt", "*.bin"]
)

