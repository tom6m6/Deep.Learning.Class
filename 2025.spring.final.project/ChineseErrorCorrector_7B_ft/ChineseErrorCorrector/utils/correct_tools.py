import random
import numpy as np
import torch
import difflib
from ChineseErrorCorrector.config import DEVICE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)


def torch_gc():
    """ Clear GPU cache for multiple devices """
    if DEVICE != "cpu":
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def res_format(sentences, result):
    """
    格式化返回结果
    :param sentences:
    :param result:
    :return:
    """
    start_idx = 0
    n = len(sentences)
    data = []
    for i in range(n):
        a, b = sentences[i], result[i]
        if len(a) == 0 or len(b) == 0 or a == "\n":
            start_idx += len(a)
            return
        s = difflib.SequenceMatcher(None, a, b)
        errors = []
        offset = 0
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag != "equal":
                e = [a[i1:i2], b[j1 + offset:j2 + offset], i1]
                # if ignore_function and ignore_function(e):
                #     # 因为不认为是错误， 所以改回原来的偏移值
                #     b = b[:j1] + a[i1:i2] + b[j2:]
                #     offset += i2 - i1 - j2 + j1
                #     continue

                errors.append(tuple(e))
        data.append({"source": a, "target": b, "errors": errors})
    return data
