import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import json
import re

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

import sys
sys.path.append('..')
from eval_utils import get_parser, compute_test_metrics
from utils import Format, get_to_string_processor


with open("configs/config_fredT5-xl-lt.json", "rb") as config:
    params = json.load(config)
print(params)

ckpt = params["save_folder"]
print(ckpt)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

out_format = Format(params["format"])
print(out_format)

data = pd.read_csv(f"~/leonya/bench_results/{ckpt}_preds.csv")
print(data.head())

model_checkpoint = f"../good_checkpoints/{ckpt}"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

parser = get_parser(tokenizer, out_format)
to_string_processor = get_to_string_processor(out_format)

responses = []
keys_set = set()

for ind in tqdm(data.index, total=len(data)):
    if ind % 100 == 0:
        print(ind)
    preds = [data.loc[ind, "Responses"]]
    labels = [to_string_processor(data.loc[ind, "bundles"]) + tokenizer.eos_token]
    
    is_valid, bundles = parser(re.sub(r'(</s>)+', '</s>', preds[0]))
    responses.append(compute_test_metrics(preds, labels, parser))
    responses[-1]["pred_bundles"] = str(bundles) if is_valid else None
    for key in responses[-1]:
        keys_set.add(key)

metrics = {key: [] for key in keys_set}
for resp in responses:
    for key in metrics:
        metrics[key].append(resp.get(key))

data = pd.concat([data, pd.DataFrame(metrics)], axis=1)
data.to_csv(f"~/leonya/bench_results/{ckpt}_metrics.csv", index=False)