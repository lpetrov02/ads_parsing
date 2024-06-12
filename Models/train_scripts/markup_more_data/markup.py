import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from tqdm.notebook import tqdm

import json
import re

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

import os
import sys
sys.path.append('..')
from eval_utils import compute_test_metrics
from utils import clean_text
from formats import Format, get_to_string_processor, get_parser


with open("configs/config_fredT5-xl-lt-ratio2.json", "rb") as config:
    params = json.load(config)
print(params)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

data_path = "~/data/merged_final.csv"
result_path = "~/data/data_for_distil.csv"

out_format = Format(params["format"])
print(out_format)

data = pd.read_csv(data_path)
print(data.shape)


print("Loading model...")
model_checkpoint = f"../../good_checkpoints/{params['save_folder']}"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


parser = get_parser(tokenizer, out_format)
to_string_processor = get_to_string_processor(out_format)


model.to(device)
ckpt_frequency = 25
bs = 8

postfix = (tokenizer.eos_token if params.get("add_eos_token", False) else "")
prefix = ("<LM>" if params.get("add_lm_token", False) else "")


result = {
    "text": [],
    "response": [],
}

if os.path.exists(result_path):
    res_data = pd.read_csv(result_path)
    result["text"] = res_data["text"].tolist()
    result["response"] = res_data["response"].tolist()

print("Total steps:", (len(data) - len(result["text"]) + bs - 1) // bs)
for ind in tqdm(range(len(result["text"]), len(data), bs), total=(len(data) + bs - 1) // bs):
    if (ind // bs) % 10 == 0:
        print(f"{ind // bs} steps made")
    result["text"] += data.loc[data.index[ind: min(ind + bs, len(data))], "Text"].values.tolist()
    postfix = tokenizer.eos_token if params.get("add_eos_token", False) else ""
    prefix = "<LM>" if params.get("add_lm_token", False) else ""
    cleaner = clean_text if params.get("clean_text", False) else (lambda x: x)
    batch = tokenizer(
        [prefix + cleaner(data.loc[data.index[i], "Text"]) + postfix for i in range(ind, min(ind + bs, len(data)))],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    preds = model.generate(
        input_ids=batch.to(device),
        max_length=512,
        num_beams=4,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    ).cpu()
    
    preds = torch.where(preds == -100, tokenizer.eos_token_id, preds)
    preds = tokenizer.batch_decode(preds, ignore_special_tokens=True)
    result["response"] += [re.sub(tokenizer.pad_token, "", pred) for pred in preds]

    if (ind // bs) % ckpt_frequency == 0:
        pd.DataFrame(result).to_csv(result_path, index=False)
        print(f"{len(result['text'])} ads marked-up!")
