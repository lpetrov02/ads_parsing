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
from eval_utils import get_parser, deduplicate, is_valid_bundle
from formats import Format, get_to_string_processor, get_parser


with open("configs/config_fredT5-xl-lt-ratio2.json", "rb") as config:
    params = json.load(config)
print(params)

ckpt = params["save_folder"]

out_format = Format(params["format"])
print(out_format)

data_path = "~/data/data_for_distil.csv"
data = pd.read_csv(data_path)
data["sequence"] = ""
data["n_bundles"] = 1

print(f"\nDataset of shape {data.shape}")
print(data.head())
print("\n")

model_checkpoint = f"../../good_checkpoints/{ckpt}"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

parser = get_parser(tokenizer, out_format)
to_string_processor = get_to_string_processor(out_format)

responses = []

for ind in tqdm(data.index, total=len(data)):
    if ind % 1000 == 0:
        print(ind)
    preds = [data.loc[ind, "response"]]
    
    is_valid, bundles = parser(re.sub(r'(</s>)+', '</s>', preds[0]))
    if is_valid and not all(is_valid_bundle(bundle) for bundle in bundles):
        is_valid = False

    if is_valid:
        bundles, _ = deduplicate(bundles)
        bundles = [{key.capitalize(): value for key, value in bundle.items()} for bundle in bundles]
    data.loc[ind, "sequence"] = json.dumps(bundles) if is_valid else np.nan
    data.loc[ind, "n_bundles"] = len(bundles) if is_valid else np.nan

data.to_csv(data_path, index=False)

print(data.isna().sum())
print(data.head())
print(data.shape)

print("Filtering...")
test_data = pd.concat([
    pd.read_csv("~/data/ads_test_100.csv"),
    pd.read_csv("~/data/ads_test_1000.csv"),
    pd.read_csv("~/data/ads_eval.csv"),
])
test_texts = set(test_data["Text"].values.tolist())
data = data[data["text"].apply(lambda x: x not in test_texts)]
print(data.shape)
data.drop_duplicates(subset=["text"]).to_csv(data_path, index=False)
