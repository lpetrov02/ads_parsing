import numpy as np
import torch
import re
import json
from enum import Enum

from transformers import TrainerCallback, DataCollatorForSeq2Seq

import nltk
nltk.download('punkt')
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer

from formats import get_to_string_processor


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        
def clean_text(text):
    punctuation = "#+_=<>\\"
    words = wordpunct_tokenize(text.lower())
    words = [re.sub(r'[.!?]+', '.', word.strip(punctuation)) for word in words]
    return " ".join([word for word in words if len(word) > 0])


def get_preprocessor(tokenizer, out_format, do_clean=False):
    def preprocess_function(examples):
        cleaner = clean_text if do_clean else (lambda x: x)
        inputs = [cleaner(text) for text in examples["Text"]]
        if "<NL>" in tokenizer.vocab:
            inputs = [re.sub('\n', "<NL>", text) for text in inputs]
        return {"input_ids": inputs, "labels": examples["bundles"]}
    return preprocess_function


class ShuffleCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, out_format, add_eos_token, add_lm_token, do_shuffle=True):
        super().__init__(tokenizer)
        self.to_string_func = get_to_string_processor(out_format)
        self.add_eos = add_eos_token
        self.add_lm = add_lm_token
        self.shuffle = do_shuffle

    def __call__(self, features):
        # Tokenize the input texts
        targets = [json.loads(feature["labels"]) for feature in features]
        if self.shuffle:
            for i in range(len(targets)):
                np.random.shuffle(targets[i])

        postfix = self.tokenizer.eos_token if self.add_eos else ""
        prefix = "<LM>" if self.add_lm else ""
        batch = self.tokenizer(
            [prefix + feature["input_ids"] + postfix for feature in features],
            text_target=[self.to_string_func(json.dumps(target)) + (self.tokenizer.eos_token if self.add_eos else "") for target in targets],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch
