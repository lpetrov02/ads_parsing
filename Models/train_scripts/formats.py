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


class Format(Enum):
    SpecTokens = "SpecTokens"
    JustJson = "JustJson"
    SimpleLightTokens = "SimpleLightTokens"
    LightTokens = "LightTokens"


def get_tokens_to_add(out_format):
    if out_format == Format.JustJson:
        return ["{", "}"]
    elif out_format == Format.SpecTokens:
        return ["<BOB>", "<EOB>", "<BOT>", "<EOT>", "<BOP>", "<EOP>", "<BOC1>", "<EOC1>", "<BOC2>", "<EOC2>"]
    elif out_format == Format.LightTokens or out_format == Format.SimpleLightTokens:
        return ["<BOB>", "<BOT>", "<BOP>", "<BOC1>", "<BOC2>"]
    else:
        return []
    

def spec_tokens_formatter(bundles_json):
    bundles = json.loads(bundles_json)
    columns = ["Title", "Price", "Currency", "Count"]
    spec_tokens = [("<BOT>", "<EOT>"), ("<BOP>", "<EOP>"), ("<BOC1>", "<EOC1>"), ("<BOC2>", "<EOC2>")]

    def wrap(s, begin, end):
        return begin + s + end

    s = "".join(
        ["<BOB> " + "".join(
            [f"{prev} " + str(bundle[col]) + f" {post} " for col, (prev, post) in zip(columns, spec_tokens)]
        ) + " <EOB> " for bundle in bundles]
    )
    return " ".join(s.split())   


def light_tokens_formatter(bundles_json, simple=False):
    bundles = json.loads(bundles_json)
    columns = ["Title", "Price", "Currency", "Count"]
    spec_tokens = ["<BOT>", "<BOP>", "<BOC1>", "<BOC2>"]

    encoded_bundles = [
        f"<BOB>{len(bundles) - 1 - i if not simple else ''}" + "".join([f"{tok}{str(bundle[col]).lower()}" for tok, col in zip(spec_tokens, columns)])
            for i, bundle in enumerate(bundles)
    ]
    return (str(len(bundles)) if not simple else "") + "".join(encoded_bundles)


def just_json_formatter(bundles_json):
    return str([{key.lower(): value for key, value in bundle.items()} for bundle in json.loads(bundles_json)])


def get_to_string_processor(format):
    if format == Format.SpecTokens:
        return spec_tokens_formatter
    elif format == Format.SimpleLightTokens:
        return (lambda bundles_json: light_tokens_formatter(bundles_json, simple=True))
    elif format == Format.LightTokens:
        return (lambda bundles_json: light_tokens_formatter(bundles_json, simple=False))
    elif format == Format.JustJson:
        return just_json_formatter

    else:
        raise ValueError(f"Not supportef format: {format}")


def spec_tokens_parser(sequence, tokenizer):
    sequence = re.sub(tokenizer.pad_token, "", sequence)
    opening = ["<BOT>", "<BOP>", "<BOC1>", "<BOC2>"]
    closing = ["<EOT>", "<EOP>", "<EOC1>", "<EOC2>"]
    names = ["title", "price", "currency", "count"]
    spec_tokens = set(opening + closing)

    eos_index = sequence.find(tokenizer.eos_token)
    if eos_index >= 0:
        sequence = sequence[:eos_index]
    for spec_token in opening + closing + ["<BOB>", "<EOB>"]:
        sequence = re.sub(spec_token, f" {spec_token} ", sequence).strip()
    sequence = list(sequence.split())

    def parse_bundle(sub_sequence):
        parsed = {}
        open_state = False
        opening_token_id = None
        current_str = []
        for index, token in enumerate(sub_sequence):
            if token in spec_tokens:
                if open_state:
                    if token != closing[opening_token_id]:
                        return False, None
                    else:
                        parsed[names[opening_token_id]] = " ".join(current_str)
                    open_state = False
                else:
                    if token not in opening:
                        return False, None
                    else:
                        open_state = True
                        opening_token_id = opening.index(token)
                        current_str = []
            else:
                current_str.append(token)
        if open_state:
            return False, None
        return True, parsed

    start_token = "<BOB>"
    end_token = "<EOB>"
    bundle_start, open_state = None, False

    bundles = []
    for index, token in enumerate(sequence):
        if token == tokenizer.eos_token:
            break
        if token == tokenizer.pad_token:
            continue
        if open_state:
            if token == end_token:
                open_state = False
                valid, bundle = parse_bundle(sequence[bundle_start + 1: index])
                if not valid:
                    return False, None
                else:
                    bundles.append(bundle)
            elif token == start_token:
                return False, None
        else:
            if token != start_token:
                return False, None
            else:
                open_state = True
                bundle_start = index
    if open_state:
        return False, None
    return True, bundles


def light_tokens_parser(sequence, tokenizer, simple=False):
    spec_tokens = ["<BOT>", "<BOP>", "<BOC1>", "<BOC2>"]
    names = ["title", "price", "currency", "count"]
    tok2name = dict(zip(spec_tokens, names))

    eos_index = sequence.find(tokenizer.eos_token)
    if eos_index >= 0:
        sequence = sequence[:eos_index]
    sequence = sequence.strip()
    if not simple:
        if re.match(r'^[0-9]+', sequence) is None:
            return False, None
        sequence = re.sub(r'^[0-9]+', '', sequence)
    if len(sequence) > 0 and re.match(r'^<BOB>', sequence) is None:
        return False, None
    sequence = re.sub(r'^<BOB>', '', sequence)

    bundles = []
    for bundle in sequence.split('<BOB>'):
        bundle = bundle.strip()
        if len(bundle) == 0:
            continue
        if not simple:
            if re.match(r'^[0-9]+', bundle) is None:
                return False, None
            bundle = re.sub(r'^[0-9]+(\s*)', '', bundle)
        for tok in spec_tokens:
            if bundle.count(tok) > 1:
                return False, None
        inds = sorted([(tok, bundle.index(tok)) for tok in spec_tokens if tok in bundle], key=lambda x: x[1])
        if len(inds) > 0 and inds[0][1] != 0:
            return False, None
        bundle_dict = {}
        for i, (tok, ind) in enumerate(inds):
            bundle_dict[tok2name[tok]] = bundle[ind + len(tok): inds[i + 1][1]] if i < len(inds) - 1 else bundle[ind + len(tok):]
            bundle_dict[tok2name[tok]] = bundle_dict[tok2name[tok]].strip()
        bundles.append(bundle_dict)
    return True, bundles


def just_json_parser(sequence, tokenizer):
    eos_index = sequence.find(tokenizer.eos_token)
    if eos_index >= 0:
        sequence = sequence[:eos_index]

    try:
        bundles = eval(sequence)
        if not isinstance(bundles, list):
            return False, None
        for bundle in bundles:
            if not isinstance(bundle, dict):
                return False, None
        return True, bundles
    except SyntaxError as e:
        return False, None


def get_parser(tokenizer, format=Format.SpecTokens):
    if format == Format.SpecTokens:
        return (lambda sequence: spec_tokens_parser(sequence, tokenizer))
    elif format == Format.LightTokens:
        return (lambda sequence: light_tokens_parser(sequence, tokenizer, simple=False))
    elif format == Format.SimpleLightTokens:
        return (lambda sequence: light_tokens_parser(sequence, tokenizer, simple=True))
    elif format == Format.JustJson:
        return (lambda sequence: just_json_parser(sequence, tokenizer))
    else:
        raise ValueError(f"Not supportef format: {format}")