import numpy as np
import torch
import re
import json
from enum import Enum
from collections import defaultdict

from transformers import TrainerCallback
import evaluate

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from formats import Format, get_parser

        
def is_valid_bundle(bundle):
    return "title" in bundle and "price" in bundle and "currency" in bundle and "count" in bundle        


def reorder_bundles(predicted_bundles, target_bundles, scorer, key):
    defaults = {"title": "", "price": "-1", "currency": "", "count": ""}
    min_common_bundles = min(len(predicted_bundles), len(target_bundles))

    scores = np.array([
        [scorer.compute(predictions=[pred_bundle.get("title", "")], references=[target_bundle["title"]])[key]
            for target_bundle in target_bundles] for pred_bundle in predicted_bundles
    ])
    
    preds, targets = [], []
    for _ in range(min_common_bundles):
        max_bleu = np.argmax(scores)
        match_pred, match_target = max_bleu // len(target_bundles), max_bleu % len(target_bundles)
        preds.append({col: predicted_bundles[match_pred].get(col, defaults[col]) for col in defaults})
        targets.append({col: target_bundles[match_target].get(col, defaults[col]) for col in defaults})
        scores[match_pred, :] = -1
        scores[:, match_target] = -1

    return preds, targets


def preprocess_text(text):
    punctuation = ",.!?#-+_=<>\"'"
    stemmer = PorterStemmer()
    
    tokens = [stemmer.stem(word.strip().strip(punctuation)).lower() for word in word_tokenize(text)]
    return " ".join(tokens)


def titles_multiple_precision_recall_f(preds, targets):
    def titles_precision_recall_f(pred, target):
        pred_set = set(list(pred.split()))
        target_set = set(list(target.split()))
        TP = len(pred_set & target_set)
        FP = len(pred_set - target_set)
        FN = len(target_set - pred_set)
        precision, recall = TP / ((TP + FP) or 1), TP / ((TP + FN) or 1)
        return precision, recall, 2 * precision * recall / ((precision + recall) or 1)

    metrics = {"precision": [], "recall": [], "f1": []}
    for pred, target in zip(preds, targets):
        p, r, f = titles_precision_recall_f(pred, target)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f)
    return metrics


def count_title_metrics(preds, targets, scorers, keys):
    assert len(preds) == len(targets)
    metrics = titles_multiple_precision_recall_f(preds, targets)
    metrics["exact_match"] = [int(pred == target) for pred, target in zip(preds, targets)]

    for name in scorers:
        instances = [scorers[name].compute(predictions=[pred], references=[target]) for pred, target in zip(preds, targets)]
        for key in keys.get(name, ["score"]):
            metrics[f"{name}_{key}"] = [instance[key] for instance in instances]
    return metrics


def compare_currency(pred, target):
    groups = {
        "lari": ("lari", "лари", "лар", "gel", "₾",),
        "rub": ("р", "руб", "рубли", "рублей", "rub", "₽",),
        "eur": ("е", "евро", "eur", "euro", "€",),
        "usd": ("доллар", "долларов", "доллары", "usd", "$",),
    }
    pred, target = pred.lower().strip(".,"), target.lower().strip(".,")
    if pred == target:
        return True
    for curr_name in groups:
        if pred in groups[curr_name] and target in groups[curr_name]:
            return True
    return False
    

def count_other_metrics(preds, targets):
    assert len(preds) == len(targets)
    metrics = {
        "price_match": [int(pred["price"] == target["price"]) for pred, target in zip(preds, targets)],
        "currency_match": [int(compare_currency(pred["currency"], target["currency"])) for pred, target in zip(preds, targets)],
        "count_match": [int(pred["count"].lower() == target["count"].lower()) for pred, target in zip(preds, targets)],
    }
    return metrics


def count_bundle_metrics(preds, targets, title_scorers, title_keys):
    assert len(preds) == len(targets)
    n_pairs = len(preds)
    metrics = count_title_metrics([pred.get("title") for pred in preds], [target.get("title") for target in targets], title_scorers, title_keys)
    metrics.update(count_other_metrics(preds, targets))
    metrics["match"] = []
    for i in range(n_pairs):
        metrics["match"].append(metrics["price_match"][i] * metrics["currency_match"][i] * metrics["count_match"][i] * metrics["f1"][i])
    return {key: np.sum(val) if val else 0 for key, val in metrics.items()}


def deduplicate(bundles):
    fields = ("title", "price", "currency", "count")
    tuples = [tuple([bundle.get(col) for col in fields]) for bundle in bundles]
    tuples = list(set(tuples))
    dedup_bundles = [{col: tup[i] for i, col in enumerate(fields) if tup[i] is not None} for tup in tuples]
    return dedup_bundles, len(bundles) - len(dedup_bundles)

    
def compute_metrics(decoded_preds, decoded_targets, parser):
    metrics = {}
    
    title_scorers = {
        "bleu": evaluate.load("sacrebleu"),
        "chrf": evaluate.load("chrf"),
    }
    title_scorer_keys = {
    }

    duplicates_count = 0
    valid_prediction_structures = 0
    valid_bundles_count, bundles_count = 0, 0
    n_bundles_error, over_bundles, under_bundles = 0, 0, 0
    one_bundle_count, multi_bundle_count = 0, 0
    one_bundle_metrics = defaultdict(float)
    multi_bundle_metrics = defaultdict(float)
    skipped = 0
    for pred, target in zip(decoded_preds, decoded_targets):
        pred = re.sub(r'(</s>)+', '</s>', pred)
        target = re.sub(r'(</s>)+', '</s>', target)
        target_is_valid, target_bundles = parser(target)
        prediction_is_valid, pred_bundles = parser(pred)
        
        if not target_is_valid:
            skipped += 1
            continue

        valid_prediction_structures += int(prediction_is_valid)
        if prediction_is_valid:
            # preprocess titles
            for i in range(len(pred_bundles)):
                if "title" in pred_bundles[i]:
                    pred_bundles[i]["title"] = preprocess_text(pred_bundles[i]["title"])
            for i in range(len(target_bundles)):
                if "title" in target_bundles[i]:
                    target_bundles[i]["title"] = preprocess_text(target_bundles[i]["title"])
            
            # remove duplicates
            pred_bundles, n_duplicates = deduplicate(pred_bundles)
            duplicates_count += n_duplicates
            
            #count bundles
            bundles_count += len(pred_bundles)
            valid_bundles_count += sum(is_valid_bundle(bundle) for bundle in pred_bundles)
            n_bundles_error += abs(len(pred_bundles) - len(target_bundles))
            over_bundles += max(len(pred_bundles) - len(target_bundles), 0)
            under_bundles += max(len(target_bundles) - len(pred_bundles), 0)

            # reorder_bundles
            original_gt_length = len(target_bundles)
            pred_bundles, target_bundles = reorder_bundles(pred_bundles, target_bundles, title_scorers["chrf"], "score")

            if original_gt_length == 1:
                one_bundle_count += 1
            elif original_gt_length > 1:
                multi_bundle_count += 1

            ad_metrics = count_bundle_metrics(pred_bundles, target_bundles, title_scorers, title_scorer_keys)
            if original_gt_length == 0:
                continue
            
            if original_gt_length == 1:
                for key in ad_metrics:
                    one_bundle_metrics[key] += ad_metrics[key] 
            else:
                for key in ad_metrics:
                    multi_bundle_metrics[key] += ad_metrics[key] / original_gt_length

    metrics["valid_answer_structure_precent"] = valid_prediction_structures / ((len(decoded_preds) - skipped) or 1) * 100
    metrics["mean_duplicates"] = duplicates_count / (valid_prediction_structures or 1)
    metrics["valid_bundles_precent"] = valid_bundles_count / (bundles_count or 1) * 100
    metrics["n_bundles_mae"] = n_bundles_error / (valid_prediction_structures or 1)
    metrics["mean_over_bundles"] = over_bundles / (valid_prediction_structures or 1)
    metrics["mean_under_bundles"] = under_bundles / (valid_prediction_structures or 1)
    
    for key in one_bundle_metrics:
        metrics[f"1b_{key}"] = one_bundle_metrics[key] / (one_bundle_count or 1)

    for key in multi_bundle_metrics:
        metrics[f"mb_{key}"] = multi_bundle_metrics[key] / (multi_bundle_count or 1)
    return metrics


def compute_test_metrics(decoded_preds, decoded_targets, parser):
    metrics = {}
    title_scorers = {
        "bleu": evaluate.load("sacrebleu"),
        "chrf": evaluate.load("chrf"),
    }
    title_scorer_keys = {
        "rouge": ["rouge1", "rouge2", "rougeL"],
    }
    
    pred, target = decoded_preds[0], decoded_targets[0]
    
    pred = re.sub(r'(</s>)+', '</s>', pred)
    target = re.sub(r'(</s>)+', '</s>', target)
    target_is_valid, target_bundles = parser(target)
    prediction_is_valid, pred_bundles = parser(pred)

    if not target_is_valid:
        return metrics

    metrics["valid_structure"] = prediction_is_valid
    if prediction_is_valid:
        # preprocess titles
        for i in range(len(pred_bundles)):
            if "title" in pred_bundles[i]:
                pred_bundles[i]["title"] = preprocess_text(pred_bundles[i]["title"])
        for i in range(len(target_bundles)):
            if "title" in target_bundles[i]:
                target_bundles[i]["title"] = preprocess_text(target_bundles[i]["title"])
                
        # remove duplicates
        pred_bundles, n_duplicates = deduplicate(pred_bundles)
        metrics["n_duplicates"] = n_duplicates
        
        # count bundles
        metrics["pred_n_bundles"] = len(pred_bundles)
        metrics["valid_bundles"] = sum(is_valid_bundle(bundle) for bundle in pred_bundles)
        metrics["delta_bundles"] = len(pred_bundles) - len(target_bundles)

        original_gt_length = len(target_bundles)
        pred_bundles, target_bundles = reorder_bundles(pred_bundles, target_bundles, title_scorers["chrf"], "score")

        ad_metrics = count_bundle_metrics(pred_bundles, target_bundles, title_scorers, title_scorer_keys)
        if original_gt_length == 0:
            return metrics
        
        for key in ad_metrics:
            metrics[key] = ad_metrics[key] / original_gt_length
    return metrics


def cut_sequences(seqs, sep_token_id):
    def process(tokens_list):
        if sep_token_id in tokens_list:
            tokens_list = tokens_list[tokens_list.index(sep_token_id) + 1:]
        if -100 in tokens_list:
            tokens_list = tokens_list[:tokens_list.index(-100)]
        return tokens_list

    clean_seqs = []
    max_len = 0
    for seq in seqs:
        clean_seqs.append(process(seq.tolist()))
        max_len = max(max_len, len(clean_seqs[-1]))

    return np.array([seq + [-100] * (max_len - len(seq)) for seq in clean_seqs])


def get_metrics_computer(tokenizer, format):
    parser = get_parser(tokenizer, format)

    def metrics_computer(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        targets = np.where(targets == -100, tokenizer.eos_token_id, targets)
        decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = np.where(preds == -100, tokenizer.eos_token_id, preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        return compute_metrics(decoded_preds, decoded_targets, parser)

    return metrics_computer
    

class PrintCallback(TrainerCallback):
    def __init__(self, format, show=0, device='cpu'):
        self.format=format
        self.show = show
        self.device = device

    @staticmethod
    def decode(items, tokenizer):
        items = torch.where(items == -100, tokenizer.eos_token_id, items)
        return tokenizer.batch_decode(items, ignore_special_tokens=True)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        ads_count = self.show
        parser = get_parser(tokenizer, self.format)

        def Print(s, title, try_parse=False):
            subs = [(tokenizer.pad_token, ''), ('<pad>', ''), (r'(</s>)+', '</s>'), ('<LM>', ''), ('<NL>', '\n')]
            for token_a, token_b in subs:
                s = re.sub(token_a, token_b, s)
            print('\n' + title)

            print("\tRAW\n\t", s)
            if try_parse:
                is_valid, bundles = parser(s)
                if is_valid:
                    print("\tDECODED:\n\t", bundles)

        for batch in kwargs['eval_dataloader']:
            texts = self.decode(batch["input_ids"][:ads_count], tokenizer)

            preds = model.generate(
                input_ids=batch["input_ids"][:ads_count].to(self.device),
                max_length=256,
                num_beams=4,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id
            ).cpu()
            preds = self.decode(preds, tokenizer)
            labels = self.decode(batch["labels"][:ads_count], tokenizer)

            for text, label, pred in zip(texts, labels, preds):
                Print(text, "ORIGINAL TEXT")
                Print(label, "TARGET", try_parse=True)
                Print(pred, "PREDICTED", try_parse=True)
                print('-' * 50 + '\n')

            ads_count -= len(batch["input_ids"])
            if ads_count <= 0:
                break
