import numpy as np
import torch
import re
import json
from enum import Enum

from transformers import TrainerCallback
import evaluate

from utils import Format


def get_parser(tokenizer, format=Format.SpecTokens):
    def spec_tokens_parser(sequence):
        opening = ["<BOT>", "<BOP>", "<BOC1>", "<BOC2>"]
        closing = ["<EOT>", "<EOP>", "<EOC1>", "<EOC2>"]
        names = ["title", "price", "currency", "count"]
        spec_tokens = set(opening + closing)

        eos_index = sequence.find(tokenizer.eos_token)
        if eos_index >= 0:
            sequence = sequence[:eos_index]
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
    
    def just_json_parser(sequence):
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

    if format == Format.SpecTokens:
        return spec_tokens_parser
    elif format == Format.JustJson:
        return just_json_parser
    else:
        raise ValueError(f"Not supportef format: {format}")

        
def is_valid_bundle(bundle):
    return "title" in bundle and "price" in bundle and "currency" in bundle and "count" in bundle        


def reorder_bundles(predicted_bundles, target_bundles):
    defaults = {"title": "", "price": "-1", "currency": "", "count": ""}
    min_common_bundles = min(len(predicted_bundles), len(target_bundles))
    # bleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")

    scores = np.array([
        [chrf_score.compute(predictions=[pred_bundle.get("title", "")], references=[target_bundle["title"]])["score"]
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


def compute_metrics(decoded_preds, decoded_targets, parser):
    metrics = {}
    bleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")

    bleu = bleu_score.compute(predictions=decoded_preds, references=decoded_targets)
    metrics["Global BLEU"] = bleu["score"]

    valid_prediction_structures = 0
    valid_bundles_count, bundles_count = 0, 0
    n_bundles_error, over_bundles, under_bundles = 0, 0, 0
    one_bundle_count, multi_bundle_count = 0, 0
    one_bundle_title_bleu, multi_bundle_title_bleu = 0, 0
    one_bundle_title_chrf, multi_bundle_title_chrf = 0, 0
    one_bundle_price_match, multi_bundle_price_match = 0, 0
    one_bundle_currency_match, multi_bundle_currency_match = 0, 0
    one_bundle_count_match, multi_bundle_count_match = 0, 0
    skipped = 0
    for pred, target in zip(decoded_preds, decoded_targets):
        pred = re.sub(r'(</s>)+', '</s>', pred)
        target = re.sub(r'(</s>)+', '</s>', target)
        # print("T", target)
        # print("P", pred)
        target_is_valid, target_bundles = parser(target)
        prediction_is_valid, pred_bundles = parser(pred)
        
        if not target_is_valid:
            skipped += 1
            continue

        valid_prediction_structures += int(prediction_is_valid)
        if prediction_is_valid:
            bundles_count += len(pred_bundles)
            valid_bundles_count += sum(is_valid_bundle(bundle) for bundle in pred_bundles)
            n_bundles_error += abs(len(pred_bundles) - len(target_bundles))
            over_bundles += max(len(pred_bundles) - len(target_bundles), 0)
            under_bundles += max(len(target_bundles) - len(pred_bundles), 0)

            pred_bundles, target_bundles = reorder_bundles(pred_bundles, target_bundles)

            if len(target_bundles) == 0:
                continue
            elif len(target_bundles) == 1:
                one_bundle_count += 1
                one_bundle_title_bleu += bleu_score.compute(predictions=[pred_bundles[0]["title"]],
                                                            references=[target_bundles[0]["title"]])["score"]
                one_bundle_title_chrf += chrf_score.compute(predictions=[pred_bundles[0]["title"]],
                                                            references=[target_bundles[0]["title"]])["score"]
                one_bundle_price_match += int(pred_bundles[0]["price"] == target_bundles[0]["price"])
                one_bundle_currency_match += int(pred_bundles[0]["currency"] == target_bundles[0]["currency"])
                one_bundle_count_match += int(pred_bundles[0]["count"] == target_bundles[0]["count"])       
            else:
                multi_bundle_count += 1
                multi_bundle_title_bleu += np.mean([bleu_score.compute(predictions=[pred_bundle["title"]], references=[target_bundle["title"]])["score"]
                                                        for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)])
                multi_bundle_title_chrf += np.mean([chrf_score.compute(predictions=[pred_bundle["title"]], references=[target_bundle["title"]])["score"]
                                                        for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)])
                multi_bundle_price_match += sum(pred_bundle["price"] == target_bundle["price"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)) / \
                    len(target_bundles)
                multi_bundle_currency_match += sum(pred_bundle["currency"] == target_bundle["currency"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)) / \
                    len(target_bundles)
                multi_bundle_count_match += sum(pred_bundle["count"] == target_bundle["count"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)) / \
                    len(target_bundles)

    metrics["valid_answer_structure_precent"] = valid_prediction_structures / ((len(decoded_preds) - skipped) or 1) * 100
    metrics["valid_bundles_precent"] = valid_bundles_count / (bundles_count or 1) * 100
    metrics["n_bundles_mae"] = n_bundles_error / (valid_prediction_structures or 1)
    metrics["mean_over_bundles"] = over_bundles / (valid_prediction_structures or 1)
    metrics["mean_under_bundles"] = under_bundles / (valid_prediction_structures or 1)

    metrics["title_bleu_1_bundle"] = one_bundle_title_bleu / (one_bundle_count or 1)
    metrics["title_chrf_1_bundle"] = one_bundle_title_chrf / (one_bundle_count or 1)
    metrics["price_match_precent_1_bundle"] = one_bundle_price_match / (one_bundle_count or 1) * 100
    metrics["currency_match_precent_1_bundle"] = one_bundle_currency_match / (one_bundle_count or 1) * 100
    metrics["count_match_precent_1_bundle"] = one_bundle_count_match / (one_bundle_count or 1) * 100

    metrics["title_bleu_multi_bundle"] = multi_bundle_title_bleu / (multi_bundle_count or 1)
    metrics["title_chrf_multi_bundle"] = multi_bundle_title_chrf / (multi_bundle_count or 1)
    metrics["price_match_precent_multi_bundle"] = multi_bundle_price_match / (multi_bundle_count or 1) * 100
    metrics["currency_match_precent_multi_bundle"] = multi_bundle_currency_match / (multi_bundle_count or 1) * 100
    metrics["count_match_precent_multi_bundle"] = multi_bundle_count_match / (multi_bundle_count or 1) * 100
    return metrics


def compute_test_metrics(decoded_preds, decoded_targets, parser):
    metrics = {}
    bleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")
    
    pred, target = decoded_preds[0], decoded_targets[0]

    pred = re.sub(r'(</s>)+', '</s>', pred)
    target = re.sub(r'(</s>)+', '</s>', target)
    target_is_valid, target_bundles = parser(target)
    prediction_is_valid, pred_bundles = parser(pred)

    if not target_is_valid:
        return metrics

    metrics["valid_structure"] = prediction_is_valid
    if prediction_is_valid:
        metrics["pred_n_bundles"] = len(pred_bundles)
        metrics["valid_bundles"] = sum(is_valid_bundle(bundle) for bundle in pred_bundles)
        metrics["delta_bundles"] = len(pred_bundles) - len(target_bundles)

        pred_bundles, target_bundles = reorder_bundles(pred_bundles, target_bundles)

        if len(target_bundles) == 0:
            return metrics
        metrics["mean_bleu"] = np.mean([bleu_score.compute(predictions=[pred_bundle["title"]], references=[target_bundle["title"]])["score"]
                                        for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)])
        metrics["mean_chrf"] = np.mean([chrf_score.compute(predictions=[pred_bundle["title"]], references=[target_bundle["title"]])["score"]
                                        for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)])
        metrics["mean_price_match"] = sum(pred_bundle["price"] == target_bundle["price"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles))                                                                 / len(target_bundles)
        metrics["mean_currency_match"] = sum(pred_bundle["currency"] == target_bundle["currency"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)) \
                                             / len(target_bundles)
        metrics["mean_count_match"] = sum(pred_bundle["count"] == target_bundle["count"] for pred_bundle, target_bundle in zip(pred_bundles, target_bundles)) \
                                          / len(target_bundles)
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
