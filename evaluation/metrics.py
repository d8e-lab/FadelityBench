import evaluate
import numpy as np
from bart_score import BARTScorer
from bert_score import BERTScorer
import argparse
import json
from bleurt import score
import tensorflow as tf
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import math
import pickle
import os
import csv
from typing import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--hf_evaluate", action="store_true", help="use hf evaluate for BERT score")
parser.add_argument("--bert_model", type=str, default="microsoft/deberta-xlarge-mnli", help="bert model to use for BERT score")
parser.add_argument("--parabank", action="store_true", help="use bart model finetuned on parabank")
parser.add_argument("--bleurt_model", type=str, default="", help="bleurt model PATH to use for BLEURT score, default is BERT-Tiny")
parser.add_argument("--predict_path", type=str, default="data/tst_pred.pkl", help="path to predictions")
parser.add_argument("--reference_path", type=str, default="data/tst_ref.pkl", help="path to references")
args = parser.parse_args()

# your api key
api_key = os.environ["API_KEY"]

# your api base
api_base = os.environ["API_BASE_URL"]

# client = OpenAI(base_url=api_base, api_key=api_key)

with open("evaluation/system_prompt.txt", "r") as f:
    system_prompt = f.read()

class MetricScore:
    def __init__(self):
        self.pred_input_path = args.predict_path
        self.ref_input_path = args.reference_path
        self.data = []
        self.ref_data = []
        with open(self.pred_input_path,'r') as f:
            for line in f:
                self.data.append(line)
        with open(self.ref_input_path,'r') as f:
            for line in f:
                self.ref_data.append(line)

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
        ) = BERT_score(self.data, self.ref_data)

        # gpt_score, gpt_std = get_gpt_score(self.data, self.ref_data)
        gpt_score, gpt_std = math.nan,math.nan
        bart_score, bart_score_std = BART_score(self.data, self.ref_data)
        bleurt_score, bleurt_score_std = BLEURT_score(self.data, self.ref_data)
        
        tokens_predict = [s.split() for s in self.data]

        scores["gpt_score"] = gpt_score
        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1

        scores["gpt_std"] = gpt_std
        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std

        scores["bart_score"] = bart_score
        scores["bart_score_std"] = bart_score_std
        scores["bleurt_score"] = bleurt_score
        scores["bleurt_score_std"] = bleurt_score_std
        return scores

    def print_score(self):
        scores = self.get_score()
        self.save_score(scores)
        print("Explanability Evaluation Metrics:")
        print(f"gpt_score: {scores['gpt_score']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"bart_score: {scores['bart_score']:.4f}")
        print(f"bleurt_score: {scores['bleurt_score']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"gpt_std: {scores['gpt_std']:.4f}")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        print(f"bart_score_std: {scores['bart_score_std']:.4f}")
        print(f"bleurt_score_std: {scores['bleurt_score_std']:.4f}")
    
    def save_score(self,scores:Dict):
        with open("experiments/metric_scores.csv", "a+", newline='') as csvfile:
            pred_file = os.path.basename(self.pred_input_path)
            scores.update({"File":pred_file})
            writer = csv.DictWriter(csvfile,fieldnames=[key for key in scores.keys()])
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(scores)

def get_gpt_response(prompt):
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
    )
    response = completion.choices[0].message.content
    return float(response)


def get_gpt_score(predictions, references):
    prompts = []
    for i in range(len(predictions)):
        prompt = {
            "prediction": predictions[i],
            "reference": references[i],
        }
        prompts.append(json.dumps(prompt))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(get_gpt_response, prompt) for prompt in prompts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Processing GPT responses"):
            results.append(future.result())

    return np.mean(results), np.std(results)

def BERT_score(predictions, references):
    if args.hf_evaluate:
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            rescale_with_baseline=True,
        )
        precision = results["precision"]
        recall = results["recall"]
        f1 = results["f1"]
    else:
        bertscore = BERTScorer(model_type=args.bert_model, # "microsoft/deberta-xlarge-mnli","roberta-large",
                               rescale_with_baseline=True, 
                               lang='en',
                               device="cuda:0")
        precision, recall, f1 = bertscore.score(predictions, references)
        precision = precision.cpu().numpy()
        recall = recall.cpu().numpy()
        f1 = f1.cpu().numpy()
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )

def BART_score(predictions, references):
    bart_scorer = BARTScorer(device='cuda:0', checkpoint="facebook/bart-large-cnn")
    if args.parabank:
        bart_scorer.load(path='evaluation/models/bart_score.pth')
    scores = []
    for i in tqdm(range(0, len(predictions), 4), desc="Computing BART scores"):
        batch_pred = predictions[i:i+4]
        batch_ref = references[i:i+4]
        batch_scores = bart_scorer.score(batch_pred, batch_ref, batch_size=4)
        scores.extend(batch_scores)
    return np.mean(scores), np.std(scores)

def BLEURT_score(predictions, references):
    bleurt = score.BleurtScorer(checkpoint=args.bleurt_model)
    scores = bleurt.score(references=references, candidates=predictions)
    return np.mean(scores), np.std(scores)