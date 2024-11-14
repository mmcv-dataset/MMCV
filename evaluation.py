import os
import json
import pickle
import re
import argparse
import random
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

class MultiHopEvaluator:
    def __init__(self, model_name, prompt_type):
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.datasets = ['1hop', '2hop', '3hop', '4hop']

    def load_data(self, dataset_name):
        with open(f"dataset/{dataset_name}.json", "r") as f:
            return json.load(f)

    def load_predictions(self, dataset_name):
        file_path = f"MLLM_Results/{dataset_name}_{self.model_name}_{self.prompt_type}.pkl"
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def process_predictions(self, predictions):
        processed = []
        for pred in predictions:
            if pred is None:
                processed.append("")
            elif isinstance(pred, str):
                if self.model_name in ["gpt-4", "gemini"]:
                    match = re.search(r"Prediction:\s*(.*)", pred, re.IGNORECASE)
                    if match:
                        processed.append(re.sub(r"[\W_]+", "", match.group(1)))
                    else:
                        processed.append("")
                elif self.model_name == "llava":
                    try:
                        match = pred.split("Prediction:")[2].split("Explanation:")[0]
                        processed.append(re.sub(r"[\W_]+", "", match))
                    except:
                        processed.append("")
            else:
                processed.append("")
        return processed

    def clean_data(self, predictions, gold_labels, data):
        valid_indices = [i for i, pred in enumerate(predictions) if pred in ["SUPPORT", "REFUTE"]]
        return ([predictions[i] for i in valid_indices], 
                [gold_labels[i] for i in valid_indices], 
                [data[i] for i in valid_indices])

    def evaluate_dataset(self, dataset_name):
        data = self.load_data(dataset_name)
        predictions = self.load_predictions(dataset_name)
        
        processed_predictions = self.process_predictions(predictions)
        gold_labels = [entry["label"] for entry in data]

        processed_predictions, gold_labels, cleaned_data = self.clean_data(processed_predictions, gold_labels, data)

        accuracy = sum(1 for p, g in zip(processed_predictions, gold_labels) if p == g) / len(gold_labels)

        target_names = ["REFUTE", "SUPPORT"]
        label_map = {"REFUTE": 0, "SUPPORT": 1}
        labels = [label_map[e] for e in gold_labels]
        pred_labels = [label_map[e] for e in processed_predictions]

        report = classification_report(labels, pred_labels, target_names=target_names, digits=4, output_dict=True)
        conf_matrix = confusion_matrix(labels, pred_labels)

        false_positives = [
            (pred, data, raw_pred)
            for pred, label, data, raw_pred in zip(processed_predictions, gold_labels, cleaned_data, predictions)
            if pred == "SUPPORT" and label == "REFUTE" and data.get("image_evidence") and data["image_evidence"] != []
        ]

        true_negatives = [
            (pred, data, raw_pred)
            for pred, label, data, raw_pred in zip(processed_predictions, gold_labels, cleaned_data, predictions)
            if pred == "REFUTE" and label == "REFUTE" and data.get("image_evidence") and data["image_evidence"] != []
        ]

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "false_positives": false_positives,
            "true_negatives": true_negatives
        }

    def print_results(self, results):
        for dataset, result in results.items():
            print(f"\nResults for {dataset}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("Classification Report:")
            for label, metrics in result['classification_report'].items():
                if isinstance(metrics, dict):
                    print(f"  {label}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.4f}")
            print("Confusion Matrix:")
            print(result['confusion_matrix'])
            
            if result['false_positives']:
                fp_example = random.choice(result['false_positives'])
                print("\nFalse Positive Example (with non-empty image evidence):")
                print(f"Claim: {fp_example[1]['claim']}")
                print(f"True Label: REFUTE")
                print(f"Predicted Label: {fp_example[0]}")
                print(f"Image Evidence: {fp_example[1]['image_evidence']}")
                print(f"Raw Prediction from .pkl: {fp_example[2]}")
            else:
                print("\nNo False Positive examples with non-empty image evidence found for this dataset.")
            
            if result['true_negatives']:
                tn_example = random.choice(result['true_negatives'])
                print("\nTrue Negative Example (with non-empty image evidence):")
                print(f"Claim: {tn_example[1]['claim']}")
                print(f"True Label: REFUTE")
                print(f"Predicted Label: {tn_example[0]}")
                print(f"Image Evidence: {tn_example[1]['image_evidence']}")
                print(f"Raw Prediction from .pkl: {tn_example[2]}")
            else:
                print("\nNo True Negative examples with non-empty image evidence found for this dataset.")
            
            print("=" * 50)

    def run_evaluation(self):
        results = {}
        for dataset in self.datasets:
            print(f"Evaluating {dataset}...")
            results[dataset] = self.evaluate_dataset(dataset)

        self.print_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name [llava, gpt-4, gemini]")
    parser.add_argument("--prompt_type", type=str, required=True, help="prompt type [open_book, closed_book]")
    args = parser.parse_args()

    evaluator = MultiHopEvaluator(args.model, args.prompt_type)
    evaluator.run_evaluation()