import os
import json
import argparse
import pickle
import re
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm

from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

import get_api_res
from prompt_library import *

load_dotenv()


class MLLM_EXP:
    def __init__(
        self,
        dataset_name,
        model_name,
        prompt_method,
        device,
        raw_image_path="MMQA_Raw/final_dataset_images/",
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.prompt_method = prompt_method
        self.raw_image_path = raw_image_path

        self.torch_device = device

        if self.model_name == "llava":
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16,
            ).to(self.torch_device)
            self.llava_processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            )

    def load_data(self):
        with open(f"dataset/{self.dataset_name}", "r") as f:
            data = json.load(f)
        return data

    def save_to_jsonl(self, data, output_file):
        with open(output_file, "a") as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + "\n")

    def save_to_pickle(self, data, output_file):
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

    def retrieve_text_evidence(self, text_id_list):
        with open("MMQA_Raw/MMQA_texts.jsonl", "r") as f:
            json_list = list(f)

        for json_str in json_list:
            entry = json.loads(json_str)
            for text_id in text_id_list:
                if text_id == entry["id"]:
                    return "".join(entry["text"])

    def retrieve_image_path(self, image_id_list):
        with open("MMQA_Raw/MMQA_images.jsonl", "r") as f:
            json_list = list(f)

        ret_list = []
        for json_str in json_list:
            entry = json.loads(json_str)
            for image_id in image_id_list:
                if image_id == entry["id"]:
                    image_name = entry["path"]
                    ret_list.append(f"{self.raw_image_path}{image_name}")
        return ret_list

    def retrieve_table_evidence(self, table_id_list):
        with open("MMQA_Raw/MMQA_tables.jsonl", "r") as f:
            json_list = list(f)

        ret_list = []
        for json_str in json_list:
            entry = json.loads(json_str)
            for table_id in table_id_list:
                if table_id == entry["id"]:
                    ret_list.append(entry["table"])
        return ret_list

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def call_gemini(self, claim, text_evidence, image_list, table_evidence):
        if self.prompt_method == "closed_book":
            system_prompt = closed_book_system
            prompt = f"Claim: {claim}"

            response = get_api_res.get_gemini_text_response(
                prompt,
                image_list,
                model="gemini-1.5-flash",
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return response
        
        if self.prompt_method == "cot":
            system_prompt = chain_of_thought
            prompt = f"Claim: {claim}"

            response = get_api_res.get_gemini_text_response(
                prompt,
                image_list,
                model="gemini-1.5-flash",
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return response

        if self.prompt_method == "open_book":
            system_prompt = open_book_system
            prompt = f"""Claim: {claim}
            Text Evidence: {text_evidence}
            Table Evidence: {table_evidence}
            Image Evidence: """

            response = get_api_res.get_gemini_text_response(
                prompt,
                image_list,
                model="gemini-1.5-flash",
                system_prompt=system_prompt,
                temperature=0.0,
            )
            return response

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def call_openai(self, claim, text_evidence, image_list, table_evidence):
        if self.prompt_method == "closed_book":
            system_prompt = closed_book_system
            prompt = f"Claim: {claim}"

            response = get_api_res.get_openai_text_response(
                prompt,
                image_list,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return response
        
        if self.prompt_method == "cot":
            system_prompt = chain_of_thought
            prompt = f"""Claim: {claim}
            Text Evidence: {text_evidence}
            Table Evidence: {table_evidence}
            Image Evidence: """

            response = get_api_res.get_openai_text_response(
                prompt,
                image_list,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return response

        if self.prompt_method == "open_book":
            system_prompt = open_book_system
            prompt = f"""Claim: {claim}
            Text Evidence: {text_evidence}
            Table Evidence: {table_evidence}
            Image Evidence: """

            response = get_api_res.get_openai_text_response(
                prompt,
                image_list,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return response

    def call_llava(self, claim, text_evidence, image_list, table_evidence):
        if self.prompt_method == "closed_book":
            system_prompt = closed_book_system
            prompt = f"""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
            ###Human: {system_prompt} \nClaim: {claim}\n###Assistant:"""

            inputs = self.llava_processor(text=prompt, return_tensors="pt").to(
                self.torch_device
            )
            generate_ids = self.llava_model.generate(**inputs, max_new_tokens=10000)
            res = self.llava_processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return res
        
        if self.prompt_method == "cot":
            system_prompt = chain_of_thought
            prompt = f"""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
            ###Human: {system_prompt} \nClaim: {claim}\n###Assistant:"""

            inputs = self.llava_processor(text=prompt, return_tensors="pt").to(
                self.torch_device
            )
            generate_ids = self.llava_model.generate(**inputs, max_new_tokens=10000)
            res = self.llava_processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return res

        if self.prompt_method == "open_book":
            system_prompt = open_book_system

            if len(image_list) >= 1:
                prompt = f"""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
                ###Human: <image> \n{system_prompt} \nClaim: {claim} \nText Evidence: {text_evidence} \nTable Evidence: {table_evidence}\n###Assistant:"""

                image = Image.open(image_list[0])

                inputs = self.llava_processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.torch_device)
                generate_ids = self.llava_model.generate(**inputs, max_new_tokens=10000)
                res = self.llava_processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                return res
            else:
                prompt = f"""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
                ###Human: \n{system_prompt} \nClaim: {claim} \nText Evidence: {text_evidence} \nTable Evidence: {table_evidence}\n###Assistant:"""

                inputs = self.llava_processor(text=prompt, return_tensors="pt").to(
                    self.torch_device
                )
                generate_ids = self.llava_model.generate(**inputs, max_new_tokens=10000)
                res = self.llava_processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                return res

    def get_output(self, data):
        results = []

        for entry in tqdm(data, total=len(data), desc="Getting Output"):
            claim = entry["claim"]

            wiki_context = entry["wiki_context"]
            if len(entry["text_evidence"]) != 0:
                text_evidence = self.retrieve_text_evidence(entry["text_evidence"])
            else:
                text_evidence = ""
            context = wiki_context + text_evidence

            if len(entry["image_evidence"]) != 0:
                image_list = self.retrieve_image_path(entry["image_evidence"])
            else:
                image_list = ""

            if len(entry["table_evidence"]) != 0:
                table_evidence = self.retrieve_table_evidence(entry["table_evidence"])
                table_evidence_str = ",".join(
                    str(element) for element in table_evidence
                )
            else:
                table_evidence_str = ""

            if self.model_name == "gemini":
                res = self.call_gemini(claim, context, image_list, table_evidence_str)
                # print(res)
                results.append(res)

            if self.model_name == "gpt-4o-mini":
                res = self.call_openai(claim, context, image_list, table_evidence_str)
                # print(res)
                results.append(res)

            if self.model_name == "llava":
                res = self.call_llava(claim, context, image_list, table_evidence_str)
                # print(res)
                results.append(res)

        self.save_to_pickle(
            results,
            f"MLLM_Results/{self.dataset_name[:-5]}_{self.model_name}_{self.prompt_method}.pkl",
        )  # TODO: Fix dataset_name

    def evaluate(self, data):
        with open(
            f"MLLM_Results/{self.dataset_name[:-5]}_{self.model_name}_{self.prompt_method}.pkl",
            "rb",
        ) as f:
            output = pickle.load(f)
        # print(output)

        if args.model == "gpt-4o-mini" or args.model == "gemini":
            prediction_list, confidence_list = [], []
            for i in output:
                try:
                    pattern1 = re.compile(
                        r"Prediction:\s*(.*)",
                        re.IGNORECASE,
                    )
                    match = pattern1.match(i)[1]
                    prediction_list.append(re.sub("[\W_]+", "", str(match)))
                except:
                    prediction_list.append("")

            print(prediction_list)
            print(len(prediction_list))
            # print(confidence_list)

        if args.model == "llava":
            prediction_list, confidence_list = [], []
            for i in output:
                try:
                    match = i.split("Prediction:")[2].split("Explanation:")[0]
                    prediction_list.append(re.sub("[\W_]+", "", str(match)))
                except:
                    prediction_list.append("")

            print(prediction_list)
            print(len(prediction_list))
            # print(confidence_list)

        num_correct = 0
        gold_label = [entry["label"] for entry in data]
        # gold_label = [entry["label"] for entry in data][:-1] # For Gemini 2hop
        # gold_label = [entry["label"] for entry in data][:-2]  # For Gemini 1hop
        print(len(gold_label))
        assert len(prediction_list) == len(gold_label)

        # Cleanup bad output
        to_remove = []
        for i in range(len(gold_label)):
            if prediction_list[i] != "SUPPORT" and prediction_list[i] != "REFUTE":
                to_remove.append(i)
        for index in sorted(to_remove, reverse=True):
            del prediction_list[index]
            del gold_label[index]
        assert len(prediction_list) == len(gold_label)

        # Calculate accuracy
        for i in range(len(gold_label)):
            if prediction_list[i] == gold_label[i]:
                num_correct = num_correct + 1
        accuracy = num_correct / len(gold_label)
        print(f"Accuracy: {accuracy}")

        # Print result
        target_names = ["REFUTE", "SUPPORT"]
        label_map = {"REFUTE": 0, "SUPPORT": 1}
        labels = [label_map[e] for e in gold_label]
        predictions = [label_map[e] for e in prediction_list]
        print("Classification Report")
        print("=" * 60)
        print(
            classification_report(
                labels, predictions, target_names=target_names, digits=4
            )
        )
        print(confusion_matrix(labels, predictions))

    def run(self):
        data = self.load_data()  #[:50]  # TODO: Toy Example

        if not os.path.exists("MLLM_Results"):
            os.makedirs("MLLM_Results")

        self.get_output(data)  # TODO: Get response
        self.evaluate(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="4hop.json",
        help="dataset [1hop, 2hop, 3hop, 4hop]",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava",
        help="model [llava, gpt-4o-mini, gemini]",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="open_book",
        help="prompt type [open_book, closed_book, cot]",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device")
    args = parser.parse_args()
    print(args)

    mllm = MLLM_EXP(args.dataset, args.model, args.prompt_type, args.device)
    mllm.run()
