import json
import pickle
import re
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

import get_api_res

load_dotenv(".env")

hop = "4hop"


def save_to_pickle(data, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def retrieve_text_evidence(text_id_list):
    with open("MMQA_Raw/MMQA_texts.jsonl", "r") as f:
        json_list = list(f)

    for json_str in json_list:
        entry = json.loads(json_str)
        for text_id in text_id_list:
            if text_id == entry["id"]:
                return "".join(entry["text"])


def retrieve_image_path(image_id_list):
    with open("MMQA_Raw/MMQA_images.jsonl", "r") as f:
        json_list = list(f)

    ret_list = []
    for json_str in json_list:
        entry = json.loads(json_str)
        for image_id in image_id_list:
            if image_id == entry["id"]:
                image_name = entry["path"]
                ret_list.append(f"MMQA_Raw/final_dataset_images/{image_name}")
                return ret_list


def retrieve_table_evidence(table_id_list):
    with open("MMQA_Raw/MMQA_tables.jsonl", "r") as f:
        json_list = list(f)

    ret_list = []
    for json_str in json_list:
        entry = json.loads(json_str)
        for table_id in table_id_list:
            if table_id == entry["id"]:
                ret_list.append(entry["table"])
                return ret_list


def call_gemini(claim, text_evidence, image_list, table_evidence):
    # system_prompt = (
    #     "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
    #     + "Let's think step-by-step!"
    #     + "Please only return the label and nothing else."
    # )

    # system_prompt = (
    #     "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
    #     + "Instruction: 1. Decompose the Claim: Break down the claim into a series of follow-up questions that will help assess the relationship between the claim and the evidence."
    #     + "2. Answer the Follow-Up Questions: Evaluate the evidence to answer each follow-up question."
    #     + "3. Aggregate and Conclude: Use the answers to the follow-up questions to determine the overall relationship between the claim and the evidence."
    #     + "Please only return the label and nothing else."
    # )

    system_prompt = (
        "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
        + "Generate Reasoning Steps by writing a Python-like program that outlines the step-by-step reasoning needed to verify the claim."
        + "You can call three functions in the program : 1. Question () to answer a question ; 2. Verify () to verify a simple claim ; 3. Predict () to predict the veracity label."
        + "Execute the generated reasoning steps to determine the overall relationship between the claim and the evidence and return your judgement."
        + "Please only return the label and nothing else. Just respond with the label without the program. Do not respond with the Python-like Program."
    )

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
    print(response)
    return response


def call_openai(claim, text_evidence, image_list, table_evidence):
    # system_prompt = (
    #     "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
    #     + "Let's think step-by-step!"
    #     + "Please only return the label and nothing else."
    # )

    # system_prompt = (
    #     "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
    #     + "Instruction: 1. Decompose the Claim: Break down the claim into a series of follow-up questions that will help assess the relationship between the claim and the evidence."
    #     + "2. Answer the Follow-Up Questions: Evaluate the evidence to answer each follow-up question."
    #     + "3. Aggregate and Conclude: Use the answers to the follow-up questions to determine the overall relationship between the claim and the evidence."
    #     + "Please only return the label and nothing else."
    # )

    system_prompt = (
        "Given a claim and evidence  (which can be text, table, or an image), determine whether the claim is SUPPORT or REFUTE by the evidence."
        + "Generate Reasoning Steps by writing a Python-like program that outlines the step-by-step reasoning needed to verify the claim."
        + "You can call three functions in the program : 1. Question () to answer a question ; 2. Verify () to verify a simple claim ; 3. Predict () to predict the veracity label."
        + "Execute the generated reasoning steps to determine the overall relationship between the claim and the evidence and return your judgement."
        + "Please only return the label and nothing else. Just respond with the label without the program. Do not respond with the Python-like Program."
    )

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
    print(response)
    return response


with open(f"dataset/{hop}_sampled.pkl", "rb") as f:
    data = pickle.load(f)


results = []
for entry in tqdm(data, total=len(data), desc="Getting Output"):
    claim = entry["claim"]

    wiki_context = entry["wiki_context"]
    if len(entry["text_evidence"]) != 0:
        text_evidence = retrieve_text_evidence(entry["text_evidence"])
    else:
        text_evidence = ""
    context = wiki_context + text_evidence

    if len(entry["image_evidence"]) != 0:
        image_list = retrieve_image_path(entry["image_evidence"])
    else:
        image_list = ""

    if len(entry["table_evidence"]) != 0:
        table_evidence = retrieve_table_evidence(entry["table_evidence"])
        table_evidence_str = ",".join(str(element) for element in table_evidence)
    else:
        table_evidence_str = ""

    # res = call_gemini(claim, context, image_list, table_evidence_str)
    res = call_openai(claim, context, image_list, table_evidence_str)
    results.append(res)
    save_to_pickle(
        results,
        f"MLLM_Results/{hop}_sampled_gpt4_symbolic.pkl",
    )

with open(f"MLLM_Results/{hop}_sampled_gpt4_symbolic.pkl", "rb") as f:
    out = pickle.load(f)

num_correct = 0
gold_label = [entry["label"] for entry in data]

assert len(out) == len(gold_label)

# Cleanup bad output
prediction_list = [re.sub("\s+", "", i) for i in out]
print(prediction_list)
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
print(classification_report(labels, predictions, target_names=target_names, digits=4))
print(confusion_matrix(labels, predictions))
