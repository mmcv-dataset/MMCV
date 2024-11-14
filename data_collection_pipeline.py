import json
import pandas as pd
import os
import logging
import wikipedia

from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from datetime import datetime
from random import randrange
from dotenv import load_dotenv

import get_api_res
from in_context_example import *

load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
VERBOSE = False
TIMESTAMP = datetime.now().isoformat()

logging.basicConfig(
    filename="mmcv_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def load_raw_json(file_path):
    with open(file_path, "r") as f:
        json_list = list(f)

    df_list = []
    df = pd.DataFrame()
    for json_str in json_list:
        entry = json.loads(json_str)
        df_list.append(pd.json_normalize(entry))
    df = pd.concat(df_list, ignore_index=True)
    # print(df.columns)
    return df


def preprocess(df):
    combined_data, wiki_titles_question, wiki_titles_answer = [], [], []
    for i in range(len(df)):
        question = df.iloc[i]["question"]
        answer = df.iloc[i]["answers"][0]["answer"]
        combined = f"Question: {question} Answer: {answer}"
        combined_data.append(combined)

        if len(df.iloc[i]["metadata.wiki_entities_in_question"]) != 0:
            temp = []
            for entity in df.iloc[i]["metadata.wiki_entities_in_question"]:
                if isinstance(entity, dict):
                    temp.append(entity.get("wiki_title"))
            wiki_titles_question.append(temp)
        else:
            wiki_titles_question.append([])

        if len(df.iloc[i]["metadata.wiki_entities_in_answers"]):
            temp = []
            for entity in df.iloc[i]["metadata.wiki_entities_in_answers"]:
                if isinstance(entity, dict):
                    temp.append(entity.get("wiki_title"))
            wiki_titles_answer.append(temp)
        else:
            wiki_titles_answer.append([])

    return combined_data, wiki_titles_question, wiki_titles_answer


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def claim_from_questions(questions):
    system_prompt = (
        "You are an expert in converting question-answers into claims. For example:\n"
    ) + claim_rewrite_example[randrange(2)]
    prompt = (
        "Convert the question-answer into claim. Return only the claim and nothing else. \n"
    ) + questions
    response = get_api_res.get_openai_text_response(
        prompt, "gpt-4o-mini", system_prompt=system_prompt, temperature=0.0
    )
    return response


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def claim_modification(general_claim, wiki_question, wiki_answer):
    system_prompt = (
        (
            "Generate a multi-hop specific claim based on the given general claim and Wikipedia context. The specific claim should: \
            Incorporate information from Wikipedia context. \
            Provided context should always be factually correct. \
            Obscure key information by: \
                a) Replacing one or two central entities with related fact using the Wikipedia context. \
                b) Alluding to critical details without explicitly stating them. Claim should be short and concise. For example:\n"
        )
        + claim_modification_example[randrange(2)]
    )

    try:
        if len(wiki_answer) != 0:
            wiki_context = "".join(
                [wikipedia.summary(i, sentences=5) for i in wiki_answer]
            )
        else:
            wiki_context = "".join(
                [wikipedia.summary(i, sentences=5) for i in wiki_question]
            )
        # print(wiki_context)
        prompt = f"Convert general claims into multi-hop claims by taking wikipedia context and return only multi-hop claims and nothing else. \n General Claims: {general_claim} \n Wikipedia context: {wiki_context}"
        # print(prompt)
        response = get_api_res.get_openai_text_response(
            prompt, "gpt-4o-mini", system_prompt=system_prompt, temperature=0.2
        )
        return response
    except:
        return general_claim


def claim_fluency(claim):
    system_prompt = """
    You are tasked with improving a claim focusing on three key areas: Fluency, Correctness, and Clearness. Your goal is to enhance the text while maintaining its original meaning and intent.
    Improvement Criteria:
    Fluency:
    1. Review the text for grammar, syntax, and punctuation errors.
    2. Rephrase any awkward or unnatural sentences to make the text flow more smoothly.
    3. Ensure that the text reads naturally and is easy to follow.
    Correctness:
    1. Verify the factual accuracy of the content and correct any errors.
    2. Ensure that the text adheres to the prompt's instructions.
    3. Clarify any ambiguities and correct any inconsistencies in the information presented.
    Clearness:
    1. Simplify complex sentences or ideas to make the text easier to understand.
    2. Improve the organization of ideas to enhance readability.
    3. Ensure that the message is conveyed clearly and effectively, eliminating any confusion or ambiguity.
    Final Output:
    Once you have made the necessary improvements, provide the revised text. Ensure that the improved version is more fluent, accurate, and clear than the original while preserving the original meaning and intent.
    Example Improvement:
    Original Claim: "The results of the survey was very positive, with many respondents saying that they would recommend the service to others, however, some were also mentioned issues with the customer support."
    Improved Claim: "The survey results were overwhelmingly positive, with many respondents stating they would recommend the service to others. However, some also noted issues with customer support."
    """

    prompt = f"Claim: {claim} \n Give me claim only and nothing else."

    response = get_api_res.get_openai_text_response(
        prompt, "gpt-4o-mini", system_prompt=system_prompt, temperature=0.0
    )
    return response


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def textual_feedback(claim):
    system_prompt = """
    Given the claim, determine if the claim is good enough or needs further refinement. It needs refinement if the claim lacks clarity, fluency or claim is not clear.
    Respond with 'YES' if the claim is satisfactory, or 'NO' if it needs improvement.
    """

    prompt = f"Claim: {claim} \n Determine if the claim is good enough or needs further refinement. Return in YES or NO only."

    response = get_api_res.get_openai_text_response(
        prompt, "gpt-4o-mini", system_prompt=system_prompt, temperature=0.0
    )
    return response


def prompt_diversify(validation_claim, wiki_title):
    templates = [
        f"Is it true that {validation_claim}?",
        f"Verify the following statement: {validation_claim}",
        f"What evidence supports the claim that {validation_claim}?",
    ]

    system_prompt = """
    You are an expert fact-checker. Your task is to convert the claim into one of provided templates and validate it based on wikipedia context. 
    1. Analyze the claim critically.
    2. Provide a response either SUPPORT or REFUTE.
    Remember to be objective and base your responses on factual information on wikipedia context.
    """
    try:
        wiki_context = wikipedia.summary(wiki_title, sentences=5)
        prompt = f"Convert the claim into one of provided {templates} and validate it based on wikipedia context. \n Wikipedia context: {wiki_context}. \
                         \n Always return either SUPPORT or REFUTE"
        response = get_api_res.get_openai_text_response(
            prompt, "gpt-4o-mini", system_prompt=system_prompt
        )
        return response
    except:
        return validation_claim


def save_to_jsonl(data, output_file):
    with open(output_file, "a") as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + "\n")


def get_processed_count(output_file):
    if not os.path.exists(output_file):
        return 0
    with open(output_file, "r") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    input_file = "MMQA_Raw/MMQA_dev.jsonl"

    if not os.path.exists("Output"):
        os.makedirs("Output")

    output_file = f"Output/dev_{TIMESTAMP}.jsonl"

    processed_count = get_processed_count(output_file)

    raw_df = load_raw_json(input_file)
    extracted_qa, wiki_question, wiki_answer = preprocess(raw_df)

    for i, (qa, wiki_question, wiki_answer) in tqdm(
        enumerate(
            zip(
                extracted_qa[processed_count:],
                wiki_question[processed_count:],
                wiki_answer[processed_count:],
            ),
            start=processed_count,
        ),
        total=len(extracted_qa[processed_count:]),
        desc="Processing MMCV",
    ):
        try:
            max_iterations = 3
            iteration = 0
            while iteration < max_iterations:
                if iteration == 0:
                    claim_from_llm = claim_from_questions(qa)
                else:
                    claim_from_llm = claim_from_questions(qa)
                if len(wiki_answer) > 1:
                    specific_claim_from_llm = claim_modification(
                        claim_from_llm, wiki_question, wiki_answer
                    )
                else:
                    specific_claim_from_llm = claim_from_llm
                refined_claim = claim_fluency(specific_claim_from_llm)
                text_feedback_of_claim = textual_feedback(refined_claim)
                if text_feedback_of_claim.strip().upper() == "YES":
                    break
                iteration += 1
            data_to_save = {
                "QA": qa,
                "QA_claim": claim_from_llm,
                "final_claim": refined_claim,
            }
            save_to_jsonl(data_to_save, output_file)
            if VERBOSE:
                print(
                    f"Processed and saved instance {i+1}. Final Claim: {refined_claim}"
                )
        except Exception as e:
            print(f"Error processing instance {i+1}: {str(e)}")
            logging.error(f"Error processing instance {i+1}: {str(e)}")
            continue
    print(f"Total processed instances: {get_processed_count(output_file)}")
