import json
import os
import logging

from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from datetime import datetime
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


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def textual_feedback(claim):
    system_prompt = """
    Given the true labelled claim, you need to negate it. You can use word substitution or entity substitution or temporal mutation. You can apply your own techniques as well. But make sure the claim returned must be false.
    Example:
    1. Word Subsitution:
    Original Claim: The Eiffel Tower in Paris attracts millions of visitors annually.
    Negation Claim: The Eiffel Tower in Paris houses millions of residents annually.
    2. Entity Substitution:
    Original Claim: The Eiffel Tower in Paris attracts millions of visitors annually.
    Negation Claim: The Colosseum in Paris attracts millions of visitors annually.
    3. Temporal Mutation:
    Original Claim: Since its construction in the 19th century, the Eiffel Tower has become a major Parisian attraction.
    Negation Claim: Ever since its renovation in 2050, the Eiffel Tower has been Paris's top tourist site.
    """
    prompt = f"Negate the true claim into false claim. Return only the negation claim and nothing else. Claim: {claim}"
    response = get_api_res.get_openai_text_response(
        prompt, "gpt-4o-mini", system_prompt=system_prompt, temperature=0.0
    )
    return response


def process_json_file_live(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    print(f"Total instances to process: {len(data)}")
    processed_data = []
    for item in tqdm(data, desc="Processing claims"):
        final_claim = item["final_claim"]
        item["negation_claim"] = textual_feedback(final_claim).strip()
        processed_data.append(item)
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=2)
    print(f"Processing complete. Output saved to {output_file}")


# input_file = "Final/mmcv_train_part2.json"
# output_file = "Final/neg_mmcv_train_part2.json"
# process_json_file_live(input_file, output_file)
