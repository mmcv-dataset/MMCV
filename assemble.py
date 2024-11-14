import os
import json
import pandas as pd


def load_raw_json(file_path):
    with open(file_path, "r") as f:
        json_list = list(f)
    df_list = []
    df = pd.DataFrame()
    for json_str in json_list:
        entry = json.loads(json_str)
        df_list.append(pd.json_normalize(entry))
    df = pd.concat(df_list, ignore_index=True)
    return df


def write_json(df, file_name):
    out_list = []
    for i in range(len(df)):
        out_list.append(
            {
                "claim": df.iloc[i]["claim"],
                "text_evidence": df.iloc[i]["text_evidence"],
                "image_evidence": df.iloc[i]["image_evidence"],
                "table_evidence": df.iloc[i]["table_evidence"],
            }
        )
    with open(file_name, "w") as fp:
        json.dump(out_list, fp)


if __name__ == "__main__":
    raw_df = load_raw_json("MMQA_Raw/MMQA_dev.jsonl")
    claim_df = load_raw_json("Output/dev_2024-08-31T23:16:29.008107.jsonl")
    assert len(raw_df) == len(claim_df)
    hops_count = {}
    total_supported = 0

    text_context, image_context, table_context = [], [], []
    for i in range(len(raw_df)):
        text_temp, image_temp, table_temp = [], [], []
        supporting_context = raw_df.iloc[i]["supporting_context"]
        num_hops = len(supporting_context)
        hops_count[num_hops] = hops_count.get(num_hops, 0) + 1
        total_supported += 1
        for c in raw_df.iloc[i]["supporting_context"]:
            if c["doc_part"] == "text":
                text_temp.append(c["doc_id"])
            if c["doc_part"] == "image":
                image_temp.append(c["doc_id"])
            if c["doc_part"] == "table":
                table_temp.append(c["doc_id"])
        text_context.append(text_temp)
        image_context.append(image_temp)
        table_context.append(table_temp)
    print(hops_count)

    claim_list = list(claim_df["final_claim"])
    assert (
        len(claim_list) == len(text_context) == len(image_context) == len(table_context)
    )
    final_df = pd.DataFrame(
        list(zip(claim_list, text_context, image_context, table_context)),
        columns=["claim", "text_evidence", "image_evidence", "table_evidence"],
    )
    if not os.path.exists("Final"):
        os.makedirs("Final")
    write_json(final_df, "Final/dev.json")
