import os
import requests
import gzip
import shutil


def download_datasets(url, filename):
    response = requests.get(url)
    with open(f"MMQA_Raw/{filename}", mode="wb") as f:
        f.write(response.content)


def unzip(dir_name):
    for filename in os.listdir(dir_name):
        if filename.endswith(".gz"):
            with gzip.open(f"{dir_name}{filename}", "rb") as f_in:
                with open(f"{dir_name}{filename[:-3]}", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    download_config = {
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_train.jsonl.gz": "MMQA_train.jsonl.gz",
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_dev.jsonl.gz": "MMQA_dev.jsonl.gz",
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_test.jsonl.gz": "MMQA_test.jsonl.gz",
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_texts.jsonl.gz": "MMQA_texts.jsonl.gz",
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_images.jsonl.gz": "MMQA_images.jsonl.gz",
        "https://github.com/allenai/multimodalqa/raw/master/dataset/MMQA_tables.jsonl.gz": "MMQA_tables.jsonl.gz",
    }

    if not os.path.exists("MMQA_Raw/"):
        os.makedirs("MMQA_Raw")

    for url, filename in download_config.items():
        download_datasets(url, filename)

    unzip("MMQA_Raw/")
