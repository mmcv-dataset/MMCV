# MMFC

---

[Installation](#installation) | [To Run](#to-run) | [Setup](#setup)

This repository contains code and data for the paper: _Piecing It All Together: Verifying Multi-Hop Multimodal Claims._

**Note:** Some files and data are removed for annoymous purposes and will be released upon acceptance.

## Installation

Requires Python 3.9 to run.

Install conda environment from `environment.yml` file.

```sh
conda env create -n MMFC --file environment.yml
conda activate MMFC
```

## Setup

**Step1:** Please create .env file and set your API key:

```sh
OPENAI_API_KEY="YOUR KEY"
GEMINI_API_KEY="YOUR KEY"
```

**Step2:** This script will create and download all raw data to directory called `MMQA_Raw`.

```sh
python download_raw.py
```

**Step3:** Run the following script to download raw image files. Then, please unzip it and put it under `MMQA_Raw`. The path will be `MMQA_Raw/final_dataset_images`.

```sh
sh download_images.sh
```

## To Run

To run claim generationa and refinement:

```sh
python data_collection_pipeline.py
python assemble.py
```

To run the negation pipeline:

```sh
python negation_pipeline.py
```

To run MLLM experiments:

```sh
python mllm_exp.py
python evaluation.py
```
