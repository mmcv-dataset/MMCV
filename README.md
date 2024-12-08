# Multi-Hop Multimodal Claim Verification

This repository contains code and data for the paper: _Piecing It All Together: Verifying Multi-Hop Multimodal Claims._ [Paper](https://arxiv.org/abs/2411.09547)

### Changelog

- `Nov 13, 2024` Initial release.

## MMCV Dataset

In the [Dataset](https://github.com/mmcv-dataset/MMCV/tree/main/Dataset) folder you will find the files for MMCV:

Each line of the MMCV files (e.g. `1hop.json`) contains one multi-hop claim, alongside its multimodal evidence.

```json
{
    "claim": "Each Cadet is equipped with a tool in their right hand, much like a coffeehouse serves a variety of beverages to its patrons.",
    "wiki_context": "A coffeehouse, coffee shop, or café is an establishment that serves various types of coffee, espresso, latte, americano and cappuccino. Some coffeehouses may serve cold beverages, such as iced coffee and iced tea, as well as other non-caffeinated beverages. A coffeehouse may also serve food, such as light snacks, sandwiches, muffins, cakes, breads, donuts or pastries. In continental Europe, some cafés also serve alcoholic beverages. Coffeehouses range from owner-operated small businesses to large multinational corporations.",
    "text_evidence": [],
    "image_evidence": [
        "50ccc7ab2db82b7feeaa3bbf6f533773"
    ],
    "table_evidence": [],
    "label": "SUPPORT"
}
```

The ids for `text_evidence`, `image_evidence`, and `table_evidence` corresponds to ids in [MMQA](https://github.com/allenai/multimodalqa). Please run `python download_raw.py` and `sh download_images.sh` in [Setup](#setup) to download the raw files.

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

**Step3:** Run the following script to download raw image files from [MMQA](https://github.com/allenai/multimodalqa). Then, please unzip it and put it under `MMQA_Raw`. The path will be `MMQA_Raw/final_dataset_images`.

```sh
sh download_images.sh
```

## Installation

Requires Python 3.9 to run.

Install conda environment from `environment.yml` file.

```sh
conda env create -n mmcv --file environment.yml
conda activate mmcv
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

All experiment results can be found in the [MLLM_Results](https://github.com/mmcv-dataset/MMCV/tree/main/MLLM_Results) folder.

## Citation

```text
@inproceedings{wang2024piecing,
  title={Piecing It All Together: Verifying Multi-Hop Multimodal Claims},
  author={Haoran Wang and Aman Rangapur and Xiongxiao Xu and Yueqing Liang and Haroon Gharwi and Carl Yang and Kai Shu.},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  year={2025}
}
```

## License

The MMCV dataset is distribued under the [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/legalcode) license.
