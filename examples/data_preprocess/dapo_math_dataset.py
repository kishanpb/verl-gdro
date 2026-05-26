"""
Preprocess the open-r1/DAPO-Math-17k-Processed (English subset) to parquet format
under ~/data/dapo-math/{train.parquet, train_example.json}.

We mirror the schema used by other math preprocessors:
  - data_source: HF repo id
  - prompt: [{role: "user", content: <question + instruction>}]
  - ability: "math"
  - reward_model: {style: "rule", ground_truth: <final answer string>}
  - extra_info: {split, index, ... (pass-through metadata if available)}
"""

import argparse
import json
import os
from typing import Any, Dict

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


DATA_SOURCE = "open-r1/DAPO-Math-17k-Processed"
DATA_CONFIG = "en"  # English ~14.1k rows


def extract_solution_str(example: Dict[str, Any]) -> str:
    # Try common field names in DAPO/open-r1 style datasets
    # Prefer explicit final answer fields if present
    for key in [
        "final_answer",
        "answer",
        "solution",
        "output",
        "response",
    ]:
        if key in example and example[key] is not None:
            val = str(example[key])
            # If boxed answers exist, normalize to raw string
            try:
                boxed = last_boxed_only_string(val)
                if boxed:
                    return remove_boxed(boxed)
            except Exception:
                pass
            return val.strip()
    # If nothing obvious, fallback to empty (will be filtered later if needed)
    return ""


def extract_question_str(example: Dict[str, Any]) -> str:
    # Try typical fields for problem text
    for key in [
        "question",
        "problem",
        "input",
        "prompt",
        "instruction",
    ]:
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/dapo-math")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    print(
        f"Loading {DATA_SOURCE} config={DATA_CONFIG} from huggingface...",
        flush=True,
    )
    dataset_dict = datasets.load_dataset(
        DATA_SOURCE, DATA_CONFIG, trust_remote_code=True
    )

    # Many HF datasets expose a single split called "train"; handle others defensively
    split_name = "train" if "train" in dataset_dict else list(dataset_dict.keys())[0]
    train_dataset = dataset_dict[split_name]

    instruction = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    def map_fn(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = extract_question_str(ex)
        answer = extract_solution_str(ex)
        # Append instruction if not already included
        question_full = (question + " " + instruction).strip()

        # Pass through optional metadata if present (no level in DAPO EN subset)
        subject = ex.get("type", None) or ex.get("subject", None)

        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": question_full}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": idx,
                "type": subject,
                "config": DATA_CONFIG,
            },
        }

    mapped = train_dataset.map(function=map_fn, with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    mapped.to_parquet(os.path.join(local_dir, "train.parquet"))

    # Save one example as JSON for reference
    if len(mapped) > 0:
        ex = mapped[0]
        with open(os.path.join(local_dir, "train_example.json"), "w") as f:
            json.dump(ex, f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


