"""
Aggregate multiple math/science evaluation benchmarks into standardized parquet files:
  - AIME (combine 2024 and 2025)
  - AMC
  - MATH (LightEval mirror)
  - Minerva
  - OlympiadBench
  - GPQA (OOD)

Each dataset is written under:
  <local_root>/<name>/test.parquet and <local_root>/<name>/test_example.json

Schema mirrors existing math preprocessors so pass@k and Prompt-GDRO/Rollout-GDRO pipelines work:
  - data_source: HF repo id
  - prompt: [{role: "user", content: question + instruction}]
  - ability: "math" (or "science" for GPQA if detected)
  - reward_model:
      * style: "rule", ground_truth: <answer string> (math-style)
      * or style: "model", eval: "multiple_choice", ground_truth: int, choices: [str]
  - extra_info: {split: "test", index: int, ... (optional passthrough)}

Notes:
  - Provide dataset IDs via CLI. Defaults known/stable for MATH only.
  - AIME is built by concatenating two provided IDs/configs (e.g., 2024/2025).
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import datasets


INSTRUCTION_MATH = "Let's think step by step and output the final answer within \\boxed{}."


def _extract_question(example: Dict[str, Any]) -> str:
    for k in ["question", "problem", "input", "prompt", "instruction", "query", "stem"]:
        if k in example and example[k] is not None:
            return str(example[k]).strip()
    return ""


def _extract_answer_text(example: Dict[str, Any]) -> str:
    for k in ["final_answer", "answer", "solution", "output", "response", "label"]:
        if k in example and example[k] is not None:
            v = example[k]
            # Normalize list-like answers by taking the first element
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[0]).strip()
            return str(v).strip()
    return ""


def _extract_mc(example: Dict[str, Any]) -> Optional[Tuple[List[str], Optional[int]]]:
    # Try common patterns for multiple choice
    # choices: list[str]; label/index: int or str convertible
    for ck in ["choices", "options", "endings"]:
        if ck in example and isinstance(example[ck], (list, tuple)):
            choices = [str(c) for c in example[ck]]
            # gold index
            for lk in ["label", "answer_index", "gold", "correct", "answer"]:
                if lk in example and example[lk] is not None:
                    try:
                        gi = int(example[lk])
                    except Exception:
                        # sometimes labels are letters like "A".."D"
                        s = str(example[lk]).strip().upper()
                        if s in ["A", "B", "C", "D", "E"]:
                            gi = ["A", "B", "C", "D", "E"].index(s)
                        else:
                            gi = None
                    return choices, gi
            return choices, None
    return None


def map_math_like(ds: datasets.Dataset, data_source: str, split: str) -> datasets.Dataset:
    def fn(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        q = _extract_question(ex)
        a_txt = _extract_answer_text(ex)
        prompt = (q + " " + INSTRUCTION_MATH).strip()
        # Optional passthroughs similar to DAPO training schema
        level = ex.get("level", None)
        subject = ex.get("type", None) or ex.get("subject", None)
        # Try MC first; if present, use MC style, else rule style
        mc = _extract_mc(ex)
        if mc is not None and any(len(c) > 0 for c in mc[0]):
            choices, gi = mc
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {
                    "style": "model",
                    "eval": "multiple_choice",
                    "ground_truth": (gi if gi is not None else 0),
                    "choices": choices,
                },
                "extra_info": {"split": split, "index": idx, "level": level, "type": subject},
            }
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": a_txt},
            "extra_info": {"split": split, "index": idx, "level": level, "type": subject},
        }

    # Remove original HF columns so output schema matches training schema exactly
    return ds.map(function=fn, with_indices=True, remove_columns=ds.column_names)


def map_gpqa_diamond(ds: datasets.Dataset, data_source: str, split: str) -> datasets.Dataset:
    # Template adapted from TIGER-AI General-Reasoner gpqa_eval_qwen.py
    GPQA_QUERY_TEMPLATE = (
        "{Question}\n\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
        "Please only provide the letter of the answer in the box."
    )

    def fn(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Shuffle incorrect answers and insert the correct option at a random index, mirroring recipe/r1/data_process.py
        import random

        choices = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, ex["Correct Answer"])

        prompt = GPQA_QUERY_TEMPLATE.format(
            Question=str(ex.get("Question", "")).strip(),
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        gold_letter = "ABCD"[gold_index]

        # Use rule style with ground_truth as the gold letter; scorer extracts boxed letter (or "Answer: X" fallback)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "science",
            "reward_model": {"style": "rule", "ground_truth": gold_letter},
            "extra_info": {"split": split, "index": idx, "type": ex.get("domain", None)},
        }

    return ds.map(function=fn, with_indices=True, remove_columns=ds.column_names)


def load_split(dataset_id: str, config: Optional[str], split: Optional[str]) -> datasets.Dataset:
    if config is not None and len(config) > 0:
        dsd = datasets.load_dataset(dataset_id, config, trust_remote_code=True)
    else:
        dsd = datasets.load_dataset(dataset_id, trust_remote_code=True)
    # Normalize split: accept 'train' or 'test' but we always save as test later
    norm_split = None
    if isinstance(split, str) and len(split) > 0:
        s = split.lower()
        if s in ["train", "training"]:
            norm_split = "train" if "train" in dsd else None
        elif s in ["test", "testing"]:
            norm_split = "test" if "test" in dsd else None
        else:
            norm_split = s if s in dsd else None
    if norm_split is None:
        # Prefer test -> validation -> first available
        for s in ["test", "validation", "dev", "eval", "val", "train"]:
            if s in dsd:
                norm_split = s
                break
    return dsd[norm_split] if norm_split is not None else dsd[list(dsd.keys())[0]]


def write_parquet(ds: datasets.Dataset, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.parquet")
    ds.to_parquet(out_path)
    if len(ds) > 0:
        with open(os.path.join(out_dir, "test_example.json"), "w") as f:
            json.dump(ds[0], f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_root", default="~/data/math-dapo-style")
    # Dataset IDs/configs/splits are configurable; provide only what you know.
    # Default to the requested HF IDs
    ap.add_argument("--math_id", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--math_config", default=None)
    ap.add_argument("--math_split", default="test")

    ap.add_argument("--aime24_id", default="HuggingFaceH4/aime_2024")
    ap.add_argument("--aime24_config", default=None)
    ap.add_argument("--aime24_split", default="test")
    ap.add_argument("--aime25_id", default="MathArena/aime_2025")
    ap.add_argument("--aime25_config", default=None)
    ap.add_argument("--aime25_split", default="test")

    ap.add_argument("--amc_id", default="math-ai/amc23")
    ap.add_argument("--amc_config", default=None)
    ap.add_argument("--amc_split", default="test")

    ap.add_argument("--minerva_id", default="math-ai/minervamath")
    ap.add_argument("--minerva_config", default=None)
    ap.add_argument("--minerva_split", default="test")

    ap.add_argument("--olympiad_id", default="math-ai/olympiadbench")
    ap.add_argument("--olympiad_config", default=None)
    ap.add_argument("--olympiad_split", default="test")

    ap.add_argument("--gpqa_id", default="Idavidrein/gpqa")
    ap.add_argument("--gpqa_config", default="gpqa_diamond")
    ap.add_argument("--gpqa_split", default="test")

    args = ap.parse_args()

    root = os.path.expanduser(args.local_root)

    # MATH (always if provided)
    if args.math_id:
        ds_math = load_split(args.math_id, args.math_config, args.math_split)
        ds_math_mapped = map_math_like(
            ds_math,
            data_source=f"{args.math_id}:{args.math_config}" if args.math_config else args.math_id,
            split="test",
        )
        write_parquet(ds_math_mapped, os.path.join(root, "MATH-500"), name="MATH-500")

    # AIME: save 2024 and 2025 separately to avoid schema alignment issues
    if args.aime24_id:
        ds_aime24 = load_split(args.aime24_id, args.aime24_config, args.aime24_split)
        ds_aime24_mapped = map_math_like(ds_aime24, data_source=args.aime24_id, split="test")
        write_parquet(ds_aime24_mapped, os.path.join(root, "AIME-2024"), name="AIME-2024")
    if args.aime25_id:
        ds_aime25 = load_split(args.aime25_id, args.aime25_config, args.aime25_split)
        ds_aime25_mapped = map_math_like(ds_aime25, data_source=args.aime25_id, split="test")
        write_parquet(ds_aime25_mapped, os.path.join(root, "AIME-2025"), name="AIME-2025")

    # AMC
    if args.amc_id:
        ds_amc = load_split(args.amc_id, args.amc_config, args.amc_split)
        ds_amc_mapped = map_math_like(ds_amc, data_source=args.amc_id, split="test")
        write_parquet(ds_amc_mapped, os.path.join(root, "AMC"), name="AMC")

    # Minerva
    if args.minerva_id:
        ds_minerva = load_split(args.minerva_id, args.minerva_config, args.minerva_split)
        ds_minerva_mapped = map_math_like(ds_minerva, data_source=args.minerva_id, split="test")
        write_parquet(ds_minerva_mapped, os.path.join(root, "Minerva"), name="Minerva")

    # OlympiadBench
    if args.olympiad_id:
        ds_olymp = load_split(args.olympiad_id, args.olympiad_config, args.olympiad_split)
        ds_olymp_mapped = map_math_like(ds_olymp, data_source=args.olympiad_id, split="test")
        write_parquet(ds_olymp_mapped, os.path.join(root, "OlympiadBench"), name="OlympiadBench")

    # GPQA (OOD)
    if args.gpqa_id:
        ds_gpqa = load_split(args.gpqa_id, args.gpqa_config, args.gpqa_split)
        ds_gpqa_mapped = map_gpqa_diamond(ds_gpqa, data_source=args.gpqa_id, split="test")
        write_parquet(ds_gpqa_mapped, os.path.join(root, "GPQA"), name="GPQA")
