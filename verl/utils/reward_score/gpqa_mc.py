# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
Multiple-choice GPQA scorer.

Supports two common patterns in model outputs:
 - Boxed letter: \\boxed{A|B|C|D}
 - Answer line:  Answer: A|B|C|D (case-insensitive, optional $ around letter)
"""

import re
from typing import Union


_PATTERN_BOXED = re.compile(r"\\boxed\{\s*([A-D])\s*\}", flags=re.IGNORECASE)
_PATTERN_ANSWER = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def compute_score(solution_str: str, ground_truth: Union[str, int]) -> float:
    match = _PATTERN_BOXED.search(solution_str)
    extracted = match.group(1) if match else None
    if extracted is None:
        m2 = _PATTERN_ANSWER.search(solution_str)
        extracted = m2.group(1) if m2 else None
    if isinstance(ground_truth, int):
        # Convert numeric index to letter if ever provided
        gold = "ABCD"[ground_truth] if 0 <= ground_truth < 4 else None
    else:
        gold = str(ground_truth).strip().upper() if ground_truth is not None else None
    return 1.0 if (extracted is not None and gold is not None and extracted.upper() == gold) else 0.0
