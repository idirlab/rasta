# math/data_math.py — task data module for the math benchmark.
# Loaded by FILE PATH via utils.train_eval_shared.load_data_module
# ("math/data_math.py"), never as a package import: the directory is
# named `math`, and Python's builtin math module owns that name.
#
# Contract (identical for every task data module):
#   build_train_data(tokenizer, max_seq_len, data_seed,
#                    max_train_samples=None, shuffle_data=True)
#     -> (torch Dataset, collator, meta_dict)
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.train_eval_shared import InstructDataset, PaddingCollator  # noqa: E402

DATASET_ID = "meta-math/MetaMathQA"

# Alpaca-style prompt; kept task-local deliberately.  eval_math.py carries
# its own copy (with the CoT trigger appended) — if you change one,
# check the other.
INSTRUCTION_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def build_train_data(
    tokenizer,
    max_seq_len: int,
    data_seed: int,
    max_train_samples: Optional[int] = None,
    shuffle_data: bool = True,
):
    """Load MetaMathQA (395K). No val split — eval done post-training via
    math/eval_math.py on GSM8K + MATH-500."""
    ds  = load_dataset(DATASET_ID, split="train")
    n   = len(ds)
    idx = np.arange(n)
    if shuffle_data:
        rng = np.random.default_rng(data_seed)
        rng.shuffle(idx)
    if max_train_samples is not None:
        idx = idx[:max_train_samples]

    examples = [ds[int(i)] for i in idx]
    collator = PaddingCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    dataset = InstructDataset(
        examples, tokenizer, max_seq_len,
        prompt_template=INSTRUCTION_PROMPT,
        instruction_fields=("query", "instruction"),
        response_fields=("response", "output"),
    )
    return (
        dataset,
        collator,
        {"dataset_id": DATASET_ID, "dataset_total_rows": n,
         "train_rows": len(examples), "shuffle_data": shuffle_data,
         "data_seed": data_seed, "max_train_samples": max_train_samples},
    )
