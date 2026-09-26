# Code/data_code.py — task data module for the code task (pissa-dataset
# python split -> HumanEval/MBPP eval via EvalPlus).  Loaded by FILE
# PATH via utils.train_eval_shared.load_data_module
# ("Code/data_code.py"), never as a package import.
#
# The pipeline is a VERBATIM port of the old train_eval_shared_code.py:
# prompt_template="none" (the pissa python split's "instruction" column
# is ALREADY a fully-rendered Alpaca prompt — re-wrapping would
# reintroduce the train/inference double-wrap bug; see prompt_utils.py),
# response-only loss masking via source-length prefix, target =
# f"{output}\n{eos}".  S20 pins the template bytes and the "none"
# default.
#
# Seam contract (see utils.train_eval_shared.load_data_module):
#   build_train_data(tokenizer, max_seq_len, data_seed,
#                    max_train_samples, shuffle_data)
#       -> (train_dataset, collator, data_meta)
# No load_task_tokenizer hook: code uses the shared default tokenizer
# (pad=eos, right padding) exactly as the old code trainer did;
# generation-side LEFT padding is generate.py's own concern.
#
# Task knobs ride on env vars so the seam signature stays frozen:
#   DATA_CODE_PATH        (default fxmeng/pissa-dataset)
#   CODE_SUB_TASK         (default "python"; comma list; "dir:count" caps)
#   CODE_SPLIT            (default "train")
#   CODE_PROMPT_TEMPLATE  (default "none")

from __future__ import annotations

import importlib.util as _ilu
import os
from typing import Optional

import numpy as np
import torch

IGNORE_INDEX = -100
DEFAULT_DATA_PATH = "fxmeng/pissa-dataset"
DEFAULT_SUB_TASK = "python"
DEFAULT_SPLIT = "train"
DEFAULT_PROMPT_TEMPLATE = "none"
DATASET_FIELD = ("instruction", "output")

# prompt_utils lives beside this file; this module is itself loaded by
# file path, so load the template helper the same way (single source of
# truth with generate.py — no duplicated template strings).
_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "code_prompt_utils", os.path.join(_here, "prompt_utils.py"))
_pu = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pu)
format_prompt = _pu.format_prompt
ALPACA_TEMPLATE = _pu.ALPACA_TEMPLATE


class CodeInstructDataset(torch.utils.data.Dataset):
    """pissa-dataset, response-only loss masking (verbatim logic)."""

    def __init__(self, examples, tokenizer, max_len, prompt_template):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        qf, rf = DATASET_FIELD
        source = format_prompt(ex[qf], self.prompt_template)
        target = f"{ex[rf]}\n{self.tokenizer.eos_token}"
        full_enc = self.tokenizer(source + target, truncation=True,
                                  max_length=self.max_len, padding=False,
                                  add_special_tokens=True)
        src_enc = self.tokenizer(source, truncation=True,
                                 max_length=self.max_len, padding=False,
                                 add_special_tokens=True)
        ids = full_enc["input_ids"]
        sl = min(len(src_enc["input_ids"]), len(ids))
        return {"input_ids": ids,
                "attention_mask": full_enc["attention_mask"],
                "labels": ([IGNORE_INDEX] * sl + ids[sl:])[:len(ids)]}


class PaddingCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        from torch.nn.utils.rnn import pad_sequence
        out = {}
        for k in features[0]:
            seqs = [torch.tensor(f[k], dtype=torch.long) for f in features]
            out[k] = pad_sequence(
                seqs, batch_first=True,
                padding_value=IGNORE_INDEX if k == "labels" else self.pad_id)
        return out


def build_train_data(tokenizer, max_seq_len: int, data_seed: int,
                     max_train_samples: Optional[int] = None,
                     shuffle_data: bool = True):
    """Seam entry point.  Loads the pissa-dataset sub_task(s), shuffles
    by data_seed, optionally truncates, and returns the masked dataset +
    padding collator + meta (mirrors the math module's shape)."""
    from datasets import concatenate_datasets, load_dataset

    data_path = os.environ.get("DATA_CODE_PATH", DEFAULT_DATA_PATH)
    sub_task = [t for t in os.environ.get("CODE_SUB_TASK",
                                          DEFAULT_SUB_TASK).split(",") if t]
    split = os.environ.get("CODE_SPLIT", DEFAULT_SPLIT)
    prompt_template = os.environ.get("CODE_PROMPT_TEMPLATE",
                                     DEFAULT_PROMPT_TEMPLATE)

    parts = []
    for task in sub_task:
        if ":" in task:
            dd, num = task.split(":")
            parts.append(load_dataset(data_path, data_dir=dd,
                                      split=f"{split}[:{num}]"))
        else:
            parts.append(load_dataset(data_path, data_dir=task, split=split))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    n = len(ds)
    idx = np.arange(n)
    if shuffle_data:
        np.random.default_rng(data_seed).shuffle(idx)
    if max_train_samples is not None:
        idx = idx[:int(max_train_samples)]
    examples = [ds[int(i)] for i in idx]

    collator = PaddingCollator(tokenizer.pad_token_id
                               or tokenizer.eos_token_id)
    meta = {"dataset_id": data_path, "sub_task": sub_task,
            "dataset_split": split, "prompt_template": prompt_template,
            "dataset_total_rows": n, "train_rows": len(examples),
            "shuffle_data": shuffle_data, "data_seed": data_seed,
            "max_train_samples": max_train_samples,
            "max_seq_len": int(max_seq_len)}
    return (CodeInstructDataset(examples, tokenizer, int(max_seq_len),
                                prompt_template),
            collator, meta)
