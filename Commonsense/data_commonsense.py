# Commonsense/data_commonsense.py — task data module for commonsense_170k.
# Loaded by FILE PATH via utils.train_eval_shared.load_data_module
# ("Commonsense/data_commonsense.py"), never as a package import.
#
# The prompt template, tokenization, response-only masking, pad_token_id=0
# and LEFT padding are ported VERBATIM from the original
# train_rasta_commonsense.py / train.py lineage.  Do NOT "clean up" the
# indentation or trailing whitespace inside generate_prompt — the eval
# side (Commonsense/eval_commonsense.py) renders the matching prefix and
# any byte drift desyncs train vs eval and RaSTA vs the baselines.
# S18 in utils/sanity_check.py pins the exact bytes of both.
#
# Seam contract (see utils.train_eval_shared.load_data_module):
#   build_train_data(tokenizer, max_seq_len, data_seed,
#                    max_train_samples, shuffle_data)
#       -> (train_dataset, collator, data_meta)
#   load_task_tokenizer(model_name)   [optional hook, provided here]
#
# max_seq_len maps onto the original cutoff_len (drivers pass 256).
# The 170k json path comes from $DATA_CS (default below) so the seam
# signature stays frozen across tasks.

from __future__ import annotations

import os
from typing import Optional

DEFAULT_DATA_CS = "Commonsense/data/commonsense/commonsense_170k.json"


# ── Prompt: VERBATIM from train.py / train_rasta_commonsense.py ──────────────

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


def load_task_tokenizer(model_name: str):
    """pad_token_id=0 (unk, deliberately != eos) and LEFT padding,
    matching the original train.py / commonsense_evaluate.py exactly;
    saved into merged/ at save time so the evaluator loads a
    byte-identical tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token_id = 0
    tok.padding_side = "left"
    return tok


# ── Tokenization + response-only masking (VERBATIM logic) ────────────────────

def _tokenize(tokenizer, prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len,
                       padding=False, return_tensors=None)
    if (result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def _gen_tok(tokenizer, data_point, cutoff_len, train_on_inputs):
    tokenized = _tokenize(tokenizer, generate_prompt(data_point), cutoff_len)
    if not train_on_inputs:
        user_len = len(_tokenize(tokenizer,
                                 generate_prompt({**data_point, "output": ""}),
                                 cutoff_len,
                                 add_eos_token=False)["input_ids"])
        tokenized["labels"] = ([-100] * user_len
                               + tokenized["labels"][user_len:])
    return tokenized


def build_train_data(tokenizer, max_seq_len: int, data_seed: int,
                     max_train_samples: Optional[int] = None,
                     shuffle_data: bool = True):
    """Seam entry point.  max_seq_len == the original cutoff_len (256 in
    the unified protocol).  train_on_inputs=False fixed (the original
    default; response-only loss).  No val split — eval is post-training
    on the 8 test sets."""
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq

    data_path = os.environ.get("DATA_CS", DEFAULT_DATA_CS)
    if data_path.endswith((".json", ".jsonl")):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    total_rows = len(data["train"])
    if max_train_samples is not None:
        n = min(int(max_train_samples), total_rows)
        data["train"] = data["train"].select(range(n))

    cols = data["train"].column_names
    ds = data["train"]
    if shuffle_data:
        ds = ds.shuffle(seed=data_seed)
    ds = ds.map(lambda dp: _gen_tok(tokenizer, dp, int(max_seq_len), False),
                num_proc=8, remove_columns=cols)

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                      return_tensors="pt", padding=True)
    meta = {"dataset_id": data_path, "dataset_total_rows": total_rows,
            "train_rows": len(ds), "shuffle_data": shuffle_data,
            "data_seed": data_seed, "max_train_samples": max_train_samples,
            "cutoff_len": int(max_seq_len), "train_on_inputs": False}
    return ds, collator, meta
