# utils/train_eval_shared.py — GENERIC training/eval utilities shared by
# every task and method.  Nothing in this file may reference a specific
# dataset: task specifics (dataset id, prompt template, field names) live
# in each task's data module (e.g. Math/data_math.py), loaded by FILE
# PATH via load_data_module() below.
#
# WHY file-path loading and not `import math.data_math`: the math task
# directory is named `math/`, and Python's builtin `math` module always
# wins that name (builtins precede sys.path), so `math.` can never be a
# package prefix.  Path-based loading sidesteps the clash and keeps task
# directories importable-by-contract rather than by package machinery.
from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_eval_shared")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ALIASES = {
    "llama2-7b":  "meta-llama/Llama-2-7b-hf",
    "gemma-7b":   "google/gemma-7b",
    "gemma-2b":   "google/gemma-2b",
    "llama3-8b":  "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen25-7b":  "Qwen/Qwen2.5-7B",
}


def resolve_model_name(alias: str) -> str:
    return MODEL_ALIASES.get(alias, alias)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def distributed_barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


# ── Task data-module seam ─────────────────────────────────────────────────────

def load_data_module(spec: str):
    """Load a task data module by file path (absolute, or relative to the
    repo root), e.g. spec="Math/data_math.py".  The module must expose
    build_train_data(tokenizer, max_seq_len, data_seed, max_train_samples,
    shuffle_data) -> (Dataset, collator, meta_dict) — the exact contract
    the trainers consume."""
    path = spec if os.path.isabs(spec) else os.path.join(REPO_ROOT, spec)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"data_module {spec!r} not found (resolved to {path}). Paths "
            f"are relative to the repo root: {REPO_ROOT}")
    mod_name = "task_data_" + re.sub(r"[^A-Za-z0-9_]", "_", spec)
    spec_ = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    if not hasattr(mod, "build_train_data"):
        raise AttributeError(
            f"data_module {spec!r} lacks build_train_data(); every task "
            f"data module must implement it with the shared signature")
    return mod


# ── Model / tokenizer loading ─────────────────────────────────────────────────

def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token    = tok.eos_token
    return tok


_ATTN_LOGGED: set = set()


def load_base_model(model_name: str, torch_dtype=torch.bfloat16,
                    use_fast_attn: bool = True,
                    use_torch_sdpa: Optional[bool] = None):
    """Load with the fastest working attention backend:
    flash_attention_2 → sdpa → library default.  The engaged backend is
    logged once per model so a silent fallback is never invisible.

    use_torch_sdpa is the legacy name for use_fast_attn (kept so old
    configs keep working); when given it wins.
    """
    if use_torch_sdpa is not None:
        use_fast_attn = use_torch_sdpa
    impls: List[Optional[str]]
    if use_fast_attn and torch.cuda.is_available():
        impls = ["flash_attention_2", "sdpa", None]
    elif use_fast_attn:
        impls = ["sdpa", None]           # FA2 is CUDA-only
    else:
        impls = [None]
    last_exc = None
    for attn in impls:
        try:
            # use_safetensors=True: on torch<2.6, transformers refuses to
            # torch.load a .bin checkpoint (CVE-2025-32434 gate) even when
            # the repo ALSO ships safetensors, because from_pretrained does
            # not reliably prefer the .safetensors file on its own. Every
            # model this project loads (Mistral-7B, Llama-2-7b, Llama-3-8B,
            # and safetensors-only smoke fixtures) ships safetensors, so
            # forcing it is safe; a model that genuinely lacked them would
            # now fail with a clear "no safetensors" error rather than the
            # opaque CVE message.
            kwargs = {"use_safetensors": True}
            if attn is not None:
                kwargs["attn_implementation"] = attn
            try:
                m = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch_dtype, **kwargs)
            except TypeError:  # older transformers kwarg name
                m = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype, **kwargs)
            tag = attn or "default"
            if model_name not in _ATTN_LOGGED:
                _ATTN_LOGGED.add(model_name)
                logger.info(f"[attn] {model_name}: {tag}")
            return m
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load {model_name}: {last_exc}")


def discover_transformer_linear_modules(model: nn.Module) -> List[str]:
    discovered = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and "layers." in name
        and "lm_head" not in name
    ]
    if not discovered:
        raise RuntimeError("No transformer linear layers found under layers.*.")
    return sorted(discovered)


def linear_module_suffixes(module_names: List[str]) -> List[str]:
    return sorted({name.split(".")[-1] for name in module_names})


# ── Generic instruction dataset + collator ────────────────────────────────────

class InstructDataset(torch.utils.data.Dataset):
    """Instruction-tuning dataset with response-only loss masking.

    Task-agnostic: the prompt template and the example field names are
    constructor arguments; task data modules bind them (MetaMathQA uses
    query/response, Alpaca-style sets use instruction/output, etc.).
    instruction_fields / response_fields are tried in order — first
    present key wins."""

    def __init__(self, examples, tokenizer, max_len: int,
                 prompt_template: str,
                 instruction_fields: Sequence[str] = ("query", "instruction"),
                 response_fields: Sequence[str] = ("response", "output")):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.prompt_template = prompt_template
        self.instruction_fields = tuple(instruction_fields)
        self.response_fields = tuple(response_fields)

    def __len__(self):
        return len(self.examples)

    def _pick(self, ex, fields) -> str:
        for f in fields:
            if f in ex:
                return ex[f]
        return ""

    def __getitem__(self, idx):
        ex          = self.examples[idx]
        instruction = self._pick(ex, self.instruction_fields)
        response    = self._pick(ex, self.response_fields)
        prompt      = self.prompt_template.format(instruction=instruction)
        full_text   = prompt + response + self.tokenizer.eos_token

        full_enc   = self.tokenizer(full_text, truncation=True, max_length=self.max_len,
                                    padding=False, add_special_tokens=True)
        prompt_enc = self.tokenizer(prompt,    truncation=True, max_length=self.max_len,
                                    padding=False, add_special_tokens=True)

        input_ids      = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len     = min(len(prompt_enc["input_ids"]), len(input_ids))
        labels         = [-100] * prompt_len + input_ids[prompt_len:]
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels[:len(input_ids)],
        }


class PaddingCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        from torch.nn.utils.rnn import pad_sequence
        batch = {}
        for key in features[0].keys():
            seqs    = [torch.tensor(f[key], dtype=torch.long) for f in features]
            pad_val = -100 if key == "labels" else self.pad_id
            batch[key] = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        return batch


# ── Loss recording / misc ─────────────────────────────────────────────────────

class TrainLossStore:
    """Records training loss at each logging step for later plotting."""

    def __init__(self):
        self.records: List[Dict] = []

    def on_log(self, step: int, epoch: float, logs: Dict):
        if "loss" in logs:
            self.records.append({
                "step":          step,
                "epoch":         round(epoch or 0.0, 6),
                "train_loss":    logs["loss"],
                "grad_norm":     logs.get("grad_norm", float("nan")),
                "learning_rate": logs.get("learning_rate", float("nan")),
            })

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)


def make_loss_callback(store: TrainLossStore):
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                store.on_log(step=state.global_step, epoch=state.epoch or 0.0, logs=logs)

    return _CB()


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_run_metadata(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def get_env_metadata() -> Dict:
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "distributed": is_distributed(),
        "world_size": dist.get_world_size() if is_distributed() else 1,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


# ── Run-name helpers ──────────────────────────────────────────────────────────

def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def _fmt_drop(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def make_lora_run_name(
    model_alias: str, rank: int, lora_alpha: int, lr: float,
    seed: int, lora_dropout: float, use_dora: bool,
    max_train_samples: Optional[int],
) -> str:
    method = "dora" if use_dora else "lora"
    return "__".join([
        method, model_alias.replace("/", "_"),
        f"r{rank}", f"a{lora_alpha}", f"lr{_fmt_lr(lr)}",
        f"seed{seed}", f"drop{_fmt_drop(lora_dropout)}",
        f"n{max_train_samples}" if max_train_samples is not None else "nfull",
    ])
