#!/usr/bin/env python3
"""
Glue/train_glue.py — RoBERTa-large GLUE trainer for ALL methods
(rasta-dm, rasta-hs, lora-xs, lora, vera) behind one --method-style
config, sequence-classification edition.  Single process, single GPU
(roberta-large fp32 @ seq 128 / batch 32 is small); no torchrun.

Locked protocol (see run header docs / paper appendix):
  * roberta-large, fp32, max_seq_len 128, batch 32, AdamW,
    linear schedule + warmup 0.06, weight decay 0.01;
  * per-task epochs: sst2/qnli/stsb/mnli 10, mrpc/cola/rte/qqp 20
    (LoRA paper Table 9 lineage);
  * adapters on query+value projections only (all 24 layers);
  * the classification head (fresh init, identical across methods) is
    fully trainable in EVERY run at a FIXED head lr (default 5e-4) that
    is never searched; only the adapter lr is selected.  Head params
    are excluded from reported adapter param counts;
  * eval on the dev set every epoch; the run's score is the BEST-epoch
    dev metric (matthews for cola, pearson for stsb, accuracy
    otherwise; mnli = mean of matched/mismatched accuracy, both also
    reported);
  * training and eval are one process: this script writes
    eval_results.json itself (results.<task>.accuracy carries the
    headline metric so the ladder driver reads the same shape as the
    other tasks).  There is no merge/generation stage.

Budget semantics are rasta.py's own (identical to the LLM tasks —
these are THE paper-wide rho definitions): per-module DM core V =
floor(sqrt(sqrt(dout*din)/rho)) and HS V = pow2-rounded
sqrt(dout*din)/(2 rho), capped at min(dout,din).  On roberta-large
(1024x1024 Q,V): DM V = 45/32/22/16 and HS V = 1024/512/256/128 for
rho = 0.5/1/2/4 — all four rho points feasible for BOTH variants
(HS rho0.5 sits exactly at the V = d_model cap).  S21 pins these.

Usage:
  python Glue/train_glue.py --config cfg.json
  python Glue/train_glue.py --config cfg.json --print_run_name

Config keys:
  required: task, method, learning_rate, output_dir,
            rho (rasta-*/lora-xs) or rank (lora/vera)
  optional: model (roberta-large), head_lr (5e-4), seeds ([0]),
            num_train_epochs (per-task default), per_device_train_batch_size
            (32), max_seq_len (128), lr_scheduler_type (linear),
            warmup_ratio (0.06), weight_decay (0.01), wht_mode (fused),
            init_scale (0.01), variant (from method), lora_alpha (2*rank),
            lora_dropout (0.0), svd_cache_dir, run_tag,
            max_train_samples, max_eval_samples, logging_steps (50),
            spike_guard (True: NaN/Inf-loss abort -> UNSTABLE marker)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("train_glue")

DEFAULT_MODEL = "FacebookAI/roberta-large"
DEFAULT_HEAD_LR = 5e-4
TARGET_MODULES = ["query", "value"]   # exact last-component match
GLUE_PATH = os.environ.get("DATA_GLUE_PATH", "nyu-mll/glue")

# task -> (sentence keys, num_labels, default epochs, metric name)
# Epochs follow the LoRA-paper (Table 9) lineage.  Metric: matthews for
# cola, pearson for stsb, accuracy otherwise (mnli: matched/mismatched
# mean as the headline; both halves reported too).
TASK_META = {
    "cola":  (("sentence", None),            2, 20, "matthews_corr"),
    "sst2":  (("sentence", None),            2, 10, "accuracy"),
    "mrpc":  (("sentence1", "sentence2"),    2, 20, "accuracy"),
    "stsb":  (("sentence1", "sentence2"),    1, 10, "pearson_corr"),
    "qqp":   (("question1", "question2"),    2, 20, "accuracy"),
    "mnli":  (("premise", "hypothesis"),     3, 10, "accuracy_m_mm_mean"),
    "qnli":  (("question", "sentence"),      2, 10, "accuracy"),
    "rte":   (("sentence1", "sentence2"),    2, 20, "accuracy"),
}

MODEL_ALIASES = {"roberta-large": "FacebookAI/roberta-large",
                 "roberta-base": "FacebookAI/roberta-base"}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── metrics (numpy-only: no sklearn/scipy/evaluate dependency) ───────────────

def metric_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean()) if len(labels) else 0.0


def metric_matthews(preds: np.ndarray, labels: np.ndarray) -> float:
    """Binary MCC from confusion counts; 0.0 on degenerate denominator
    (sklearn's convention)."""
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def metric_pearson(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.float64)
    labels = labels.astype(np.float64)
    if len(labels) < 2 or preds.std() == 0 or labels.std() == 0:
        return 0.0
    return float(np.corrcoef(preds, labels)[0, 1])


# ── run naming ───────────────────────────────────────────────────────────────

def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def method_tag(cfg: Dict) -> str:
    m = cfg["method"]
    if m in ("rasta-dm", "rasta-hs"):
        return f"rasta_{m.split('-')[1]}__rho{float(cfg['rho']):g}"
    if m == "lora-xs":
        return f"loraxs__rho{float(cfg['rho']):g}"
    if m == "lora":
        return f"lora__r{int(cfg['rank'])}"
    if m == "vera":
        return f"vera__r{int(cfg['rank'])}"
    raise ValueError(f"unknown method {m!r}")


def glue_run_name(cfg: Dict, seed: int) -> str:
    """glue_<task>__<method_tag>__<model>__lr<..>[deviation tags]__seed<n>
    Deviation tags appear only when a knob leaves the locked protocol
    (same convention as the LLM run names): __hlr<..> if head_lr !=
    5e-4, __ep<n> if epochs != the task default, __sched.. if not
    linear/0.06, __wht<mode> if not fused, plus run_tag and __n<k> for
    capped train sets."""
    task = cfg["task"]
    parts = [f"glue_{task}", method_tag(cfg),
             str(cfg.get("model", "roberta-large")).replace("/", "_"),
             f"lr{_fmt_lr(float(cfg['learning_rate']))}"]
    hlr = float(cfg.get("head_lr", DEFAULT_HEAD_LR))
    if hlr != DEFAULT_HEAD_LR:
        parts.append(f"hlr{_fmt_lr(hlr)}")
    ep = int(cfg.get("num_train_epochs", TASK_META[task][2]))
    if ep != TASK_META[task][2]:
        parts.append(f"ep{ep}")
    sched = str(cfg.get("lr_scheduler_type", "linear"))
    wu = float(cfg.get("warmup_ratio", 0.06))
    if (sched, wu) != ("linear", 0.06):
        parts.append(f"sched{sched}_wu{f'{wu:g}'.replace('0.', '.')}")
    if cfg["method"] == "rasta-hs" and str(cfg.get("wht_mode",
                                                   "fused")) != "fused":
        parts.append(f"wht{cfg['wht_mode']}")
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    n = cfg.get("max_train_samples")
    parts.append(f"seed{seed}")
    parts.append(f"n{n}" if n else "nfull")
    return "__".join(parts)


# ── model assembly ───────────────────────────────────────────────────────────

def build_model(cfg: Dict, num_labels: int, task: str,
                vocab_size: Optional[int] = None):
    """Construct the classification model and inject the adapter.
    Everything runs fp32 (locked).  Returns (model, adapter_param_count,
    head_param_count).  The classifier head is (re-)enabled for
    training afterwards for the non-peft methods, since
    apply_rasta/apply_lora_xs freeze all non-adapter parameters.

    vocab_size is REQUIRED for the tiny-debug escape and must match the
    tokenizer's own vocab size: the smoke path tokenizes real GLUE text
    with the real roberta-base tokenizer (vocab ~50265) so the tiny
    model's embedding table must cover the same id range, or any token
    id >= the table size is an out-of-bounds embedding lookup that
    surfaces as an async CUDA device-side assert (typically reported
    late, e.g. at the embeddings LayerNorm rather than the lookup
    itself)."""
    from transformers import AutoModelForSequenceClassification, RobertaConfig
    from transformers import RobertaForSequenceClassification

    model_name = MODEL_ALIASES.get(str(cfg.get("model", "roberta-large")),
                                   str(cfg.get("model", DEFAULT_MODEL)))
    kwargs = dict(num_labels=num_labels)
    if task == "stsb":
        kwargs["problem_type"] = "regression"
    if model_name.startswith("tiny-debug"):
        # smoke-only escape: local random tiny roberta, no hub download
        # ("tiny-debug" or "tiny-debug:<hidden>").  vocab_size MUST
        # match the tokenizer actually used to encode the data (see
        # docstring) — never hardcode this.
        if vocab_size is None:
            raise ValueError(
                "tiny-debug model requires vocab_size (must match the "
                "tokenizer used for the data) — pass it from run_one")
        hidden = int(model_name.split(":")[1]) if ":" in model_name else 32
        conf = RobertaConfig(vocab_size=vocab_size, hidden_size=hidden,
                             num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=hidden * 4,
                             max_position_embeddings=520, **kwargs)
        model = RobertaForSequenceClassification(conf)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, use_safetensors=True, **kwargs)
    model = model.float()

    method = cfg["method"]
    if method in ("rasta-dm", "rasta-hs"):
        from rasta import apply_rasta
        variant = method.split("-")[1]
        model = apply_rasta(
            model, rho=float(cfg["rho"]), variant=variant,
            target_modules=TARGET_MODULES,
            frozen_dtype=torch.float32, trainable_dtype=torch.float32,
            init_scale=float(cfg.get("init_scale", 0.01)),
            wht_mode=str(cfg.get("wht_mode", "fused")),
            print_stats=True, verbose=True)
    elif method == "lora-xs":
        from utils.lora_xs import apply_lora_xs, build_svd_cache, \
            resolve_rank_map
        cache = str(cfg.get("svd_cache_dir",
                            os.path.join(ROOT, "utils", "svd_cache")))
        ranks = resolve_rank_map(model, rho=float(cfg["rho"]),
                                 target_modules=TARGET_MODULES)
        build_svd_cache(model, ranks, TARGET_MODULES, svd_cache_dir=cache,
                        model_id=model_name)
        model = apply_lora_xs(model, rho=float(cfg["rho"]),
                              target_modules=TARGET_MODULES,
                              svd_cache_dir=cache,
                              frozen_dtype=torch.float32,
                              trainable_dtype=torch.float32,
                              model_id=model_name)
    elif method in ("lora", "vera"):
        from peft import get_peft_model
        rank = int(cfg["rank"])
        if method == "lora":
            from peft import LoraConfig
            pconf = LoraConfig(
                task_type="SEQ_CLS", r=rank,
                lora_alpha=int(cfg.get("lora_alpha", 2 * rank)),
                lora_dropout=float(cfg.get("lora_dropout", 0.0)),
                target_modules=TARGET_MODULES,
                modules_to_save=["classifier"])
        else:
            from peft import VeraConfig
            pconf = VeraConfig(
                task_type="SEQ_CLS", r=rank,
                vera_dropout=float(cfg.get("vera_dropout", 0.0)),
                target_modules=TARGET_MODULES,
                modules_to_save=["classifier"])
        model = get_peft_model(model, pconf)
    else:
        raise ValueError(f"unknown method {method!r}")

    # non-peft paths froze the fresh head — turn it back on
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad_(True)

    adapter_params = sum(p.numel() for n, p in model.named_parameters()
                         if p.requires_grad and "classifier" not in n)
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" in n)
    logger.info(f"  trainable adapter params: {adapter_params:,}  "
                f"(+ head {head_params:,} @ fixed head_lr, "
                "excluded from adapter counts)")
    return model, adapter_params, head_params


def split_param_groups(model, adapter_lr: float, head_lr: float,
                       weight_decay: float, decay_names: Optional[set]
                       ) -> List[Dict]:
    """Four groups: {adapter, head} x {decay, no-decay}.  A param is a
    head param iff 'classifier' is in its name; decay follows the
    Trainer convention (no decay for bias / LayerNorm) via the provided
    decay_names set (fallback: decay unless name ends with .bias or the
    name contains LayerNorm)."""
    def in_decay(n, p):
        if decay_names is not None:
            return n in decay_names
        return not (n.endswith(".bias") or "LayerNorm" in n)

    g = {"a_d": [], "a_n": [], "h_d": [], "h_n": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ("h" if "classifier" in n else "a") + \
              ("_d" if in_decay(n, p) else "_n")
        g[key].append(p)
    return [
        {"params": g["a_d"], "lr": adapter_lr, "weight_decay": weight_decay},
        {"params": g["a_n"], "lr": adapter_lr, "weight_decay": 0.0},
        {"params": g["h_d"], "lr": head_lr, "weight_decay": weight_decay},
        {"params": g["h_n"], "lr": head_lr, "weight_decay": 0.0},
    ]


# ── data ─────────────────────────────────────────────────────────────────────

def build_datasets(task: str, tokenizer, max_len: int,
                   max_train: Optional[int], max_eval: Optional[int],
                   data_seed: int):
    from datasets import load_dataset
    keys, _, _, _ = TASK_META[task]
    raw = load_dataset(GLUE_PATH, task)

    def tok(batch):
        args = ([batch[keys[0]]] if keys[1] is None
                else [batch[keys[0]], batch[keys[1]]])
        return tokenizer(*args, truncation=True, max_length=max_len)

    cols_keep = {"label", "labels"}
    def prep(split):
        ds = raw[split].map(tok, batched=True)
        drop = [c for c in ds.column_names
                if c not in cols_keep and c not in (
                    "input_ids", "attention_mask", "token_type_ids")]
        return ds.remove_columns(drop)

    train = prep("train").shuffle(seed=data_seed)
    if max_train:
        train = train.select(range(min(max_train, len(train))))
    if task == "mnli":
        evals = {"matched": prep("validation_matched"),
                 "mismatched": prep("validation_mismatched")}
    else:
        evals = {"dev": prep("validation")}
    if max_eval:
        evals = {k: v.select(range(min(max_eval, len(v))))
                 for k, v in evals.items()}
    return train, evals


# ── training ─────────────────────────────────────────────────────────────────

def run_one(cfg: Dict, seed: int) -> None:
    from transformers import (AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, TrainerCallback)

    task = cfg["task"]
    assert task in TASK_META, f"unknown task {task!r}"
    keys, num_labels, default_epochs, metric_name = TASK_META[task]
    method = cfg["method"]
    adapter_lr = float(cfg["learning_rate"])
    head_lr = float(cfg.get("head_lr", DEFAULT_HEAD_LR))
    epochs = int(cfg.get("num_train_epochs", default_epochs))
    batch = int(cfg.get("per_device_train_batch_size", 32))
    max_len = int(cfg.get("max_seq_len", 128))
    wd = float(cfg.get("weight_decay", 0.01))
    guard_on = bool(cfg.get("spike_guard", True))

    run_name = glue_run_name(cfg, seed)
    run_dir = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "eval_results.json")):
        logger.info(f"[skip done] {run_name}")
        return

    logger.info("=" * 72)
    logger.info(f"GLUE {task} | {method} | {run_name}")
    logger.info(f"  adapter_lr={adapter_lr:g}  head_lr={head_lr:g}  "
                f"epochs={epochs}  batch={batch}  seq={max_len}  fp32")
    logger.info("=" * 72)
    set_all_seeds(seed)

    model_name = MODEL_ALIASES.get(str(cfg.get("model", "roberta-large")),
                                   str(cfg.get("model", DEFAULT_MODEL)))
    if model_name.startswith("tiny-debug"):
        tok_src = MODEL_ALIASES["roberta-base"]  # any roberta tokenizer
    else:
        tok_src = model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    train_ds, eval_sets = build_datasets(
        task, tokenizer, max_len,
        cfg.get("max_train_samples"), cfg.get("max_eval_samples"),
        data_seed=int(cfg.get("data_seed", seed)))
    logger.info(f"  train rows: {len(train_ds)}  eval: "
                + ", ".join(f"{k}={len(v)}" for k, v in eval_sets.items()))

    model, adapter_params, head_params = build_model(
        cfg, num_labels, task, vocab_size=len(tokenizer))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if task == "stsb":
            preds = np.squeeze(np.asarray(logits), axis=-1)
            return {"pearson": metric_pearson(preds, np.asarray(labels))}
        preds = np.argmax(np.asarray(logits), axis=-1)
        labels = np.asarray(labels)
        out = {"accuracy": metric_accuracy(preds, labels)}
        if task == "cola":
            out["matthews"] = metric_matthews(preds, labels)
        return out

    history: List[Dict] = []
    unstable = {"hit": False, "reason": ""}

    class Collect(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics:
                history.append({"epoch": state.epoch, **metrics})

    class NanGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(loss):
                unstable["hit"] = True
                unstable["reason"] = f"non-finite loss at step {state.global_step}"
                control.should_training_stop = True

    # warmup as explicit steps (version-proof: some transformers
    # versions dropped the warmup_ratio kwarg); ratio semantics kept.
    steps_per_epoch = max(1, math.ceil(len(train_ds) / batch))
    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))
    warmup_steps = int(round(warmup_ratio * total_steps))

    targs = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=int(cfg.get("eval_batch", 64)),
        learning_rate=adapter_lr,     # nominal; real lrs live in groups
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
        warmup_steps=warmup_steps,
        weight_decay=wd,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=int(cfg.get("logging_steps", 50)),
        seed=seed, data_seed=int(cfg.get("data_seed", seed)),
        report_to="none", label_names=["labels"],
        fp16=False, bf16=False,
        dataloader_num_workers=2,
        remove_unused_columns=True)

    class GlueTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                try:
                    decay_names = set(
                        self.get_decay_parameter_names(self.model))
                except AttributeError:
                    decay_names = None
                groups = split_param_groups(self.model, adapter_lr,
                                            head_lr, wd, decay_names)
                self.optimizer = torch.optim.AdamW(
                    groups, lr=adapter_lr, betas=(0.9, 0.999), eps=1e-8)
            return self.optimizer

    trainer = GlueTrainer(
        model=model, args=targs, train_dataset=train_ds,
        eval_dataset=eval_sets if task == "mnli" else eval_sets["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[Collect()] + ([NanGuard()] if guard_on else []))
    trainer.train()

    loss_hist = [{"step": h.get("step"), "loss": h["loss"],
                  "epoch": h.get("epoch")}
                 for h in trainer.state.log_history if "loss" in h]
    json.dump(loss_hist, open(os.path.join(run_dir, "loss_history.json"),
                              "w"), indent=2)

    if unstable["hit"]:
        json.dump({"reason": unstable["reason"], "lr": adapter_lr,
                   "task": task, "method": method, "seed": seed},
                  open(os.path.join(run_dir, "UNSTABLE"), "w"), indent=2)
        logger.error(f"  UNSTABLE: {unstable['reason']}")
        return

    # ── headline metric per epoch, then best-epoch ──────────────────────────
    def headline(rec: Dict) -> Optional[float]:
        if task == "stsb":
            return rec.get("eval_pearson")
        if task == "cola":
            return rec.get("eval_matthews")
        if task == "mnli":
            return None                       # handled by pairing below
        return rec.get("eval_accuracy")

    per_epoch: List[Dict] = []
    if task == "mnli":
        by_ep: Dict[float, Dict] = {}
        for rec in history:
            ep = round(float(rec["epoch"]), 3)
            slot = by_ep.setdefault(ep, {})
            for k, v in rec.items():
                if k.startswith("eval_") and k.endswith("_accuracy"):
                    slot[k] = v
        for ep in sorted(by_ep):
            s = by_ep[ep]
            m = s.get("eval_matched_accuracy")
            mm = s.get("eval_mismatched_accuracy")
            if m is not None and mm is not None:
                per_epoch.append({"epoch": ep, "metric": (m + mm) / 2,
                                  "matched": m, "mismatched": mm})
    else:
        for rec in history:
            v = headline(rec)
            if v is not None:
                per_epoch.append({"epoch": round(float(rec["epoch"]), 3),
                                  "metric": float(v)})
    if not per_epoch:
        raise RuntimeError("no eval records collected — check eval_strategy")
    best = max(per_epoch, key=lambda r: r["metric"])

    results = {task: {"accuracy": best["metric"], "metric": metric_name,
                      "best_epoch": best["epoch"],
                      "n_epochs": epochs}}
    if task == "mnli":
        results["mnli_matched"] = {"accuracy": best["matched"]}
        results["mnli_mismatched"] = {"accuracy": best["mismatched"]}

    payload = {
        "checkpoint": None,   # metrics-only run; no weights kept
        "eval_protocol": (
            f"GLUE {task} dev, eval every epoch, best-epoch "
            f"{metric_name}; fp32, seq {max_len}, batch {batch}, "
            f"AdamW linear+wu{cfg.get('warmup_ratio', 0.06)}, "
            f"head_lr fixed {head_lr:g} (never searched), "
            "adapters on query+value only"),
        "results": results,
        "history": per_epoch,
    }
    tmp = os.path.join(run_dir, "eval_results.json.tmp")
    json.dump(payload, open(tmp, "w"), indent=2)
    os.replace(tmp, os.path.join(run_dir, "eval_results.json"))

    meta = {"run_name": run_name, "task": f"glue_{task}", "method": method,
            "model": model_name, "seed": seed,
            "adapter_lr": adapter_lr, "head_lr": head_lr,
            "epochs": epochs, "batch": batch, "max_seq_len": max_len,
            "weight_decay": wd,
            "lr_scheduler_type": str(cfg.get("lr_scheduler_type",
                                             "linear")),
            "warmup_ratio": float(cfg.get("warmup_ratio", 0.06)),
            "target_modules": TARGET_MODULES,
            "adapter_trainable_params": adapter_params,
            "head_trainable_params": head_params,
            "rho": cfg.get("rho"), "rank": cfg.get("rank"),
            "wht_mode": cfg.get("wht_mode", "fused")
            if method == "rasta-hs" else None,
            "best": best, "torch": torch.__version__}
    json.dump(meta, open(os.path.join(run_dir, "run_metadata.json"), "w"),
              indent=2, sort_keys=True)
    logger.info(f"  BEST {metric_name} = {best['metric']:.4f} "
                f"@ epoch {best['epoch']}  ->  {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--print_run_name", action="store_true",
                    help="print the seed-0 run name and exit (used by "
                         "the shell drivers to stay in sync)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    seeds = cfg.get("seeds", [0])
    if args.print_run_name:
        print(glue_run_name(cfg, int(seeds[0])))
        return
    for seed in seeds:
        try:
            run_one(cfg, seed=int(seed))
        except Exception:
            logger.error(traceback.format_exc())
            rd = os.path.join(cfg["output_dir"], glue_run_name(cfg, seed))
            os.makedirs(rd, exist_ok=True)
            open(os.path.join(rd, "FAILED"), "w").write(
                traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
