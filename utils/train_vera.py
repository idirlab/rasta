#!/usr/bin/env python3
"""
utils/train_vera.py — VeRA baseline (Kopiczko et al., 2024, via PEFT),
mirroring utils/train_rasta.py's I/O contract (run_dir layout,
run_metadata.json, loss_history.json, merged/, spike guard + UNSTABLE).
TASK-AGNOSTIC: dataset via the data_module seam, like the other trainers.

TARGET MODULES: default is ALL SEVEN projections — the same selection
RaSTA and LoRA-XS use, so the method comparison holds targets fixed.
PEFT's VeRA implements the paper's slicing trick (one shared (A, B) pair
sized to the LARGEST targeted layer dims, sliced per layer), so
heterogeneous shapes — GQA k/v, wide MLP projections — are supported
natively.  Trainable params per layer = r (d-vector) + out_features
(b-vector); the per-layer b-vectors dominate, so the rank sweep probes
shared-basis capacity far more than budget.

VeRA lrs are typically 10-50x LoRA's (only scaling vectors train); the
paper uses separate head/non-head lrs which PEFT does not expose — a
single lr around 1e-2 is the community default.  The spike guard + lr
grid make this searchable the same way as RaSTA.

Usage:
  torchrun --nproc_per_node=2 -m utils.train_vera --config run.json
  (repo root on PYTHONPATH; the shell drivers export it)

Required config keys:
  model, output_dir, rank, learning_rate,
  per_device_train_batch_size, gradient_accumulation_steps

Optional (defaults):
  data_module="Math/data_math.py",
  target_modules=all 7 projections, vera_dropout=0.0, d_initial=0.1,
  num_train_epochs=1 (values != 1 append an 'ep<n>' run-name tag),
  spike_guard=true (+ knobs as train_rasta), run_tag, seeds=[0],
  max_train_samples, max_steps=-1, eval_samples=0, max_seq_len=512,
  shuffle_data=true, lr_scheduler_type="linear", warmup_ratio=0.06,
  weight_decay=0.01, logging_steps=50, use_fast_attn=true,
  trainer_bf16=true, skip_merge=false
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from utils.train_rasta import SpikeGuard, LossTracker
from utils.train_eval_shared import (
    get_env_metadata, is_main_process, load_base_model, load_data_module,
    load_tokenizer, resolve_model_name, save_run_metadata, set_all_seeds,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("train_vera")

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]  # see docstring

DEFAULT_DATA_MODULE = "Math/data_math.py"


def vera_run_name(cfg: Dict[str, Any], seed: int) -> str:
    lr = float(cfg["learning_rate"])
    lr_str = f"{lr:.0e}".replace("e-0", "e-")
    n = cfg.get("max_train_samples")
    parts = ["vera", cfg["model"].replace("/", "_"),
             f"r{int(cfg['rank'])}", f"lr{lr_str}"]
    sched = str(cfg.get("lr_scheduler_type", "linear"))
    wu = float(cfg.get("warmup_ratio", 0.06))
    if (sched, wu) != ("linear", 0.06):
        parts.append(f"sched{sched}_wu{f'{wu:g}'.replace('0.', '.')}")
    tm = cfg.get("target_modules", None)
    if tm is not None and sorted(tm) != sorted(DEFAULT_TARGETS):
        parts.append("tm" + "-".join(s.split("_")[0] for s in sorted(tm)))
    ep = int(cfg.get("num_train_epochs", 1))
    if ep != 1:
        parts.append(f"ep{ep}")
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    parts += [f"seed{seed}", f"n{n}" if n else "nfull"]
    return "__".join(parts)


def run_one(cfg: Dict[str, Any], seed: int) -> None:
    from transformers import TrainingArguments, Trainer
    from peft import VeraConfig, get_peft_model

    model_alias = cfg["model"]
    model_name = resolve_model_name(model_alias)
    rank = int(cfg["rank"])
    lr = float(cfg["learning_rate"])
    n = cfg.get("max_train_samples", None)
    max_steps = int(cfg.get("max_steps", -1)) or -1
    num_train_epochs = int(cfg.get("num_train_epochs", 1))
    per_device = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    eval_samples = int(cfg.get("eval_samples", 0))
    scheduler_type = str(cfg.get("lr_scheduler_type", "linear"))
    data_module_spec = str(cfg.get("data_module", DEFAULT_DATA_MODULE))
    target_modules = list(cfg.get("target_modules", DEFAULT_TARGETS))
    guard_on = bool(cfg.get("spike_guard", True))

    run_name = vera_run_name(cfg, seed)
    out_dir = cfg["output_dir"]
    run_dir = os.path.join(out_dir, run_name)
    adapter_dir = os.path.join(run_dir, "adapter")
    merged_dir = os.path.join(run_dir, "merged")
    os.makedirs(run_dir, exist_ok=True)

    if is_main_process():
        logger.info("=" * 72)
        logger.info(f"VeRA  |  {run_name}")
        logger.info(f"  rank={rank}  lr={lr}  epochs={num_train_epochs}  "
                    f"seed={seed}  data={data_module_spec}  "
                    f"targets={target_modules} (shared sliced A/B)")
        logger.info("=" * 72)

    set_all_seeds(seed)

    data_mod = load_data_module(data_module_spec)
    tokenizer = (data_mod.load_task_tokenizer(model_name)
                 if hasattr(data_mod, "load_task_tokenizer")
                 else load_tokenizer(model_name))
    # ^ seam hook: a task data module may own its tokenizer policy (e.g.
    # commonsense requires pad_token_id=0 + left padding, matching its
    # original train/eval scripts byte-for-byte); tasks without the hook
    # get the shared default.
    _total = (n or 999999) + eval_samples
    _ds, collator, data_meta = data_mod.build_train_data(
        tokenizer=tokenizer, max_seq_len=int(cfg.get("max_seq_len", 512)),
        data_seed=int(cfg.get("data_seed", seed)),
        shuffle_data=bool(cfg.get("shuffle_data", True)),
        max_train_samples=_total if eval_samples > 0 else n)
    if eval_samples > 0 and n is not None:
        from torch.utils.data import Subset
        # Reserve the eval split from the pool's TAIL so it can never be
        # empty when the dataset is smaller than n + eval_samples.
        total = len(_ds)
        n_eval = min(eval_samples, max(0, total - 1))
        n_train = min(n, total - n_eval)
        train_ds = Subset(_ds, range(n_train))
        eval_ds = Subset(_ds, range(total - n_eval, total))
    else:
        train_ds, eval_ds = _ds, None

    dtype = torch.bfloat16 if cfg.get("trainer_bf16", True) else torch.float32
    base = load_base_model(model_name, torch_dtype=dtype,
                           use_fast_attn=bool(cfg.get("use_fast_attn", True)))
    vera_cfg = VeraConfig(
        r=rank, target_modules=target_modules,
        vera_dropout=float(cfg.get("vera_dropout", 0.0)),
        d_initial=float(cfg.get("d_initial", 0.1)),
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, vera_cfg)
    if is_main_process():
        model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff = per_device * grad_accum * world
    if is_main_process():
        logger.info(f"  effective_batch={eff}  "
                    f"steps/epoch={math.ceil(len(train_ds)/eff)}  "
                    f"trainable={trainable_params:,}")

    tracker = LossTracker()
    guard = SpikeGuard(float(cfg.get("spike_factor", 1.5)),
                       int(cfg.get("spike_patience", 2)),
                       float(cfg.get("spike_grace_ratio", 0.05)),
                       float(cfg.get("spike_abs_rise", 0.4)))
    callbacks = [tracker] + ([guard] if guard_on else [])

    args_t = TrainingArguments(
        output_dir=os.path.join(run_dir, "hf_checkpoints"),
        num_train_epochs=num_train_epochs, max_steps=max_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type=scheduler_type,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06))
        if scheduler_type != "constant" else 0.0,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        bf16=bool(cfg.get("trainer_bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_strategy="no", eval_strategy="no",
        ddp_find_unused_parameters=False, dataloader_num_workers=2,
        remove_unused_columns=False, report_to="none",
        label_names=["labels"])
    trainer = Trainer(model=model, args=args_t, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator,
                      callbacks=callbacks)
    trainer.train()
    final_loss = tracker.latest

    if guard.triggered:
        if is_main_process():
            logger.error(f"  ⚠ SPIKE GUARD: {guard.reason}")
            json.dump({"reason": guard.reason, "lr": lr, "rank": rank,
                       "seed": seed, "loss_history": tracker.history[-20:]},
                      open(os.path.join(run_dir, "UNSTABLE"), "w"), indent=2)
            json.dump(tracker.history,
                      open(os.path.join(run_dir, "loss_history.json"), "w"),
                      indent=2)
        return

    eval_loss = None
    if eval_ds is not None:
        eval_loss = trainer.evaluate().get("eval_loss")
        if is_main_process():
            logger.info("  Eval loss: "
                        + (f"{eval_loss:.4f}" if eval_loss is not None
                           else "unavailable (empty eval set?)"))

    if is_main_process():
        save_model = trainer.model
        if hasattr(save_model, "module"):
            save_model = save_model.module
        loss_history_path = os.path.join(run_dir, "loss_history.json")
        json.dump(tracker.history, open(loss_history_path, "w"), indent=2)

        os.makedirs(adapter_dir, exist_ok=True)
        save_model.save_pretrained(adapter_dir)   # PEFT VeRA adapter
        tokenizer.save_pretrained(adapter_dir)

        if not bool(cfg.get("skip_merge", False)):
            logger.info("  Merging VeRA into base weights ...")
            merged = save_model.merge_and_unload()
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"  Merged model saved → {merged_dir}")
            del merged
            torch.cuda.empty_cache()

        save_run_metadata(os.path.join(run_dir, "run_metadata.json"), {
            "run_name": run_name, "method": "vera",
            "model_alias": model_alias, "model_name": model_name,
            "rank": rank, "target_modules": target_modules,
            "d_initial": float(cfg.get("d_initial", 0.1)),
            "data_module": data_module_spec,
            "lr": lr, "lr_scheduler_type": scheduler_type,
            "warmup_ratio": float(cfg.get("warmup_ratio", 0.06)),
            "seed": seed, "num_train_epochs": num_train_epochs,
            "max_steps": max_steps if max_steps > 0 else None,
            "effective_batch": eff, "max_train_samples": n,
            "trainable_params": trainable_params, "total_params": total_params,
            "trainable_pct": 100.0 * trainable_params / total_params,
            "final_train_loss": final_loss, "eval_loss": eval_loss,
            "spike_guard": guard_on, "loss_history_file": loss_history_path,
            "gsm8k_accuracy": None, "math_accuracy": None,
            "merged_dir": merged_dir, "data": data_meta,
            "env": get_env_metadata()})
        logger.info(f"Run complete → {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train VeRA baseline")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    try:
        for seed in cfg.get("seeds", [0]):
            try:
                run_one(cfg, seed=seed)
            except Exception:
                logger.error(traceback.format_exc())
                if is_main_process():
                    rd = os.path.join(cfg["output_dir"], vera_run_name(cfg, seed))
                    os.makedirs(rd, exist_ok=True)
                    open(os.path.join(rd, "FAILED"), "w").write(traceback.format_exc())
                raise
    finally:
        # torchrun/NCCL can report a nonzero exit code from an
        # otherwise-successful run if the process group is never
        # destroyed.  Shell drivers check torchrun's exit code as their
        # success signal.  try/finally so this runs on BOTH the normal-
        # completion and the re-raised-exception paths.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
