#!/usr/bin/env python3
"""
utils/train_lora.py — LoRA baseline training, mirroring
utils/train_rasta.py's I/O contract (run_dir layout, run_metadata.json,
loss_history.json, merged/) so the same shell drivers and per-task eval
scripts work for both.  TASK-AGNOSTIC: dataset via the data_module seam.

Usage:
  torchrun --nproc_per_node=2 -m utils.train_lora --config run.json
  (repo root on PYTHONPATH; the shell drivers export it)

Required config keys:
  model, output_dir, rank, learning_rate,
  per_device_train_batch_size, gradient_accumulation_steps

Optional (defaults):
  data_module="Math/data_math.py",
  lora_alpha = 2*rank    (paper protocol: alpha = 2r)
  lora_dropout = 0.0
  target_modules = null  (null → all 7 projections; e.g.
                          ["q_proj","v_proj"] for the (q,v) ablation)
  num_train_epochs=1 (values != 1 append an 'ep<n>' run-name tag),
  run_tag = null, seeds=[0], max_train_samples=null, max_steps=-1,
  eval_samples=0, max_seq_len=512, shuffle_data=true,
  lr_scheduler_type="linear", warmup_ratio=0.06, weight_decay=0.01,
  logging_steps=50, use_fast_attn=true, trainer_bf16=true, skip_merge=false
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

from utils.train_eval_shared import (
    get_env_metadata, is_main_process, load_base_model, load_data_module,
    load_tokenizer, resolve_model_name, save_run_metadata, set_all_seeds,
    make_lora_run_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lora")

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_DATA_MODULE = "Math/data_math.py"


def _selection_tag(target_modules: Optional[List[str]]) -> Optional[str]:
    if target_modules is None:
        return None
    if sorted(target_modules) == sorted(TARGET_MODULES):
        return None
    if sorted(target_modules) == ["q_proj", "v_proj"]:
        return "qv"
    return "tm" + "-".join(s.split("_")[0] for s in sorted(target_modules))


def lora_run_name(cfg: Dict[str, Any], seed: int) -> str:
    """make_lora_run_name from train_eval_shared, extended with the same
    deviation-only tags train_rasta.build_run_name uses (selection, sched,
    epochs, run_tag). Tags append only on deviation so historical LoRA run
    dirs keep resolving. Shell drivers should call THIS function, not
    rebuild."""
    rank  = int(cfg["rank"])
    alpha = int(cfg.get("lora_alpha", 2 * rank))
    base = make_lora_run_name(
        model_alias=cfg["model"], rank=rank, lora_alpha=alpha,
        lr=float(cfg["learning_rate"]), seed=seed,
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        use_dora=False, max_train_samples=cfg.get("max_train_samples"),
    )
    extra: List[str] = []
    sched  = str(cfg.get("lr_scheduler_type", "linear"))
    warmup = float(cfg.get("warmup_ratio", 0.06))
    if (sched, warmup) != ("linear", 0.06):
        extra.append(f"sched{sched}_wu{f'{warmup:g}'.replace('0.', '.')}")
    tag = _selection_tag(cfg.get("target_modules", None))
    if tag:
        extra.append(tag)
    ep = int(cfg.get("num_train_epochs", 1))
    if ep != 1:
        extra.append(f"ep{ep}")
    if cfg.get("run_tag"):
        extra.append(str(cfg["run_tag"]))
    if not extra:
        return base
    # insert extras before the trailing seed/n components
    parts = base.split("__")
    return "__".join(parts[:-2] + extra + parts[-2:])


def run_one(cfg: Dict[str, Any], seed: int) -> None:
    from transformers import TrainingArguments, Trainer, TrainerCallback
    from peft import LoraConfig, get_peft_model

    model_alias = cfg["model"]
    model_name  = resolve_model_name(model_alias)
    rank        = int(cfg["rank"])
    alpha       = int(cfg.get("lora_alpha", 2 * rank))
    dropout     = float(cfg.get("lora_dropout", 0.0))
    lr          = float(cfg["learning_rate"])
    n           = cfg.get("max_train_samples", None)
    max_steps   = int(cfg.get("max_steps", -1)) or -1
    num_train_epochs = int(cfg.get("num_train_epochs", 1))
    per_device  = int(cfg["per_device_train_batch_size"])
    grad_accum  = int(cfg["gradient_accumulation_steps"])
    eval_samples   = int(cfg.get("eval_samples", 0))
    scheduler_type = str(cfg.get("lr_scheduler_type", "linear"))
    data_module_spec = str(cfg.get("data_module", DEFAULT_DATA_MODULE))
    target_modules = cfg.get("target_modules", None)
    target_modules = (list(target_modules) if target_modules is not None
                      else list(TARGET_MODULES))

    run_name    = lora_run_name(cfg, seed)
    out_dir     = cfg["output_dir"]
    run_dir     = os.path.join(out_dir, run_name)
    adapter_dir = os.path.join(run_dir, "adapter")
    merged_dir  = os.path.join(run_dir, "merged")
    hf_ckpt_dir = os.path.join(run_dir, "hf_checkpoints")
    os.makedirs(run_dir, exist_ok=True)

    if is_main_process():
        logger.info("=" * 72)
        logger.info(f"LoRA  |  {run_name}")
        logger.info(f"  rank={rank}  alpha={alpha}  dropout={dropout}"
                    f"  lr={lr}  epochs={num_train_epochs}  seed={seed}"
                    f"  data={data_module_spec}"
                    f"  targets="
                    + ("default" if sorted(target_modules) == sorted(TARGET_MODULES)
                       else str(target_modules)))
        if max_steps > 0:
            logger.info(f"  [SANITY RUN] max_steps={max_steps}")
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
    _total_samples = (n or 999999) + eval_samples
    _combined_ds, collator, data_meta = data_mod.build_train_data(
        tokenizer=tokenizer,
        max_seq_len=int(cfg.get("max_seq_len", 512)),
        data_seed=int(cfg.get("data_seed", seed)),
        shuffle_data=bool(cfg.get("shuffle_data", True)),
        max_train_samples=_total_samples if eval_samples > 0 else n,
    )
    if eval_samples > 0 and n is not None:
        from torch.utils.data import Subset as _Subset
        # Reserve the eval split from the pool's TAIL so it can never be
        # empty when the dataset is smaller than n + eval_samples.
        total = len(_combined_ds)
        n_eval = min(eval_samples, max(0, total - 1))
        n_train = min(n, total - n_eval)
        train_ds = _Subset(_combined_ds, range(n_train))
        eval_ds  = _Subset(_combined_ds, range(total - n_eval, total))
    else:
        train_ds, eval_ds = _combined_ds, None

    base_model = load_base_model(
        model_name, torch_dtype=torch.bfloat16 if cfg.get("trainer_bf16", True)
        else torch.float32,
        use_fast_attn=bool(cfg.get("use_fast_attn",
                                   cfg.get("use_torch_sdpa", True))),
    )
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    if is_main_process():
        model.print_trainable_parameters()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    world_size   = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bs = per_device * grad_accum * world_size
    if is_main_process():
        logger.info(f"  effective_batch={effective_bs}"
                    f"  steps/epoch={math.ceil(len(train_ds)/effective_bs)}"
                    f"  trainable={trainable_params:,}"
                    f"  ({100.*trainable_params/total_params:.4f}%)")

    history: List[Dict] = []

    class _Loss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                history.append({"step": state.global_step,
                                "loss": float(logs["loss"]),
                                "lr": logs.get("learning_rate")})

    training_args = TrainingArguments(
        output_dir=hf_ckpt_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=scheduler_type,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06))
                     if scheduler_type != "constant" else 0.0,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        bf16=bool(cfg.get("trainer_bf16", True)),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_strategy="no",
        eval_strategy="no",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator, callbacks=[_Loss()])
    trainer.train()
    final_loss = history[-1]["loss"] if history else None

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
        with open(loss_history_path, "w") as f:
            json.dump(history, f, indent=2)

        os.makedirs(adapter_dir, exist_ok=True)
        save_model.save_pretrained(adapter_dir)   # PEFT adapter
        tokenizer.save_pretrained(adapter_dir)

        if bool(cfg.get("skip_merge", False)):
            logger.info("  skip_merge=True — skipping merge.")
        else:
            # Merge LoRA into the base weights → plain HF model, same layout
            # as RaSTA's merged/ so eval_math.py treats both identically.
            logger.info("  Merging LoRA into base weights ...")
            merged = save_model.merge_and_unload()
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"  Merged model saved → {merged_dir}")
            del merged
            torch.cuda.empty_cache()

        save_run_metadata(os.path.join(run_dir, "run_metadata.json"), {
            "run_name":          run_name,
            "method":            "lora",
            "model_alias":       model_alias,
            "model_name":        model_name,
            "rank":              rank,
            "lora_alpha":        alpha,
            "lora_dropout":      dropout,
            "target_modules":    target_modules,
            "selection":         _selection_tag(target_modules) or "all",
            "data_module":       data_module_spec,
            "lr":                lr,
            "lr_scheduler_type": scheduler_type,
            "warmup_ratio":      float(cfg.get("warmup_ratio", 0.06)),
            "seed":              seed,
            "num_train_epochs":  num_train_epochs,
            "max_steps":         max_steps if max_steps > 0 else None,
            "effective_batch":   effective_bs,
            "max_train_samples": n,
            "trainable_params":  trainable_params,
            "total_params":      total_params,
            "trainable_pct":     100.0 * trainable_params / total_params,
            "final_train_loss":  final_loss,
            "eval_loss":         eval_loss,
            "loss_history_file": loss_history_path,
            "gsm8k_accuracy":    None,
            "math_accuracy":     None,
            "merged_dir":        merged_dir,
            "data":              data_meta,
            "env":               get_env_metadata(),
        })
        logger.info(f"Run complete → {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    try:
        for seed in cfg.get("seeds", [0]):
            try:
                run_one(cfg, seed=seed)
            except Exception:
                logger.error(traceback.format_exc())
                if is_main_process():
                    run_dir = os.path.join(cfg["output_dir"],
                                           lora_run_name(cfg, seed))
                    os.makedirs(run_dir, exist_ok=True)
                    with open(os.path.join(run_dir, "FAILED"), "w") as f:
                        f.write(traceback.format_exc())
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
