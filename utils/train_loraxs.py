#!/usr/bin/env python3
"""
utils/train_loraxs.py — LoRA-XS baseline training, mirroring
utils/train_rasta.py's I/O contract (run_dir layout, run_metadata.json,
loss_history.json, merged/, spike guard + UNSTABLE marker) so the same
shell helpers and per-task eval scripts work unchanged.  TASK-AGNOSTIC:
dataset via the data_module seam, like train_rasta.

LoRA-XS (Balazy et al. 2024): frozen per-layer SVD bases of the pretrained
weight, trainable r x r core, zero-init (see utils/lora_xs.py, incl. why
the SVD cache is a correctness requirement, not an optimisation).
Closest published relative of RaSTA-DM — the comparison isolates the
basis source (weight-SVD vs random).

RANK ALLOCATION (the allocation ablation) — mirrors RaSTA's own
rho/fixed_V config exactly: rho is ALWAYS required; fixed_rank overrides
how it's realized:
  rho=<number> or "rank:<r>" (derive the budget-matched rho FROM a
    shared rank — the LoRA-XS paper's parameterization; see
    utils.lora_xs.matched_rho_for_rank).
  fixed_rank omitted/null (default): per-layer rank via rasta.dm_vocab —
    RaSTA-DM's exact sizing rule (at the given or derived rho).
  fixed_rank="auto": one shared rank, budget-matched to the rho-based
    total (invalid with rho="rank:<r>" — circular).
  fixed_rank=<int>: explicit uniform rank (the official LoRA-XS setup);
    rho is still recorded (and the allocation-comparison log still
    prints) but ignored for sizing.
See utils.lora_xs.resolve_rank_map / matched_fixed_rank /
matched_rho_for_rank.

Usage:
  torchrun --nproc_per_node=2 -m utils.train_loraxs --config run.json
  (repo root on PYTHONPATH)

Required config keys:
  model, output_dir, rho, learning_rate,
  per_device_train_batch_size, gradient_accumulation_steps

Optional (defaults):
  data_module="Math/data_math.py",
  target_modules=null (null -> all 7 projections),
  svd_cache_dir=utils.lora_xs.DEFAULT_SVD_CACHE_DIR  (utils/svd_cache —
    ONE shared repo-wide cache; build once, reuse everywhere, merge
    verifies hashes),
  spike_guard=true (+ spike_factor/patience/grace/abs_rise as train_rasta),
  run_tag, seeds=[0], max_train_samples, max_steps=-1, eval_samples=0,
  max_seq_len=512, shuffle_data=true, lr_scheduler_type="linear",
  warmup_ratio=0.06, weight_decay=0.01, logging_steps=50,
  use_fast_attn=true, trainer_bf16=true, skip_merge=false
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

from utils.lora_xs import (apply_lora_xs, save_lora_xs_adapter,
                           load_and_merge_lora_xs, LoRAXSAdapter,
                           DEFAULT_SVD_CACHE_DIR)
from utils.train_rasta import SpikeGuard, LossTracker, _selection_tag
from utils.train_eval_shared import (
    get_env_metadata, is_main_process, load_base_model, load_data_module,
    load_tokenizer, resolve_model_name, save_run_metadata, set_all_seeds,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("train_loraxs")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

DEFAULT_DATA_MODULE = "Math/data_math.py"


def _alloc_from_cfg(cfg: Dict[str, Any]):
    """rho is always required (mirrors rasta's config): a number, or
    "rank:<r>" to derive the budget-matched rho from a shared rank at
    apply time.  Returns (rho_cfg, fixed_rank, budget_tag) with rho_cfg
    passed through UNRESOLVED (resolution needs the model's shapes);
    fixed_rank: None | int | "auto".  Tags are stable without loading
    the model: numeric rho -> "rho<r>", spec -> "rhoFromR<r>" (same
    principle as rasta's fixedVauto tag)."""
    if "rho" not in cfg:
        raise ValueError("config must set 'rho' (the reference budget — "
                         "always required, like apply_rasta's rho; a "
                         "number or 'rank:<r>'; use 'fixed_rank' to "
                         "override how it's realized)")
    from utils.lora_xs import parse_rho_spec
    parsed = parse_rho_spec(cfg["rho"])   # validates; raises on garbage
    fixed_rank = cfg.get("fixed_rank", None)
    if isinstance(parsed, int):           # "rank:<r>" spec
        if fixed_rank == "auto":
            raise ValueError("rho='rank:<r>' with fixed_rank='auto' is "
                             "circular — give one derivation direction")
        rho_cfg: Any = str(cfg["rho"])
        rho_tag = f"rhoFromR{parsed}"
    else:
        rho_cfg = parsed
        rho_tag = f"rho{parsed:g}"
    if fixed_rank is None:
        return rho_cfg, None, rho_tag
    if fixed_rank == "auto":
        return rho_cfg, "auto", f"{rho_tag}_fixedRauto"
    return rho_cfg, int(fixed_rank), f"{rho_tag}_fixedR{int(fixed_rank)}"


def loraxs_run_name(cfg: Dict[str, Any], seed: int) -> str:
    lr = float(cfg["learning_rate"])
    lr_str = f"{lr:.0e}".replace("e-0", "e-")
    n = cfg.get("max_train_samples")
    _, _, budget_tag = _alloc_from_cfg(cfg)
    parts = ["loraxs", cfg["model"].replace("/", "_"),
             budget_tag, f"lr{lr_str}"]
    ep = int(cfg.get("num_train_epochs", 1))
    if ep != 1:
        parts.append(f"ep{ep}")
    sched = str(cfg.get("lr_scheduler_type", "linear"))
    wu = float(cfg.get("warmup_ratio", 0.06))
    if (sched, wu) != ("linear", 0.06):
        parts.append(f"sched{sched}_wu{f'{wu:g}'.replace('0.', '.')}")
    tag = _selection_tag(cfg.get("target_modules", None))
    if tag:
        parts.append(tag)
    if cfg.get("run_tag"):
        parts.append(str(cfg["run_tag"]))
    parts += [f"seed{seed}", f"n{n}" if n else "nfull"]
    return "__".join(parts)


def run_one(cfg: Dict[str, Any], seed: int) -> None:
    from transformers import TrainingArguments, Trainer

    model_alias = cfg["model"]
    model_name = resolve_model_name(model_alias)
    rho_cfg, fixed_rank, budget_tag = _alloc_from_cfg(cfg)
    lr = float(cfg["learning_rate"])
    n = cfg.get("max_train_samples", None)
    max_steps = int(cfg.get("max_steps", -1)) or -1
    num_train_epochs = int(cfg.get("num_train_epochs", 1))
    per_device = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    eval_samples = int(cfg.get("eval_samples", 0))
    scheduler_type = str(cfg.get("lr_scheduler_type", "linear"))
    data_module_spec = str(cfg.get("data_module", DEFAULT_DATA_MODULE))
    target_modules = cfg.get("target_modules", None)
    target_modules = (list(target_modules) if target_modules is not None
                      else list(TARGET_MODULES))
    # ONE shared repo-wide cache anchored to utils/lora_xs.py — NOT to
    # this trainer's own directory (the old default), which after the
    # repo reorganisation would silently grow a second cache and weaken
    # the "training and merge read the same factors" guarantee.
    svd_cache_dir = str(cfg.get("svd_cache_dir", DEFAULT_SVD_CACHE_DIR))
    guard_on = bool(cfg.get("spike_guard", True))

    run_name = loraxs_run_name(cfg, seed)
    out_dir = cfg["output_dir"]
    run_dir = os.path.join(out_dir, run_name)
    adapter_dir = os.path.join(run_dir, "adapter")
    merged_dir = os.path.join(run_dir, "merged")
    os.makedirs(run_dir, exist_ok=True)

    if is_main_process():
        logger.info("=" * 72)
        logger.info(f"LoRA-XS  |  {run_name}")
        if fixed_rank is None:
            alloc = f"rho={rho_cfg} (per-layer ranks via rasta.dm_vocab)"
        elif fixed_rank == "auto":
            alloc = f"rho={rho_cfg} fixed_rank='auto' (budget-matched shared rank)"
        else:
            alloc = f"fixed_rank={fixed_rank} (uniform; rho={rho_cfg} recorded only)"
        logger.info(f"  {alloc}  lr={lr}  epochs={num_train_epochs}  "
                    f"seed={seed}  data={data_module_spec}  "
                    f"svd_cache={svd_cache_dir}")
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
    model = apply_lora_xs(base, rho=rho_cfg, fixed_rank=fixed_rank,
                          target_modules=target_modules,
                          svd_cache_dir=svd_cache_dir, frozen_dtype=dtype,
                          print_stats=is_main_process())
    rank_map = model._lora_xs_meta["rank_map"]
    # Resolved numeric rho (== rho_cfg when numeric; derived when the
    # config carried a "rank:<r>" spec).  Downstream metadata/logs and —
    # crucially — the alloc-ablation driver's DM stage read THIS value,
    # so RaSTA-DM trains at the exact rho the XS rho-cell used.
    rho = float(model._lora_xs_meta["rho"])
    rho_spec = model._lora_xs_meta.get("rho_spec")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff = per_device * grad_accum * world
    if is_main_process():
        logger.info(f"  effective_batch={eff}  "
                    f"steps/epoch={math.ceil(len(train_ds)/eff)}  "
                    f"trainable={trainable_params:,} "
                    f"({100.*trainable_params/total_params:.4f}%)")

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
            json.dump({"reason": guard.reason, "lr": lr,
                       "rho": rho, "fixed_rank": fixed_rank,
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
        save_lora_xs_adapter(save_model, adapter_dir,
                             training_meta={"run_name": run_name, "lr": lr,
                                            "seed": seed,
                                            "final_loss": final_loss,
                                            "data_module": data_module_spec})
        tokenizer.save_pretrained(adapter_dir)

        if not bool(cfg.get("skip_merge", False)):
            del model, save_model
            torch.cuda.empty_cache()
            logger.info("  Loading fresh base model for merge ...")
            fresh = load_base_model(model_name, torch_dtype=dtype,
                                    use_fast_attn=bool(cfg.get("use_fast_attn", True)))
            merged = load_and_merge_lora_xs(fresh, adapter_dir)
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"  Merged model saved → {merged_dir}")
            del merged, fresh
            torch.cuda.empty_cache()

        rs = sorted(set(rank_map.values()))
        save_run_metadata(os.path.join(run_dir, "run_metadata.json"), {
            "run_name": run_name, "method": "lora_xs",
            "model_alias": model_alias, "model_name": model_name,
            "rho": rho, "rho_spec": rho_spec, "fixed_rank": fixed_rank,
            "num_train_epochs": num_train_epochs,
            "rank_min": rs[0], "rank_max": rs[-1], "ranks_unique": rs,
            "n_adapted_layers": len(rank_map),
            "data_module": data_module_spec,
            "target_modules": target_modules,
            "selection": _selection_tag(target_modules) or "all",
            "svd_cache_dir": svd_cache_dir,
            "lr": lr, "lr_scheduler_type": scheduler_type,
            "warmup_ratio": float(cfg.get("warmup_ratio", 0.06)),
            "seed": seed, "max_steps": max_steps if max_steps > 0 else None,
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
    ap = argparse.ArgumentParser(description="Train LoRA-XS baseline")
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
                    rd = os.path.join(cfg["output_dir"], loraxs_run_name(cfg, seed))
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
