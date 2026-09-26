"""
utils/train_rasta.py — Unified training script for RaSTA adapters
(rho/variant API).  TASK-AGNOSTIC: the dataset comes from a task data
module loaded by file path (config key `data_module`, e.g.
"Math/data_math.py"); everything else in this file is method logic.

Variants (paper naming):
  variant: "dm"  → Dense Mixing      ΔW = B_L M B_R^T
  variant: "hs"  → Hadamard Scaling  ΔW = B_L Diag(s_L) H Diag(s_R) B_R^T

Usage:
  torchrun --nproc_per_node=2 -m utils.train_rasta --config run.json
  (repo root must be on PYTHONPATH; the shell drivers export it)

Required config keys:
  model, output_dir, rho, variant, learning_rate,
  per_device_train_batch_size, gradient_accumulation_steps

Optional config keys (with defaults):
  data_module="Math/data_math.py"  Task data module (file path relative
                      to the repo root).  Must expose build_train_data
                      with the shared signature — see
                      utils.train_eval_shared.load_data_module.
  basis_construction="gaussian"   The public module is gaussian-only (the
                      SRHT construction was split into a separate,
                      systems-focused artifact and is not present here).
                      Any value other than "gaussian" is REJECTED with an
                      explicit error rather than silently ignored — a
                      sweep config that asked for "srht" must not appear
                      to run while quietly training gaussian instead.
                      Kept as a config/metadata field (rather than removed
                      outright) purely for continuity with already-
                      collected run directories and their naming scheme.
  init_scale=0.01
  fixed_V=null       (int → uniform vocabulary V for every adapted layer;
                      "auto" → budget-matched shared V via
                      rasta.matched_fixed_V() against the rho-based sizing
                      on the same target modules; null → default per-layer
                      sizing.  Shared-V ablation.)
  target_modules=null (null → all seven projections; a list like
                      ["q_proj","v_proj"] restricts — module-selection
                      ablation)
  shared_lr_basis=false  (ablation: draw B_L and B_R from one shared
                      underlying random matrix per layer instead of two
                      independent draws; both variants.  Recorded in the
                      checkpoint and replayed at merge.)
  freeze_side=null   ("L" | "R"; hs only — freezes that diagonal at an
                      all-ones pass-through and trains only the other,
                      halving HS's trainable count.  Rejected for
                      variant="dm".  Recorded in the checkpoint and
                      replayed at merge.)
  wht_mode="fused"   ("torch" | "fused"; "fused" is safe without the
                      optional kernel installed — auto-fallback with a
                      warning; recorded in the checkpoint and always
                      re-applied at merge).  Governs the HS variant's
                      Hadamard mixer transform; DM is unaffected.
  spike_guard=true   Loss-spike guard: stops training early when smoothed
                      train loss exceeds spike_factor x its running min for
                      spike_patience consecutive log steps (or goes
                      NaN/inf), then writes <run_dir>/UNSTABLE instead of a
                      merged model.  Drivers treat UNSTABLE as "retry at
                      the next-lower grid lr".  This is detection +
                      pipeline-level fix; the run itself is never silently
                      continued at a different lr, so every run dir stays
                      a single labelled (lr, config) point.
  spike_factor=1.5, spike_patience=2, spike_grace_ratio=0.05,
  spike_abs_rise=0.4
  run_tag=null, seeds=[0], max_train_samples=null, max_steps=-1,
  num_train_epochs=1 (values != 1 append an 'ep<n>' run-name tag),
  eval_samples=0, max_seq_len=512, shuffle_data=true, data_seed=<seed>,
  lr_scheduler_type="linear", warmup_ratio=0.06, weight_decay=0.01,
  logging_steps=50, use_fast_attn=true, trainer_bf16=true, skip_merge=false

Legacy keys from the pre-simplification module (r, use_hadamard,
num_groups, basis_seed, srht_num_rounds, share_basis_across_layers,
achlioptas_s, kane_nelson_s) are REJECTED with explicit errors — the
module no longer implements them, and silently ignoring them would make an
old sweep config appear to run while doing something different.

CHECKPOINT COMPATIBILITY NOTE: this module's basis key strings and
generation algorithm for basis_construction="gaussian" are byte-for-byte
identical to the pre-simplification module's, so adapter checkpoints from
earlier gaussian runs load and merge unchanged — no retraining needed.
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
import torch.nn as nn
from transformers import TrainerCallback, TrainerControl, TrainerState

from rasta import (
    apply_rasta, save_rasta_adapter, load_and_merge_rasta, matched_fixed_V,
    RaSTAHSAdapter, RaSTADMAdapter, clear_rasta_caches,
)
from utils.train_eval_shared import (
    get_env_metadata,
    is_main_process,
    load_base_model,
    load_data_module,
    load_tokenizer,
    resolve_model_name,
    save_run_metadata,
    set_all_seeds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_rasta")

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_DATA_MODULE = "Math/data_math.py"

# Keys the simplified module removed.  Rejecting them loudly beats running
# a config that used to mean something else.
_LEGACY_KEYS = {
    "r": "renamed to 'rho' (r collided with LoRA's rank; see paper notation)",
    "use_hadamard": "replaced by variant: 'hs' | 'dm'",
    "num_groups": "grouping was removed from the module (G=1 always)",
    "basis_seed": "basis seeding was removed (canonical bases only)",
    "srht_num_rounds": "SRHT rounds were removed (single round always)",
    "share_basis_across_layers": "basis sharing is now unconditional",
    "achlioptas_s": "achlioptas construction was removed",
    "kane_nelson_s": "kane_nelson construction was removed",
}


def _reject_legacy(cfg: Dict[str, Any]) -> None:
    bad = [k for k in _LEGACY_KEYS if k in cfg]
    if bad:
        msgs = "; ".join(f"'{k}': {_LEGACY_KEYS[k]}" for k in bad)
        raise ValueError(f"Config uses removed keys — {msgs}")


def _resolve_basis_construction(cfg: Dict[str, Any]) -> str:
    """The public module is gaussian-only.  basis_construction is kept as
    a recognised config/metadata key (defaulting to "gaussian", matching
    the pre-simplification module's default) purely so existing run
    directories, naming, and metadata stay comparable — but any value
    other than "gaussian" is now REJECTED, not silently coerced."""
    bc = str(cfg.get("basis_construction", "gaussian"))
    if bc != "gaussian":
        raise ValueError(
            f"basis_construction={bc!r} is not supported by this module "
            f"build — it is gaussian-only (SRHT was split into a separate "
            f"systems-focused artifact, not present here). Set "
            f"basis_construction to \"gaussian\" or omit it. If you need "
            f"to merge/evaluate an existing srht-trained checkpoint, use "
            f"the full module that produced it, not this script.")
    return bc


# ── Config-derived tag helpers (shared by run_one and main's failure path) ────

def _selection_tag(target_modules: Optional[List[str]]) -> Optional[str]:
    """None when the selection is the full default; 'qv' for the (q,v)
    ablation; a generic 'tm<...>' tag for anything else."""
    if target_modules is None:
        return None
    if sorted(target_modules) == sorted(TARGET_MODULES):
        return None
    if sorted(target_modules) == ["q_proj", "v_proj"]:
        return "qv"
    return "tm" + "-".join(s.split("_")[0] for s in sorted(target_modules))


def _fixed_v_tag(fixed_V_cfg: Any) -> Optional[str]:
    """'fixedVauto' for auto mode (stable name independent of the resolved
    value, which needs the model loaded); 'fixedV<n>' for an explicit int;
    None for default per-layer sizing."""
    if fixed_V_cfg is None:
        return None
    if fixed_V_cfg == "auto":
        return "fixedVauto"
    return f"fixedV{int(fixed_V_cfg)}"


def _ablation_tags(cfg: Dict[str, Any]) -> List[Optional[str]]:
    """Deviation-only tags for the N-14/N-15 ablations."""
    tags: List[Optional[str]] = []
    fs = cfg.get("freeze_side", None)
    tags.append(f"fz{fs}" if fs else None)
    tags.append("shLR" if cfg.get("shared_lr_basis", False) else None)
    return tags


def _epoch_tag(cfg: Dict[str, Any]) -> Optional[str]:
    """Deviation-only: 'ep<n>' when num_train_epochs != 1.  Lives in
    _tags_from_cfg (config-derived tags) so the shell name-prediction
    path — which calls _tags_from_cfg(extra) — produces it identically
    to the trainer, provided the driver puts num_train_epochs in extra."""
    ep = int(cfg.get("num_train_epochs", 1))
    return f"ep{ep}" if ep != 1 else None


def _tags_from_cfg(cfg: Dict[str, Any]) -> List[Optional[str]]:
    return [
        _selection_tag(cfg.get("target_modules", None)),
        _fixed_v_tag(cfg.get("fixed_V", None)),
        *_ablation_tags(cfg),
        _epoch_tag(cfg),
    ]


# ── Loss tracking + spike guard ───────────────────────────────────────────────

class LossTracker(TrainerCallback):
    """Record training loss at each logging step."""

    def __init__(self) -> None:
        self.latest: Optional[float] = None
        self.history: List[Dict] = []

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs: Optional[Dict] = None, **kwargs) -> None:
        if logs and "loss" in logs:
            self.latest = float(logs["loss"])
            self.history.append({
                "step": state.global_step,
                "loss": self.latest,
                "lr": logs.get("learning_rate"),
            })


class SpikeGuard(TrainerCallback):
    """Stop training when the (smoothed) train loss destabilises.

    Trigger (either criterion, `patience` consecutive log events):
      * ratio:    smoothed > spike_factor x running-min   (low-loss regime)
      * absolute: smoothed > running-min + abs_rise nats  (high-loss floors,
        where cross-entropy saturates near ln(vocab) and a diverged run can
        sit far above its min without ever reaching the ratio threshold)
    NaN/inf triggers immediately.  A grace window of
    grace_ratio x total steps is exempt (warmup noise).  On trigger the
    callback sets should_training_stop and records why; run_one() then
    writes <run_dir>/UNSTABLE and skips saving/merging so the driver's
    fallback ladder can retry at a lower lr.  This exists because three
    (model, variant, rho=1) cells in the screening phase were stable at the
    150-step selection horizon but destabilised at 500 steps — instability
    that emerges with horizon must be caught DURING the (much longer) main
    runs, not at eval time.
    """

    def __init__(self, factor: float = 1.5, patience: int = 2,
                 grace_ratio: float = 0.05, abs_rise: float = 0.4) -> None:
        self.factor = factor
        self.patience = patience
        self.grace_ratio = grace_ratio
        self.abs_rise = abs_rise
        self.smooth: Optional[float] = None
        self.run_min = float("inf")
        self.strikes = 0
        self.triggered = False
        self.reason: Optional[str] = None

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs: Optional[Dict] = None, **kwargs) -> None:
        if not logs or "loss" not in logs:
            return
        v = float(logs["loss"])
        if math.isnan(v) or math.isinf(v):
            self.triggered = True
            self.reason = f"non-finite loss at step {state.global_step}"
            control.should_training_stop = True
            return
        self.smooth = v if self.smooth is None else 0.7 * self.smooth + 0.3 * v
        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        in_grace = (total is not None
                    and state.global_step < self.grace_ratio * total)
        self.run_min = min(self.run_min, self.smooth)
        if in_grace:
            return
        ratio_hit = self.smooth > self.factor * self.run_min
        abs_hit = self.smooth > self.run_min + self.abs_rise
        if ratio_hit or abs_hit:
            self.strikes += 1
            if self.strikes >= self.patience:
                which = "ratio" if ratio_hit else "absolute-rise"
                self.triggered = True
                self.reason = (f"smoothed loss {self.smooth:.4f} vs running "
                               f"min {self.run_min:.4f} tripped the {which} "
                               f"criterion for {self.strikes} consecutive "
                               f"logs (step {state.global_step})")
                control.should_training_stop = True
        else:
            self.strikes = 0


# ── Run-name builder ──────────────────────────────────────────────────────────

def build_run_name(
    model_alias: str,
    rho: float,
    variant: str,
    bc: str,
    lr: float,
    seed: int,
    n: Optional[int],
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.06,
    wht_mode: str = "fused",
    extra_tag: Optional[str] = None,
    config_tags: Optional[List[Optional[str]]] = None,
) -> str:
    """
    Fresh naming scheme for the simplified module (rho/variant; no G, no
    basis seed).  Deviation-only tags: lr_scheduler_type/warmup_ratio
    append only when they differ from (linear, 0.06); wht_mode appends
    only when not "fused"; config_tags entries append only when non-None.
    extra_tag (run_tag in the JSON config) appends unconditionally when
    given.  Shell drivers must call THIS function (python3 -c ...) rather
    than re-implementing the scheme — a second copy WILL drift and predict
    a directory the trainer never creates.
    """
    lr_str = f"{lr:.0e}".replace("e-0", "e-")
    n_tag = f"n{n}" if n else "nfull"
    parts = [
        f"rasta_{variant}_{bc}",
        f"rho{rho:g}",
        model_alias.replace("/", "_"),
        f"lr{lr_str}",
    ]
    if (lr_scheduler_type, warmup_ratio) != ("linear", 0.06):
        wu_str = f"{warmup_ratio:g}".replace("0.", ".")
        parts.append(f"sched{lr_scheduler_type}_wu{wu_str}")
    if wht_mode != "fused":
        parts.append(f"wht{wht_mode}")
    for tag in (config_tags or []):
        if tag:
            parts.append(tag)
    if extra_tag:
        parts.append(str(extra_tag))
    parts += [f"seed{seed}", n_tag]
    return "__".join(parts)


# ── Adapter parameter stats ───────────────────────────────────────────────────

def _log_adapter_stats(model: nn.Module, variant: str, label: str) -> None:
    """Print mean-abs of trainable adapter params for a quick sanity check."""
    AdapterCls = RaSTAHSAdapter if variant == "hs" else RaSTADMAdapter
    stats = []
    for name, mod in model.named_modules():
        if isinstance(mod, AdapterCls):
            if variant == "hs":
                stats.append((name,
                              mod.s_L.detach().cpu().abs().mean().item(),
                              mod.s_R.detach().cpu().abs().mean().item()))
            else:
                stats.append((name, mod.M.detach().cpu().abs().mean().item()))
            if len(stats) >= 4:
                break
    if not stats:
        logger.warning(f"  [{label}] No adapter modules found for stats.")
        return
    logger.info(f"  [{label}] adapter param mean-abs (first {len(stats)} layers):")
    for s in stats:
        if variant == "hs":
            logger.info(f"    {s[0]}: s_L={s[1]:.6f}  s_R={s[2]:.6f}")
        else:
            logger.info(f"    {s[0]}: M={s[1]:.6f}")

    all_vals = []
    for name, mod in model.named_modules():
        if isinstance(mod, AdapterCls):
            if variant == "hs":
                all_vals.append(mod.s_L.detach().cpu().abs())
                all_vals.append(mod.s_R.detach().cpu().abs())
            else:
                all_vals.append(mod.M.detach().cpu().abs())
    if all_vals:
        combined = torch.cat([v.flatten() for v in all_vals])
        logger.info(f"    Global: mean={combined.mean():.6f} "
                    f"max={combined.max():.6f} std={combined.std():.6f}")
        if combined.mean() < 1e-7:
            logger.warning(
                f"  [{label}] ⚠  Adapter params are near zero — "
                "training may not have updated them (check DDP/trainer setup).")


# ── Single-seed training run ──────────────────────────────────────────────────

def run_one(cfg: Dict[str, Any], seed: int) -> None:
    from transformers import TrainingArguments, Trainer

    _reject_legacy(cfg)

    # ── Unpack config ──────────────────────────────────────────────────────────
    model_alias = cfg["model"]
    model_name = resolve_model_name(model_alias)
    rho = float(cfg["rho"])
    variant = str(cfg["variant"])
    if variant not in ("dm", "hs"):
        raise ValueError(f"variant must be 'dm'|'hs', got {variant!r}")
    bc = _resolve_basis_construction(cfg)   # always "gaussian"; raises otherwise
    init_scale = float(cfg.get("init_scale", 0.01))
    wht_mode = str(cfg.get("wht_mode", "fused"))
    run_tag = cfg.get("run_tag", None)
    eval_samples = int(cfg.get("eval_samples", 0))
    scheduler_type = str(cfg.get("lr_scheduler_type", "linear"))
    skip_merge = bool(cfg.get("skip_merge", False))
    lr = float(cfg["learning_rate"])
    n = cfg.get("max_train_samples", None)
    max_steps = int(cfg.get("max_steps", -1)) or -1
    num_train_epochs = int(cfg.get("num_train_epochs", 1))
    per_device = int(cfg["per_device_train_batch_size"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    data_module_spec = str(cfg.get("data_module", DEFAULT_DATA_MODULE))

    fixed_V_cfg = cfg.get("fixed_V", None)          # None | int | "auto"
    shared_lr_basis = bool(cfg.get("shared_lr_basis", False))
    freeze_side = cfg.get("freeze_side", None)      # None | "L" | "R"
    target_modules = cfg.get("target_modules", None)
    target_modules = (list(target_modules) if target_modules is not None
                      else list(TARGET_MODULES))

    guard_on = bool(cfg.get("spike_guard", True))
    spike_factor = float(cfg.get("spike_factor", 1.5))
    spike_patience = int(cfg.get("spike_patience", 2))
    spike_grace = float(cfg.get("spike_grace_ratio", 0.05))
    spike_abs = float(cfg.get("spike_abs_rise", 0.4))

    # ── Paths ──────────────────────────────────────────────────────────────────
    run_name = build_run_name(model_alias, rho, variant, bc, lr, seed, n,
                              lr_scheduler_type=scheduler_type,
                              warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
                              wht_mode=wht_mode,
                              extra_tag=run_tag,
                              config_tags=_tags_from_cfg(cfg))
    out_dir = cfg["output_dir"]
    run_dir = os.path.join(out_dir, run_name)
    adapter_dir = os.path.join(run_dir, "adapter")
    merged_dir = os.path.join(run_dir, "merged")
    hf_ckpt_dir = os.path.join(run_dir, "hf_checkpoints")
    os.makedirs(run_dir, exist_ok=True)

    if is_main_process():
        logger.info("=" * 72)
        logger.info(f"RaSTA-{variant.upper()}  |  {run_name}")
        logger.info(f"  rho={rho}  bc={bc}  wht_mode={wht_mode}"
                    f"  init_scale={init_scale}  data={data_module_spec}")
        logger.info(f"  lr={lr}  wd={cfg.get('weight_decay', 0.01)}"
                    f"  warmup={cfg.get('warmup_ratio', 0.06)}"
                    f"  seed={seed}  fixed_V={fixed_V_cfg}"
                    f"  shared_lr_basis={shared_lr_basis}"
                    f"  freeze_side={freeze_side}"
                    f"  guard={'on' if guard_on else 'OFF'}"
                    f"  targets="
                    + ("default" if sorted(target_modules) == sorted(TARGET_MODULES)
                       else str(target_modules)))
        if max_steps > 0:
            logger.info(f"  [SANITY RUN] max_steps={max_steps}")
        logger.info("=" * 72)

    set_all_seeds(seed)

    # ── Data (via the task data-module seam) ───────────────────────────────────
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
        # Identical to a head split whenever the pool suffices.
        total = len(_combined_ds)
        n_eval = min(eval_samples, max(0, total - 1))
        n_train = min(n, total - n_eval)
        train_ds = _Subset(_combined_ds, range(n_train))
        eval_ds = _Subset(_combined_ds, range(total - n_eval, total))
    else:
        train_ds, eval_ds = _combined_ds, None

    # ── Model ──────────────────────────────────────────────────────────────────
    base_model = load_base_model(
        model_name,
        torch_dtype=torch.bfloat16 if cfg.get("trainer_bf16", True)
        else torch.float32,
        use_fast_attn=bool(cfg.get("use_fast_attn",
                                   cfg.get("use_torch_sdpa", True))),
    )

    # Resolve fixed_V.  "auto" = budget-matched shared V against the
    # rho-based per-layer sizing on the SAME target modules.
    fixed_V: Optional[int] = None
    fv_budget_r = fv_budget_shared = None
    if fixed_V_cfg == "auto":
        fixed_V, fv_budget_r, fv_budget_shared = matched_fixed_V(
            base_model, rho=rho, variant=variant,
            target_modules=target_modules, verbose=is_main_process(),
        )
    elif fixed_V_cfg is not None:
        fixed_V = int(fixed_V_cfg)

    model = apply_rasta(
        model=base_model,
        rho=rho,
        variant=variant,
        target_modules=target_modules,
        frozen_dtype=torch.bfloat16 if cfg.get("trainer_bf16", True)
        else torch.float32,
        trainable_dtype=torch.float32,
        init_scale=init_scale,
        fixed_V=fixed_V,
        wht_mode=wht_mode,
        shared_lr_basis=shared_lr_basis,
        freeze_side=freeze_side,
        print_stats=is_main_process(),
        verbose=is_main_process(),
    )

    if is_main_process():
        _log_adapter_stats(model, variant, "pre-train")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    # ── Train ──────────────────────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_bs = per_device * grad_accum * world_size
    steps_per_epoch = math.ceil(len(train_ds) / effective_bs)
    if is_main_process():
        logger.info(f"  effective_batch={effective_bs}"
                    f"  steps/epoch={steps_per_epoch}"
                    f"  trainable={trainable_params:,}"
                    f"  ({100.*trainable_params/total_params:.4f}%)")

    loss_tracker = LossTracker()
    guard = SpikeGuard(spike_factor, spike_patience, spike_grace, spike_abs)
    callbacks: List[TrainerCallback] = [loss_tracker]
    if guard_on:
        callbacks.append(guard)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    final_loss = loss_tracker.latest

    # ── Instability path: marker instead of artifacts ─────────────────────────
    if guard.triggered:
        if is_main_process():
            logger.error(f"  ⚠ SPIKE GUARD TRIGGERED: {guard.reason}")
            with open(os.path.join(run_dir, "UNSTABLE"), "w") as f:
                json.dump({"reason": guard.reason, "lr": lr, "rho": rho,
                           "variant": variant, "seed": seed,
                           "loss_history": loss_tracker.history[-20:]},
                          f, indent=2)
            with open(os.path.join(run_dir, "loss_history.json"), "w") as f:
                json.dump(loss_tracker.history, f, indent=2)
            logger.error(f"  UNSTABLE marker written → {run_dir}/UNSTABLE "
                         f"(no adapter/merge saved; driver should retry at "
                         f"the next-lower grid lr)")
        return

    # Eval loss on held-out samples (all processes must participate)
    eval_loss = None
    if eval_ds is not None:
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        if is_main_process():
            logger.info("  Eval loss: "
                        + (f"{eval_loss:.4f}" if eval_loss is not None
                           else "unavailable (empty eval set?)"))

    # ── Post-training sanity + saves (main process only) ──────────────────────
    if is_main_process():
        save_model = trainer.model
        if hasattr(save_model, "module"):
            save_model = save_model.module

        _log_adapter_stats(save_model, variant, "post-train")

        if loss_tracker.history:
            first_loss = loss_tracker.history[0]["loss"]
            last_loss = loss_tracker.history[-1]["loss"]
            logger.info(f"  Loss: {first_loss:.4f} → {last_loss:.4f} "
                        f"over {len(loss_tracker.history)} log steps")

        loss_history_path = os.path.join(run_dir, "loss_history.json")
        with open(loss_history_path, "w") as f:
            json.dump(loss_tracker.history, f, indent=2)
        logger.info(f"  Loss history ({len(loss_tracker.history)} points) "
                    f"saved → {loss_history_path}")

        os.makedirs(adapter_dir, exist_ok=True)
        save_rasta_adapter(
            model=save_model,
            save_dir=adapter_dir,
            target_modules=target_modules,
            rho=rho,
            variant=variant,
            init_scale=init_scale,
            fixed_V=fixed_V,
            shared_lr_basis=shared_lr_basis,
            freeze_side=freeze_side,
            training_meta={
                "run_name": run_name,
                "lr": lr,
                "seed": seed,
                "final_loss": final_loss,
                "loss_history": loss_tracker.history[-10:],
                "basis_construction": bc,
                "data_module": data_module_spec,
            },
        )
        tokenizer.save_pretrained(adapter_dir)

        if skip_merge:
            logger.info("  skip_merge=True — skipping merge.")
            del model, save_model
            torch.cuda.empty_cache()
        else:
            del model
            del save_model
            torch.cuda.empty_cache()
            clear_rasta_caches()

            logger.info("  Loading fresh base model for merge ...")
            fresh = load_base_model(
                model_name,
                torch_dtype=torch.bfloat16 if cfg.get("trainer_bf16", True)
                else torch.float32,
                use_fast_attn=bool(cfg.get("use_fast_attn",
                                           cfg.get("use_torch_sdpa", True))),
            )
            merged = load_and_merge_rasta(fresh, adapter_dir)

            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"  Merged model saved → {merged_dir}")
            del merged, fresh
            torch.cuda.empty_cache()

        save_run_metadata(os.path.join(run_dir, "run_metadata.json"), {
            "run_name":           run_name,
            "method":             "rasta",
            "variant":            variant,
            "model_alias":        model_alias,
            "model_name":         model_name,
            "rho":                rho,
            "basis_construction": bc,
            "data_module":        data_module_spec,
            "init_scale":         init_scale,
            "wht_mode":           wht_mode,
            "fixed_V_mode":       ("auto" if fixed_V_cfg == "auto"
                                   else ("int" if fixed_V_cfg is not None else None)),
            "fixed_V":            fixed_V,
            "fixed_V_budget_rho_based": fv_budget_r,
            "fixed_V_budget_shared":    fv_budget_shared,
            "target_modules":     target_modules,
            "selection":          _selection_tag(target_modules) or "all",
            "shared_lr_basis":    shared_lr_basis,
            "freeze_side":        freeze_side,
            "lr":                 lr,
            "lr_scheduler_type":  scheduler_type,
            "warmup_ratio":       float(cfg.get("warmup_ratio", 0.06)),
            "seed":               seed,
            "num_train_epochs":   num_train_epochs,
            "max_steps":          max_steps if max_steps > 0 else None,
            "effective_batch":    effective_bs,
            "max_train_samples":  n,
            "trainable_params":   trainable_params,
            "total_params":       total_params,
            "trainable_pct":      100.0 * trainable_params / total_params,
            "final_train_loss":   final_loss,
            "eval_loss":          eval_loss,
            "spike_guard":        guard_on,
            "loss_history_file":  loss_history_path,
            "gsm8k_accuracy":     None,
            "math_accuracy":      None,
            "merged_dir":         merged_dir,
            "data":               data_meta,
            "env":                get_env_metadata(),
        })
        logger.info(f"Run complete → {run_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RaSTA adapter")
    parser.add_argument("--config", required=True, help="JSON config file path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    seeds: List[int] = cfg.get("seeds", [0])
    try:
        for seed in seeds:
            try:
                run_one(cfg, seed=seed)
            except Exception:
                logger.error(traceback.format_exc())
                if is_main_process():
                    try:
                        run_name = build_run_name(
                            cfg["model"], float(cfg["rho"]),
                            str(cfg.get("variant", "hs")),
                            str(cfg.get("basis_construction", "gaussian")),
                            float(cfg["learning_rate"]),
                            seed, cfg.get("max_train_samples"),
                            lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
                            warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
                            wht_mode=str(cfg.get("wht_mode", "fused")),
                            extra_tag=cfg.get("run_tag", None),
                            config_tags=_tags_from_cfg(cfg),
                        )
                        run_dir = os.path.join(cfg["output_dir"], run_name)
                        os.makedirs(run_dir, exist_ok=True)
                        with open(os.path.join(run_dir, "FAILED"), "w") as f:
                            f.write(traceback.format_exc())
                    except Exception:
                        pass
                raise
    finally:
        # torchrun/NCCL can report a nonzero exit code from an otherwise-
        # successful run if the process group is never destroyed.  Shell
        # drivers check torchrun's exit code as their success signal, so
        # that warning made genuinely completed runs register as failures.
        # (Ported from train_loraxs, where the fix first landed.)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
