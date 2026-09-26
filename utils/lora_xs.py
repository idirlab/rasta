#!/usr/bin/env python3
"""
lora_xs.py — LoRA-XS baseline (Balazy et al., 2024).  Task-agnostic;
lives in utils/ and is shared by every instruction-tuning experiment.

Formulation: for each adapted weight W (out, in), take the truncated SVD
W ≈ U_r S_r V_r^T and set FROZEN bases
    A = U_r sqrt(S_r)          (out, r)
    B = sqrt(S_r) V_r^T        (r, in)
with a TRAINABLE core R (r, r), initialised to zero:
    ΔW = A R B                 -> r^2 trainable params per layer.

This is the closest published relative of RaSTA-DM (frozen bases, small
trainable core); the comparison isolates the basis source: SVD of the
pretrained weight (weight-adapted) vs RaSTA's random bases.

RANK ALLOCATION — mirrors rasta.apply_rasta(rho, fixed_V) exactly, so the
allocation ablation is a direct analogy rather than a separate scheme:
  rho is ALWAYS given (the reference budget); fixed_rank overrides how
  it is realized:
    fixed_rank=None (default): per-layer r = rasta.dm_vocab(d_out, d_in,
        rho) — RaSTA-DM's exact sizing rule, so DM and LoRA-XS become
        allocation-identical and differ ONLY in basis construction.
    fixed_rank="auto": one shared rank for every adapted layer, chosen
        by matched_fixed_rank() to budget-match the rho-based total
        (RaSTA's fixed_V="auto" ablation, transplanted).  NOT identical
        to the rho-based total (rounding a single shared value can't
        hit every layer's target exactly) — apply_lora_xs() always logs
        both totals and their ratio so the discrepancy is visible.
    fixed_rank=<int>: uniform rank set explicitly (the official LoRA-XS
        paper's own setup, e.g. r=100); rho is still recorded (and the
        comparison log still prints) but sizing ignores it.
BOTH DERIVATION DIRECTIONS are supported:
    rho=<number> + fixed_rank="auto"  : shared rank derived FROM rho.
    rho="rank:<r>"                    : rho derived FROM a shared rank
        via matched_rho_for_rank() — the direction the LoRA-XS paper's
        setup needs (they report a shared rank, we derive the budget-
        matched rho).  Combine with fixed_rank=<r> for the uniform cell
        (rho then recorded/logged only) or leave fixed_rank unset for
        the rho-based cell at the derived rho.  rho="rank:<r>" with
        fixed_rank="auto" is rejected (circular derivation).
The resolved per-layer map is recorded in the adapter config and reused
verbatim by save/hash/merge.

SVD method — matches the OFFICIAL LoRA-XS repo: randomized SVD via
  torch.svd_lowrank with niter=5 power iterations, fp32.  Two deliberate
  differences from a naive per-run port, both quality-neutral-or-better:
  (1) computed ONCE per (model, layer) at sketch size q = K+10 (K =
  max(layer_rank, 64)) and SLICED per rank — a strictly larger sketch
  than the official per-run q=r, so the top-r factors are at least as
  accurate; (2) the sketch RNG is seeded per layer (crc32 of the layer
  name), so cache builds are reproducible.  Build cost: ~1-3 min per
  7-8B model (vs ~10-20 min for exact SVD).

SVD caching — correctness, not just speed:
  randomized SVD depends on RNG state, and even exact SVD signs can flip
  across devices/backends.  Training-time and merge-time bases must be
  bit-identical, so factors are computed once, stored under
  DEFAULT_SVD_CACHE_DIR (utils/svd_cache — ONE shared cache for the
  whole repo; training scripts import the constant rather than
  re-deriving a path next to themselves), and every later use (any rank
  <= cached K, any run, the merge step) READS the cache rather than
  recomputing.  save_lora_xs_adapter() records a SHA-256 of each
  layer's sliced (A, B); load_and_merge_lora_xs() recomputes the hashes
  from the cache and refuses to merge on mismatch.

API (mirrors rasta.py's):
  apply_lora_xs(model, rho, fixed_rank=None, target_modules,
                svd_cache_dir, ...)      rho: number | "rank:<r>"
  save_lora_xs_adapter(model, save_dir, ...)
  load_and_merge_lora_xs(fresh_base_model, adapter_dir)
  build_svd_cache(model, ranks, target_modules, svd_cache_dir)
  resolve_rank_map(model, target_modules, rho, fixed_rank=None)
  matched_fixed_rank(model, rho, target_modules) -> (rank, total_rho,
                     total_shared)   — rasta.matched_fixed_V's analogue
  matched_rho_for_rank(model, rank, target_modules) -> (rho, total_rho,
                     total_shared)   — the INVERSE derivation

Prebuild CLI (optional; training builds the cache automatically):
  python -m utils.lora_xs --model meta-llama/Llama-2-7b-hf --rho 1
  python -m utils.lora_xs --model mistralai/Mistral-7B-v0.1 \\
      --rho rank:64 --fixed-rank 64
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from rasta import dm_vocab   # repo root on PYTHONPATH (as for every script)

_CACHE_K_DEFAULT = 64   # min factors stored per layer; rank <= K slices this

# ONE shared cache for the whole repo, anchored to THIS file's location
# (utils/svd_cache), NOT to whichever training script happens to import
# it — otherwise each task directory silently grows its own cache and
# the "training and merge read the same factors" guarantee weakens to
# "…the same factors, if you launched both from the same directory".
DEFAULT_SVD_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "svd_cache")

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _cache_path(cache_dir: str, model_id: str, layer_name: str) -> str:
    return os.path.join(cache_dir, _sanitize(model_id),
                        _sanitize(layer_name) + ".pt")


def _matches_target(module_name: str, target_modules: List[str]) -> bool:
    last = module_name.split(".")[-1]
    return last in target_modules


def _is_adaptable(name: str, mod: nn.Module,
                  target_modules: List[str]) -> bool:
    # "layers." matches llama-style "...layers.N..." paths; "layer."
    # matches encoder-style "...layer.N..." (roberta/bert).  NB the two
    # are NOT substrings of each other ("layers.0" does not contain
    # "layer."), hence the explicit disjunction.  The target_modules
    # last-component match is the real gate.
    return (isinstance(mod, nn.Linear)
            and ("layers." in name or "layer." in name)
            and "lm_head" not in name
            and _matches_target(name, target_modules))


# ── Rank allocation ──────────────────────────────────────────────────────────
# Mirrors rasta.apply_rasta(rho, fixed_V) exactly: rho is always the
# reference budget; fixed_rank overrides how it is realized.

def _rho_based_layers(model: nn.Module, target_modules: List[str],
                      rho: float) -> List[Tuple[str, int, int, int]]:
    """Dry scan: (name, dout, din, r) for every target layer the
    rho-based rule would adapt (r = dm_vocab(...), skipped if it exceeds
    min(dout, din) — the same skip rule apply_rasta uses)."""
    out = []
    for name, mod in model.named_modules():
        if not _is_adaptable(name, mod, target_modules):
            continue
        dout, din = mod.weight.shape
        r = dm_vocab(dout, din, rho)
        if r > min(dout, din):
            continue
        out.append((name, dout, din, r))
    return out


def matched_fixed_rank(model: nn.Module, rho: float,
                       target_modules: Optional[List[str]] = None,
                       verbose: bool = True) -> Tuple[int, int, int]:
    """rasta.matched_fixed_V's analogue for LoRA-XS: the single uniform
    rank that, applied to every layer the rho-based rule would adapt,
    best matches that rule's total trainable budget (V^2 per DM/XS
    layer).  Capped at the smallest adapted layer's min(d_out, d_in) so
    it adapts exactly the same layer set as the rho-based run — no
    silent skips, mirroring rasta's own rationale.  NOT dry-run-free of
    apply_lora_xs's own scan (this exists mainly for the "auto" mode and
    for apply_lora_xs's informational parity log); the dry scan itself
    touches no SVD cache and is cheap.

    Returns (rank_shared, total_rho_based, total_shared) — the two
    totals will generally differ (rounding a single value can't hit
    every layer's per-layer target exactly), which is exactly the
    comparison this function exists to surface."""
    target_modules = (list(target_modules) if target_modules is not None
                      else list(DEFAULT_TARGET_MODULES))
    layers = _rho_based_layers(model, target_modules, rho)
    if not layers:
        raise ValueError("matched_fixed_rank: no target layers found/adaptable.")
    L = len(layers)
    total_rho = sum(r * r for _, _, _, r in layers)
    cap = min(min(dout, din) for _, dout, din, _ in layers)

    r_shared = max(2, int(round((total_rho / L) ** 0.5)))
    r_shared = min(r_shared, cap)
    total_shared = r_shared * r_shared * L

    if verbose:
        ratio = total_shared / total_rho if total_rho else float("nan")
        print(f"  [lora_xs/matched_fixed_rank] rho={rho} layers={L} "
              f"-> rank_shared={r_shared} | budget rho-based={total_rho:,} "
              f"shared={total_shared:,} (ratio {ratio:.3f})")
    return r_shared, total_rho, total_shared


def parse_rho_spec(rho: Union[float, int, str]) -> Union[float, int]:
    """Validate a rho spec: a number (returned as float) or the string
    "rank:<int>" (returns the int reference rank).  Callers distinguish
    the two by return type."""
    if isinstance(rho, str):
        if not rho.startswith("rank:"):
            raise ValueError(
                f"rho must be a number or 'rank:<int>', got {rho!r}")
        try:
            return int(rho.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"bad rank in rho spec {rho!r}")
    return float(rho)


def matched_rho_for_rank(model: nn.Module, rank: int,
                         target_modules: Optional[List[str]] = None,
                         verbose: bool = True) -> Tuple[float, int, int]:
    """INVERSE of matched_fixed_rank: given a shared/uniform rank (the
    quantity the LoRA-XS paper reports), find the rho whose rho-based
    per-layer allocation (dm_vocab rule) best matches the shared setup's
    total trainable budget L * rank^2 over the same layer set.

    The rho-based total is a decreasing step function of rho (dm_vocab
    floors), so no rho generally hits the target exactly; this searches
    a log-spaced grid around the continuous-approximation solution
    rho0 = sum_l sqrt(m_l * n_l) / (L * rank^2) and keeps the closest
    match (ties broken toward the larger budget), then snaps rho to the
    shortest decimal that preserves the chosen total, so configs and
    run names stay readable.

    Returns (rho, total_rho_based_at_rho, total_shared) — same shape as
    matched_fixed_rank, with the roles of "given" and "derived" swapped.
    The trainers' allocation-comparison logs print these, so the
    residual mismatch is visible per run, never assumed away."""
    target_modules = (list(target_modules) if target_modules is not None
                      else list(DEFAULT_TARGET_MODULES))
    rank = int(rank)
    layers = []
    for name, mod in model.named_modules():
        if not _is_adaptable(name, mod, target_modules):
            continue
        dout, din = mod.weight.shape
        if rank > min(dout, din):
            continue
        layers.append((name, dout, din))
    if not layers:
        raise ValueError(
            f"matched_rho_for_rank: no target layers can hold rank {rank}")
    L = len(layers)
    total_shared = L * rank * rank
    s = [(dout * din) ** 0.5 for _, dout, din in layers]
    rho0 = sum(s) / total_shared

    def _total(rho: float) -> int:
        return sum(dm_vocab(dout, din, rho) ** 2
                   for _, dout, din in layers)

    best_rho, best_diff, best_total = None, None, None
    n = 2001
    for i in range(n):
        rho = rho0 * (8.0 ** ((i / (n - 1)) * 2 - 1))   # rho0/8 .. rho0*8
        t = _total(rho)
        d = abs(t - total_shared)
        if (best_diff is None or d < best_diff
                or (d == best_diff and t > best_total)):
            best_rho, best_diff, best_total = rho, d, t
    # Snap to the shortest decimal that keeps the same total (dm_vocab is
    # a step function, so an interval of rhos yields the identical map).
    for nd in (1, 2, 3, 4, 6):
        cand = round(best_rho, nd)
        if cand > 0 and _total(cand) == best_total:
            best_rho = cand
            break

    if verbose:
        ratio = best_total / total_shared if total_shared else float("nan")
        print(f"  [lora_xs/matched_rho_for_rank] rank={rank} layers={L} "
              f"-> rho={best_rho:g} | budget shared={total_shared:,} "
              f"rho-based={best_total:,} (ratio {ratio:.3f})")
    return best_rho, best_total, total_shared


def resolve_rank_map(model: nn.Module,
                     target_modules: Optional[List[str]],
                     rho: float,
                     fixed_rank: Union[None, int, str] = None
                     ) -> Dict[str, int]:
    """Per-layer rank map.

    fixed_rank=None (default): r = rasta.dm_vocab(d_out, d_in, rho) per
      layer, byte-for-byte RaSTA-DM's sizing rule — the two methods'
      per-layer budgets are IDENTICAL and the comparison isolates the
      basis source.
    fixed_rank="auto": one shared rank via matched_fixed_rank(rho),
      applied to exactly the layers the rho-based rule would adapt (so
      switching fixed_rank never changes the adapted layer set).
    fixed_rank=<int>: that rank, uniform, for every target-matching
      layer (skipped where it exceeds min(d_out, d_in)) — the official
      LoRA-XS setup; rho is not used for sizing in this mode."""
    target_modules = (list(target_modules) if target_modules is not None
                      else list(DEFAULT_TARGET_MODULES))
    if fixed_rank is None:
        return {name: r for name, _, _, r
               in _rho_based_layers(model, target_modules, rho)}
    if fixed_rank == "auto":
        r_shared, _, _ = matched_fixed_rank(model, rho, target_modules,
                                            verbose=False)
        return {name: r_shared for name, _, _, _
               in _rho_based_layers(model, target_modules, rho)}
    r = int(fixed_rank)
    ranks: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if not _is_adaptable(name, mod, target_modules):
            continue
        dout, din = mod.weight.shape
        if r > min(dout, din):
            continue
        ranks[name] = r
    if not ranks:
        raise ValueError("resolve_rank_map: no adaptable target layers "
                         f"at fixed_rank={r}")
    return ranks


_SVD_NITER = 5        # matches the official LoRA-XS repo default
_SVD_OVERSAMPLE = 10  # sketch size q = k + 10 (> official per-run q=r)


def _compute_layer_svd(weight: torch.Tensor, k: int,
                       seed_name: str = "") -> Dict[str, torch.Tensor]:
    """Top-k factors via randomized SVD (torch.svd_lowrank, niter=5),
    fp32, on the weight's device.  RNG seeded per layer for reproducible
    cache builds; correctness across runs comes from the cache itself."""
    W = weight.detach().float()
    q = min(min(W.shape), k + _SVD_OVERSAMPLE)
    devices = [W.device] if W.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(zlib.crc32(seed_name.encode()) & 0x7FFFFFFF)
        U, S, V = torch.svd_lowrank(W, q=q, niter=_SVD_NITER)
    k = min(k, S.shape[0])
    return {"U": U[:, :k].contiguous().cpu(),
            "S": S[:k].contiguous().cpu(),
            "Vh": V[:, :k].T.contiguous().cpu(),
            "method": f"svd_lowrank(q={q},niter={_SVD_NITER})"}


def build_svd_cache(model: nn.Module,
                    ranks: Union[int, Dict[str, int]],
                    target_modules: List[str],
                    svd_cache_dir: str = DEFAULT_SVD_CACHE_DIR,
                    model_id: Optional[str] = None,
                    verbose: bool = True) -> str:
    """Compute-and-store top-max(layer_rank, 64) factors for every
    targeted layer not already cached with sufficient K.  `ranks` is a
    uniform int or a per-layer map from resolve_rank_map().  Idempotent;
    safe to call at the start of every run.  Because the stored K is
    >= 64 and sliced on read, a cache built for one allocation mode
    serves the other too whenever its ranks fit under the stored K."""
    model_id = model_id or getattr(model.config, "_name_or_path", "model")
    n_new = 0
    for name, mod in model.named_modules():
        if not _is_adaptable(name, mod, target_modules):
            continue
        r = ranks if isinstance(ranks, int) else ranks.get(name)
        if r is None:
            continue
        k = max(r, _CACHE_K_DEFAULT)
        p = _cache_path(svd_cache_dir, model_id, name)
        if os.path.exists(p):
            cached = torch.load(p, map_location="cpu")
            if cached["S"].shape[0] >= r:
                continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save(_compute_layer_svd(mod.weight, k, seed_name=name), p)
        n_new += 1
    if verbose:
        print(f"[lora_xs] svd cache: {n_new} layers computed, "
              f"rest reused ({svd_cache_dir})")
    return model_id


def _bases_from_cache(cache_dir: str, model_id: str, layer_name: str,
                      rank: int, device, dtype):
    p = _cache_path(cache_dir, model_id, layer_name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"lora_xs svd cache missing for {layer_name}: {p} — call "
            f"build_svd_cache() first (training does this automatically; "
            f"the merge step deliberately does NOT recompute, see module "
            f"docstring)")
    c = torch.load(p, map_location="cpu")
    if c["S"].shape[0] < rank:
        raise ValueError(f"cached K={c['S'].shape[0]} < rank {rank} for "
                         f"{layer_name}; rerun build_svd_cache with the "
                         f"larger rank")
    sq = c["S"][:rank].sqrt()
    A = (c["U"][:, :rank] * sq.unsqueeze(0))            # (out, r)
    B = (sq.unsqueeze(1) * c["Vh"][:rank, :])           # (r, in)
    return A.to(device, dtype), B.to(device, dtype)


def _basis_hash(A: torch.Tensor, B: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(A.float().cpu().numpy().tobytes())
    h.update(B.float().cpu().numpy().tobytes())
    return h.hexdigest()


class LoRAXSAdapter(nn.Module):
    """Wraps a frozen nn.Linear with ΔW = A R B (A, B frozen SVD bases;
    R trainable, zero-init)."""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor,
                 trainable_dtype=torch.float32, update_scale: float = 1.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = A.shape[1]
        self.weight = nn.Parameter(base.weight.detach(), requires_grad=False)
        if base.bias is not None:
            self.register_buffer("bias", base.bias.detach())
        else:
            self.bias = None
        self.register_buffer("A", A)          # (out, r) frozen
        self.register_buffer("B", B)          # (r, in) frozen
        self.R = nn.Parameter(torch.zeros(self.rank, self.rank,
                                          dtype=trainable_dtype,
                                          device=A.device))
        # budget-stabilized update scaling gamma(rho); 1.0 = vanilla
        self.update_scale = float(update_scale)

    def delta_w(self) -> torch.Tensor:
        return (self.A.float() @ self.R.float() @ self.B.float()) \
            * self.update_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x, self.weight, self.bias)
        # adapter branch in the frozen compute dtype (bf16 path parity
        # with rasta.py's N-4 note)
        R = self.R.to(x.dtype)
        d = ((x @ self.B.t().to(x.dtype)) @ R.t()) @ self.A.t().to(x.dtype)
        if self.update_scale != 1.0:
            d = d * self.update_scale
        return y + d


def apply_lora_xs(model: nn.Module,
                  rho: Union[float, str],
                  fixed_rank: Union[None, int, str] = None,
                  target_modules: Optional[List[str]] = None,
                  svd_cache_dir: str = DEFAULT_SVD_CACHE_DIR,
                  frozen_dtype=torch.bfloat16,
                  trainable_dtype=torch.float32,
                  model_id: Optional[str] = None,
                  print_stats: bool = True,
                  update_scale: Optional[float] = None) -> nn.Module:
    """Inject LoRA-XS adapters.  rho is always given — a number, or the
    string "rank:<r>" to DERIVE the budget-matched rho from a shared
    rank (the LoRA-XS paper's own parameterization); fixed_rank
    overrides how the budget is realized — None (default, rho-based per
    layer) | "auto" (budget-matched shared rank; invalid with a
    "rank:<r>" rho spec — circular) | int (explicit uniform rank).  See
    resolve_rank_map / matched_fixed_rank / matched_rho_for_rank.

    update_scale: budget-stabilized scaling gamma on ΔW (mirrors
    rasta.py — required for the controlled DM-vs-XS basis comparison,
    where everything except the bases must match).  None (default)
    auto-derives gamma = sqrt(rho/RHO_REF) ONLY in plain numeric-rho,
    per-layer mode (fixed_rank None, no "rank:<r>" spec); the
    published-LoRA-XS parameterizations (uniform rank / rank spec)
    default to 1.0.  gamma(1) == 1 exactly, so rho=1 adapters are
    bit-identical to the unscaled formulation.  Pass an explicit float
    to override."""
    target_modules = (list(target_modules) if target_modules is not None
                      else list(DEFAULT_TARGET_MODULES))
    rho_spec: Optional[str] = None
    parsed = parse_rho_spec(rho)
    if isinstance(parsed, int):          # "rank:<r>" spec
        if fixed_rank == "auto":
            raise ValueError(
                "rho='rank:<r>' with fixed_rank='auto' is circular — "
                "the rank would be derived from the rho derived from "
                "the rank; give one direction only")
        rho_spec = str(rho)
        rho, _, _ = matched_rho_for_rank(model, parsed, target_modules,
                                         verbose=print_stats)
    else:
        rho = parsed
    if update_scale is None:
        from rasta import update_scale_for
        eff_update_scale = (update_scale_for(rho)
                            if (fixed_rank is None and rho_spec is None)
                            else 1.0)
    else:
        eff_update_scale = float(update_scale)
    rank_map = resolve_rank_map(model, target_modules, rho=rho,
                                fixed_rank=fixed_rank)
    # Computed BEFORE the replace loop below: matched_fixed_rank's dry
    # scan requires nn.Linear instances at the target names, and the
    # loop replaces them with LoRAXSAdapter in place.  Doing this after
    # the replace silently finds zero adaptable layers on the second and
    # later calls in a process (or raises outright, as it did here) —
    # compute the comparison numbers while the model is still all-Linear
    # and only print them afterward.
    _, _parity_total_rho, _parity_total_shared = matched_fixed_rank(
        model, rho, target_modules, verbose=False)
    model_id = build_svd_cache(model, rank_map, target_modules,
                               svd_cache_dir, model_id,
                               verbose=print_stats)
    for p in model.parameters():
        p.requires_grad = False
    n_layers, n_train = 0, 0
    for name, mod in list(model.named_modules()):
        r = rank_map.get(name)
        if r is None:
            continue
        A, B = _bases_from_cache(svd_cache_dir, model_id, name, r,
                                 mod.weight.device, frozen_dtype)
        adapter = LoRAXSAdapter(mod, A, B, trainable_dtype,
                                update_scale=eff_update_scale)
        parent_name = name.rsplit(".", 1)
        parent = (model.get_submodule(parent_name[0])
                  if len(parent_name) == 2 else model)
        setattr(parent, name.split(".")[-1], adapter)
        n_layers += 1
        n_train += adapter.R.numel()
    model._lora_xs_meta = {
        "rho": rho, "rho_spec": rho_spec, "fixed_rank": fixed_rank,
        "update_scale": eff_update_scale,
        "rank_map": rank_map,
        "target_modules": list(target_modules),
        "svd_cache_dir": os.path.abspath(svd_cache_dir),
        "model_id": model_id,
    }
    if print_stats:
        rs = sorted(set(rank_map.values()))
        src = f" (derived from {rho_spec})" if rho_spec else ""
        alloc = (f"rho={rho}{src} (per-layer r in {rs})" if fixed_rank is None
                 else f"rho={rho}{src} fixed_rank='auto' -> r={rs[0]}"
                 if fixed_rank == "auto"
                 else f"fixed_rank={fixed_rank} (rho={rho}{src} recorded only)")
        print(f"[lora_xs] adapted {n_layers} layers, {alloc}, "
              f"trainable={n_train:,}")
        # Allocation-comparison log requested for the DM-vs-XS ablation:
        # rho-based per-layer total vs the shared/uniform total that
        # matches it (computed above, BEFORE the replace loop — see the
        # comment there).  These generally DIFFER — a single shared
        # value can't hit every layer's rho-derived target exactly — so
        # this is printed unconditionally, not only in "auto" mode, so
        # every run reports where it sits relative to the other
        # allocation.
        ratio = (_parity_total_shared / _parity_total_rho
                 if _parity_total_rho else float("nan"))
        print(f"[lora_xs] allocation comparison @ rho={rho}: "
              f"rho-based total={_parity_total_rho:,}  "
              f"shared-V-matched total={_parity_total_shared:,}  "
              f"ratio={ratio:.4f}")
    return model


def save_lora_xs_adapter(model: nn.Module, save_dir: str,
                         training_meta: Optional[Dict] = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    meta = model._lora_xs_meta
    weights, hashes = {}, {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAXSAdapter):
            weights[f"{name}.R"] = mod.R.detach().cpu()
            # Hash the CANONICAL fp32 bases re-derived from the cache —
            # NOT mod.A/mod.B, which live in the training compute dtype
            # (bf16 in real runs).  Hashing the bf16 copies made the
            # stored hash depend on trainer_bf16, so a fresh fp32 merge
            # could never match it.  Cache-derived fp32 on both sides is
            # dtype-invariant and still certifies "these cache factors".
            # Each layer is sliced at ITS OWN rank from rank_map, so the
            # hash certifies the exact factors this layer trained with
            # under either allocation mode.
            A32, B32 = _bases_from_cache(meta["svd_cache_dir"],
                                         meta["model_id"], name,
                                         meta["rank_map"][name], "cpu",
                                         torch.float32)
            hashes[name] = _basis_hash(A32, B32)
    torch.save(weights, os.path.join(save_dir, "adapter_weights.pt"))
    cfg = {"method": "lora_xs", **meta, "basis_hashes": hashes,
           "training_meta": training_meta or {}}
    with open(os.path.join(save_dir, "lora_xs_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_and_merge_lora_xs(model: nn.Module, adapter_dir: str) -> nn.Module:
    cfg = json.load(open(os.path.join(adapter_dir, "lora_xs_config.json")))
    weights = torch.load(os.path.join(adapter_dir, "adapter_weights.pt"),
                         map_location="cpu")
    cache_dir = cfg["svd_cache_dir"]
    model_id, hashes = cfg["model_id"], cfg["basis_hashes"]
    # rank_map is authoritative and always present for adapters saved by
    # this module.  The fallback below is ONLY for adapters saved by the
    # original pre-utils, uniform-only lora_xs.py (config carried "rank",
    # no map) — irrelevant for anything trained after this revision, but
    # kept so those old checkpoints still merge.
    rank_map = cfg.get("rank_map") or {
        name: cfg["rank"] for name in hashes}
    update_scale = float(cfg.get("update_scale", 1.0))
    n = 0
    for name, mod in model.named_modules():
        key = f"{name}.R"
        if key not in weights:
            continue
        r = rank_map[name]
        A, B = _bases_from_cache(cache_dir, model_id, name, r,
                                 "cpu", torch.float32)
        h = _basis_hash(A, B)
        if h != hashes[name]:
            raise RuntimeError(
                f"lora_xs basis hash mismatch for {name}: the cached SVD "
                f"factors are not the ones this adapter trained with "
                f"(cache moved/rebuilt?). Refusing to merge.")
        R = weights[key].float()
        if R.shape != (r, r):
            raise RuntimeError(
                f"lora_xs rank mismatch for {name}: saved R is "
                f"{tuple(R.shape)} but rank_map says {r}")
        dW = ((A @ R @ B) * update_scale).to(mod.weight.dtype) \
            .to(mod.weight.device)
        with torch.no_grad():
            mod.weight += dW
        n += 1
    if n != len(hashes):
        raise RuntimeError(f"merged {n} layers but adapter has {len(hashes)}")
    print(f"[lora_xs merge] merged {n} layers (all basis hashes verified)")
    return model


# ── Prebuild CLI ─────────────────────────────────────────────────────────────
# Optional convenience: warm the cache without launching a training run
# (apply_lora_xs builds it automatically otherwise).

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser(
        description="Prebuild the LoRA-XS SVD cache for a model")
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--rho", type=str, required=True,
                    help="reference budget: a number, or 'rank:<r>' to "
                         "derive the budget-matched rho from a shared "
                         "rank (mirrors apply_lora_xs — see --fixed-rank)")
    ap.add_argument("--fixed-rank", default=None,
                    help='"auto" (budget-matched shared rank), an int '
                         "(explicit uniform rank), or omit for rho-based "
                         "per-layer sizing")
    ap.add_argument("--svd-cache-dir", default=DEFAULT_SVD_CACHE_DIR)
    ap.add_argument("--target-modules", default=None,
                    help="comma-separated; default: all 7 projections")
    args = ap.parse_args()

    tm = (args.target_modules.split(",") if args.target_modules
          else DEFAULT_TARGET_MODULES)
    fixed_rank = args.fixed_rank
    if fixed_rank is not None and fixed_rank != "auto":
        fixed_rank = int(fixed_rank)
    m = AutoModelForCausalLM.from_pretrained(args.model,
                                             torch_dtype=torch.float32)
    parsed = parse_rho_spec(args.rho)
    if isinstance(parsed, int):
        if fixed_rank == "auto":
            raise SystemExit("--rho rank:<r> with --fixed-rank auto is circular")
        rho_val, _, _ = matched_rho_for_rank(m, parsed, tm)
    else:
        rho_val = parsed
    ranks = resolve_rank_map(m, tm, rho=rho_val, fixed_rank=fixed_rank)
    build_svd_cache(m, ranks, tm, args.svd_cache_dir, model_id=args.model)
    rs = sorted(set(ranks.values()))
    print(f"[lora_xs] cache ready for {args.model}: {len(ranks)} layers, "
          f"ranks {rs} → {args.svd_cache_dir}")
