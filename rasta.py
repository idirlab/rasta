"""
rasta.py — RaSTA: Random Subspace Tuning Adaptation.

Unified implementation of Algorithm 1.  One file, two formulations,
shared infrastructure.  (gaussian-only module — see project notes.)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Variant = Literal["dm", "hs"]

_BASIS_CACHE: Dict[Tuple, torch.Tensor] = {}
_DENSE_VIEWS: Dict[Tuple, torch.Tensor] = {}
_RAW_BASIS_CACHE: Dict[Tuple, torch.Tensor] = {}


def clear_rasta_caches() -> None:
    _BASIS_CACHE.clear()
    _DENSE_VIEWS.clear()
    _RAW_BASIS_CACHE.clear()


def _seed_from_key(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "little", signed=False)


def _canon_device(device) -> str:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    return str(dev)


def _floor_pow2(n: int) -> int:
    return 1 if n <= 0 else 1 << (n.bit_length() - 1)


_WHT_MODE = "fused"
_FUSED = {"checked": False, "ok": False}


def set_wht_mode(mode: str) -> None:
    global _WHT_MODE, _FUSED
    if mode not in ("torch", "fused"):
        raise ValueError(f"wht_mode must be 'torch'|'fused', got {mode!r}")
    if mode != _WHT_MODE:
        _FUSED = {"checked": False, "ok": False}
    _WHT_MODE = mode


def get_wht_mode() -> str:
    return _WHT_MODE


def fused_engaged() -> bool:
    return _WHT_MODE == "fused" and _FUSED["ok"]


def prime_fused(device) -> None:
    if _WHT_MODE != "fused":
        return
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return
    probe = torch.randn(8, 256, device=dev, dtype=torch.float32)
    _maybe_check_fused(probe)


def _fast_wht_torch(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    assert (d & (d - 1)) == 0, f"fast_wht: last dim must be power of 2, got {d}"
    h = d >> 1
    while h >= 1:
        s = x.shape[:-1]
        x = x.view(*s, -1, 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.cat([a + b, a - b], dim=-2).view(*s, d)
        h >>= 1
    return x * (d ** -0.5)


def _sylvester(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    assert (d & (d - 1)) == 0, f"WHT dim must be power of 2, got {d}"
    s = x.shape[:-1]
    h = 1
    while h < d:
        x = x.reshape(*s, d // (2 * h), 2 * h)
        a = x[..., :h]
        b = x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1).reshape(*s, d)
        h *= 2
    return x * (d ** -0.5)


def _fused_kernel(x: torch.Tensor) -> torch.Tensor:
    from fast_hadamard_transform import hadamard_transform
    d = x.shape[-1]
    return hadamard_transform(x.contiguous(), scale=d ** -0.5)


def _maybe_check_fused(x: torch.Tensor) -> None:
    if _FUSED["checked"] or not x.is_cuda:
        return
    _FUSED["checked"] = True
    try:
        import fast_hadamard_transform  # noqa: F401
    except Exception as e:
        warnings.warn(
            f"[rasta] wht_mode='fused' but fast_hadamard_transform is "
            f"unavailable ({e}); using the pure-PyTorch Sylvester WHT "
            f"instead (correct, just slower).", stacklevel=2)
        return
    d = 256
    probe = torch.randn(8, d, device=x.device, dtype=torch.float32)
    try:
        got = _fused_kernel(probe.clone()).float()
    except Exception as e:
        warnings.warn(f"[rasta] fused kernel call failed ({e}); using "
                      f"pure-PyTorch Sylvester WHT.", stacklevel=2)
        return
    ref = _sylvester(probe)
    if torch.allclose(ref, got, atol=1e-3, rtol=1e-3):
        _FUSED["ok"] = True
    else:
        warnings.warn(
            "[rasta] fused Hadamard kernel disagrees with the Sylvester "
            "reference -- disabling fused, using pure-PyTorch Sylvester WHT.",
            stacklevel=2)


def fast_wht(x: torch.Tensor) -> torch.Tensor:
    if _WHT_MODE == "torch":
        return _fast_wht_torch(x)
    _maybe_check_fused(x)
    if x.is_cuda and _FUSED["ok"]:
        return _fused_kernel(x)
    return _sylvester(x)


def _gaussian(d: int, V: int, key: str) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(_seed_from_key(key))
    A = torch.randn(d, V, generator=gen, dtype=torch.float32)
    return A / A.norm(dim=0, keepdim=True).clamp(min=1e-12)


def get_basis(key: str, d: int, V: int, dtype: torch.dtype,
             device: torch.device) -> torch.Tensor:
    cache_key = (key, d, V)
    if cache_key not in _BASIS_CACHE:
        _BASIS_CACHE[cache_key] = _gaussian(d, V, key)
    basis = _BASIS_CACHE[cache_key]
    view_key = (key, d, V, str(dtype), _canon_device(device))
    view = _DENSE_VIEWS.get(view_key)
    if view is None:
        view = basis.to(dtype=dtype, device=device)
        _DENSE_VIEWS[view_key] = view
    return view


def get_shared_basis(key: str, dmax: int, d: int, V: int, dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    raw_key = (key, dmax, V)
    if raw_key not in _RAW_BASIS_CACHE:
        gen = torch.Generator()
        gen.manual_seed(_seed_from_key(key))
        _RAW_BASIS_CACHE[raw_key] = torch.randn(dmax, V, generator=gen,
                                                dtype=torch.float32)
    raw = _RAW_BASIS_CACHE[raw_key]
    cache_key = (key, dmax, d, V)
    if cache_key not in _BASIS_CACHE:
        sl = raw[:d]
        _BASIS_CACHE[cache_key] = sl / sl.norm(dim=0, keepdim=True).clamp(min=1e-12)
    basis = _BASIS_CACHE[cache_key]
    view_key = (cache_key, str(dtype), _canon_device(device))
    view = _DENSE_VIEWS.get(view_key)
    if view is None:
        view = basis.to(dtype=dtype, device=device)
        _DENSE_VIEWS[view_key] = view
    return view


def dm_vocab(dout: int, din: int, rho: float) -> int:
    v = math.sqrt(math.sqrt(dout * din) / rho)
    return max(2, int(math.floor(v)))


def hs_vocab(dout: int, din: int, rho: float, min_v: int = 64) -> int:
    v = math.sqrt(dout * din) / (2.0 * rho)
    v = max(float(min_v), v)
    v_rounded = 2 ** round(math.log2(v))
    cap = _floor_pow2(min(dout, din))
    return min(v_rounded, cap)


# ── Budget-stabilized update scaling ─────────────────────────────────────────
# The per-layer parameter budget is P = sqrt(dout*din)/rho for BOTH
# variants, and AdamW's per-parameter step is ~lr regardless of gradient
# magnitude, so the aggregate step on the core (and hence on ΔW) grows
# like sqrt(P) ∝ rho^(-1/2): at a fixed lr, rho=0.5 trains with ~1.4x
# the effective step of rho=1, and rho=4 with ~0.5x.  Scaling the update
# by gamma(rho) = sqrt(rho/RHO_REF) makes the effective step
# budget-invariant — the same repair rsLoRA (Kalajdzievski, 2023)
# applies to LoRA's alpha/r (-> alpha/sqrt(r)), transposed from rank to
# budget.  RHO_REF = 1 anchors the scale so every rho=1 run is
# bit-identical to the unscaled formulation.
RHO_REF = 1.0


def update_scale_for(rho: float) -> float:
    """gamma(rho) = sqrt(rho / RHO_REF); gamma(RHO_REF) == 1 exactly."""
    return math.sqrt(float(rho) / RHO_REF)


def matched_fixed_V(model: nn.Module, rho: float, variant: Variant,
                    target_modules: Optional[List[str]] = None,
                    verbose: bool = True) -> Tuple[int, int, int]:
    shapes: List[Tuple[int, int]] = []

    def _scan(m: nn.Module, pfx: str = "") -> None:
        for name, child in m.named_children():
            full = f"{pfx}.{name}" if pfx else name
            if _matches_target(full, target_modules) and isinstance(child, nn.Linear):
                shapes.append(tuple(child.weight.shape))
            else:
                _scan(child, full)

    _scan(model)

    use_hs = variant == "hs"
    total_r = 0
    adapted: List[Tuple[int, int]] = []
    for dout, din in shapes:
        V = (hs_vocab(dout, din, rho) if use_hs else dm_vocab(dout, din, rho))
        if V > min(dout, din):
            continue
        adapted.append((dout, din))
        total_r += (2 * V) if use_hs else (V * V)

    if not adapted:
        raise ValueError("matched_fixed_V: no target layers found/adaptable.")

    L = len(adapted)
    cap = min(min(dout, din) for dout, din in adapted)

    if use_hs:
        v = total_r / (2.0 * L)
        V_shared = 2 ** round(math.log2(max(v, 1.0)))
        V_shared = min(V_shared, _floor_pow2(cap))
        total_s = 2 * V_shared * L
    else:
        v = math.sqrt(total_r / L)
        V_shared = max(2, int(round(v)))
        V_shared = min(V_shared, cap)
        total_s = V_shared * V_shared * L

    if verbose:
        print(f"  [matched_fixed_V/{variant}] rho={rho} layers={L} "
              f"-> V_shared={V_shared} | budget rho-based={total_r:,} "
              f"shared={total_s:,} (ratio {total_s/total_r:.3f})")
    return V_shared, total_r, total_s


def _matches_target(name: str, target_modules: Optional[List[str]]) -> bool:
    if target_modules is None:
        return True
    return name.rsplit(".", 1)[-1] in target_modules


class _RaSTAAdapterBase(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 W_frozen: torch.Tensor, V: int, left_key: str, right_key: str,
                 left_vocab_max: int, right_vocab_max: int,
                 bias: Optional[torch.Tensor], frozen_dtype: torch.dtype,
                 trainable_dtype: torch.dtype, init_scale: float,
                 shared_lr_basis: bool = False,
                 update_scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.V = V
        self.left_key = left_key
        self.right_key = right_key
        self.left_vocab_max = left_vocab_max
        self.right_vocab_max = right_vocab_max
        self.frozen_dtype = frozen_dtype
        self.trainable_dtype = trainable_dtype
        self.init_scale_cfg = init_scale
        self.shared_lr_basis = shared_lr_basis
        # budget-stabilized update scaling gamma(rho); 1.0 = vanilla
        self.update_scale = float(update_scale)

        self.register_buffer("W_frozen", W_frozen.to(frozen_dtype))
        self.register_buffer(
            "bias", bias.clone().to(frozen_dtype) if bias is not None else None)

        dev = self.W_frozen.device
        self._basis_L(self.frozen_dtype, dev)
        self._basis_R(self.frozen_dtype, dev)

    def _get_basis(self, key, dim, vmax, dt, dev):
        if self.shared_lr_basis:
            dmax = max(self.out_features, self.in_features)
            b = get_shared_basis(key, dmax, dim, vmax, dt, dev)
        else:
            b = get_basis(key, dim, vmax, dt, dev)
        return b if vmax == self.V else b[:, :self.V]

    def _basis_L(self, dt, dev):
        return self._get_basis(self.left_key, self.out_features,
                               self.left_vocab_max, dt, dev)

    def _basis_R(self, dt, dev):
        return self._get_basis(self.right_key, self.in_features,
                               self.right_vocab_max, dt, dev)

    def _base_and_input(self, x: torch.Tensor):
        xf = x.to(self.frozen_dtype)
        h_base = F.linear(xf, self.W_frozen, self.bias)
        return h_base, xf

    def get_weight(self) -> torch.Tensor:
        return self.W_frozen.float() + self.get_delta_W()

    def get_delta_W(self) -> torch.Tensor:
        raise NotImplementedError


class RaSTADMAdapter(_RaSTAAdapterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M = nn.Parameter(
            torch.zeros(self.V, self.V, dtype=self.trainable_dtype,
                        device=self.W_frozen.device))

    def trainable_param_count(self) -> int:
        return self.V * self.V

    def get_delta_W(self) -> torch.Tensor:
        dev = self.W_frozen.device
        BL = self._basis_L(torch.float32, dev)
        BR = self._basis_R(torch.float32, dev)
        return (BL @ self.M.float() @ BR.T) * self.update_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        h_base, xf = self._base_and_input(x)
        dt, dev = xf.dtype, xf.device
        M = self.M.to(dt)
        BR = self._basis_R(dt, dev)
        BL = self._basis_L(dt, dev)
        h = xf @ BR
        h = h @ M.T
        delta = h @ BL.T
        if self.update_scale != 1.0:
            delta = delta * self.update_scale
        return (h_base + delta).to(out_dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear, V: int,
                    left_vocab_max: Optional[int] = None,
                    right_vocab_max: Optional[int] = None,
                    frozen_dtype: torch.dtype = torch.bfloat16,
                    trainable_dtype: torch.dtype = torch.float32,
                    init_scale: float = 0.01,
                    shared_lr_basis: bool = False,
                    update_scale: float = 1.0) -> "RaSTADMAdapter":
        dout, din = linear.weight.shape
        if V > min(dout, din):
            raise ValueError(f"V={V} > min({dout},{din})")
        lvm = left_vocab_max if left_vocab_max is not None else V
        rvm = right_vocab_max if right_vocab_max is not None else V
        if shared_lr_basis:
            lk = rk = f"rasta_dm_LR|dmax={max(dout, din)}|V={V}|bc=gaussian"
            lvm = rvm = V
        else:
            lk = f"rasta_dm_L|dim={dout}|V={lvm}|bc=gaussian"
            rk = f"rasta_dm_R|dim={din}|V={rvm}|bc=gaussian"
        return cls(in_features=din, out_features=dout,
                   W_frozen=linear.weight.data.clone(), V=V, left_key=lk,
                   right_key=rk, left_vocab_max=lvm, right_vocab_max=rvm,
                   bias=linear.bias.data if linear.bias is not None else None,
                   frozen_dtype=frozen_dtype, trainable_dtype=trainable_dtype,
                   init_scale=init_scale, shared_lr_basis=shared_lr_basis,
                   update_scale=update_scale)


class RaSTAHSAdapter(_RaSTAAdapterBase):
    def __init__(self, *args, freeze_side: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert (self.V & (self.V - 1)) == 0, \
            f"HS mixer V must be power of 2, got {self.V}"
        if freeze_side is not None and freeze_side not in ("L", "R"):
            raise ValueError(f"freeze_side must be None|'L'|'R', got {freeze_side!r}")
        self.freeze_side = freeze_side
        _dev = self.W_frozen.device
        if freeze_side == "L":
            self.register_buffer("s_L", torch.ones(self.V, dtype=self.trainable_dtype, device=_dev))
            self.s_R = nn.Parameter(torch.zeros(self.V, dtype=self.trainable_dtype, device=_dev))
        elif freeze_side == "R":
            self.register_buffer("s_R", torch.ones(self.V, dtype=self.trainable_dtype, device=_dev))
            self.s_L = nn.Parameter(torch.zeros(self.V, dtype=self.trainable_dtype, device=_dev))
        else:
            self.s_L = nn.Parameter(torch.zeros(self.V, dtype=self.trainable_dtype, device=_dev))
            self.s_R = nn.Parameter(
                torch.randn(self.V, dtype=self.trainable_dtype, device=_dev) * self.init_scale_cfg)

    def trainable_param_count(self) -> int:
        return self.V if self.freeze_side else 2 * self.V

    def get_delta_W(self) -> torch.Tensor:
        dev = self.W_frozen.device
        BL = self._basis_L(torch.float32, dev)
        BR = self._basis_R(torch.float32, dev)
        h = BR * self.s_R.float()
        h = fast_wht(h)
        h = h * self.s_L.float()
        return (BL @ h.T) * self.update_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        h_base, xf = self._base_and_input(x)
        dt, dev = xf.dtype, xf.device
        s_L = self.s_L.to(dt)
        s_R = self.s_R.to(dt)
        BR = self._basis_R(dt, dev)
        BL = self._basis_L(dt, dev)
        h = xf @ BR
        h = h * s_R
        h = fast_wht(h)
        h = h * s_L
        delta = h @ BL.T
        if self.update_scale != 1.0:
            delta = delta * self.update_scale
        return (h_base + delta).to(out_dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear, V: int,
                    left_vocab_max: Optional[int] = None,
                    right_vocab_max: Optional[int] = None,
                    frozen_dtype: torch.dtype = torch.bfloat16,
                    trainable_dtype: torch.dtype = torch.float32,
                    init_scale: float = 0.01,
                    shared_lr_basis: bool = False,
                    freeze_side: Optional[str] = None,
                    update_scale: float = 1.0) -> "RaSTAHSAdapter":
        dout, din = linear.weight.shape
        if V > min(dout, din):
            raise ValueError(f"V={V} > min({dout},{din})")
        lvm = left_vocab_max if left_vocab_max is not None else V
        rvm = right_vocab_max if right_vocab_max is not None else V
        if shared_lr_basis:
            lk = rk = f"rasta_hs_LR|dmax={max(dout, din)}|V={V}|bc=gaussian"
            lvm = rvm = V
        else:
            lk = f"rasta_hs_L|dim={dout}|V={lvm}|bc=gaussian"
            rk = f"rasta_hs_R|dim={din}|V={rvm}|bc=gaussian"
        return cls(in_features=din, out_features=dout,
                   W_frozen=linear.weight.data.clone(), V=V, left_key=lk,
                   right_key=rk, left_vocab_max=lvm, right_vocab_max=rvm,
                   bias=linear.bias.data if linear.bias is not None else None,
                   frozen_dtype=frozen_dtype, trainable_dtype=trainable_dtype,
                   init_scale=init_scale, shared_lr_basis=shared_lr_basis,
                   freeze_side=freeze_side, update_scale=update_scale)


def apply_rasta(model: nn.Module, rho: float, variant: Variant,
                target_modules: Optional[List[str]] = None,
                frozen_dtype: torch.dtype = torch.bfloat16,
                trainable_dtype: torch.dtype = torch.float32,
                init_scale: float = 0.01, fixed_V: Optional[int] = None,
                wht_mode: str = "fused", shared_lr_basis: bool = False,
                freeze_side: Optional[str] = None, print_stats: bool = True,
                verbose: bool = True,
                update_scale: Optional[float] = None) -> nn.Module:
    """update_scale: budget-stabilized scaling gamma on ΔW.  None
    (default) auto-derives gamma = sqrt(rho/RHO_REF) from the rho
    budget — exactly 1.0 at rho = RHO_REF = 1, so rho=1 runs are
    bit-identical to the unscaled formulation.  With fixed_V (budget
    semantics overridden) auto is 1.0.  Pass an explicit float (e.g.
    1.0) to override — used by the scaling ablation."""
    if variant not in ("dm", "hs"):
        raise ValueError(f"variant must be 'dm'|'hs', got {variant!r}")
    use_hs = variant == "hs"
    if freeze_side is not None and not use_hs:
        raise ValueError("freeze_side is only meaningful for variant='hs'")
    set_wht_mode(wht_mode)
    try:
        prime_fused(next(model.parameters()).device)
    except StopIteration:
        pass

    if fixed_V is not None:
        if fixed_V < 2:
            raise ValueError(f"fixed_V must be >= 2, got {fixed_V}")
        if use_hs and (fixed_V & (fixed_V - 1)) != 0:
            raise ValueError(f"fixed_V must be a power of 2 for variant='hs', got {fixed_V}")

    if update_scale is None:
        eff_update_scale = (1.0 if fixed_V is not None
                            else update_scale_for(rho))
    else:
        eff_update_scale = float(update_scale)

    def _compute_V(dout: int, din: int) -> int:
        if fixed_V is not None:
            return fixed_V
        return (hs_vocab(dout, din, rho) if use_hs else dm_vocab(dout, din, rho))

    vmax_L: Dict[int, int] = {}
    vmax_R: Dict[int, int] = {}

    def _scan(m: nn.Module, pfx: str = "") -> None:
        for name, child in m.named_children():
            full = f"{pfx}.{name}" if pfx else name
            if _matches_target(full, target_modules) and isinstance(child, nn.Linear):
                dout, din = child.weight.shape
                V = _compute_V(dout, din)
                if V <= min(dout, din):
                    vmax_L[dout] = max(vmax_L.get(dout, 0), V)
                    vmax_R[din] = max(vmax_R.get(din, 0), V)
            else:
                _scan(child, full)

    _scan(model)

    replaced, skipped = 0, []

    def _recurse(m: nn.Module, pfx: str = "") -> None:
        nonlocal replaced
        for name, child in list(m.named_children()):
            full = f"{pfx}.{name}" if pfx else name
            if _matches_target(full, target_modules) and isinstance(child, nn.Linear):
                try:
                    dout, din = child.weight.shape
                    V = _compute_V(dout, din)
                    kwargs = dict(V=V, left_vocab_max=vmax_L.get(dout, V),
                                 right_vocab_max=vmax_R.get(din, V),
                                 frozen_dtype=frozen_dtype,
                                 trainable_dtype=trainable_dtype,
                                 init_scale=init_scale,
                                 shared_lr_basis=shared_lr_basis,
                                 update_scale=eff_update_scale)
                    if use_hs:
                        new = RaSTAHSAdapter.from_linear(child, freeze_side=freeze_side, **kwargs)
                    else:
                        new = RaSTADMAdapter.from_linear(child, **kwargs)
                    setattr(m, name, new)
                    replaced += 1
                    if verbose:
                        lvm, rvm = new.left_vocab_max, new.right_vocab_max
                        slice_tag = (f" [slice {V}/{lvm}x{rvm}]" if V < lvm or V < rvm else "")
                        print(f"  [RaSTA-{variant.upper()}] {full} ({din}->{dout})"
                              f"  V={V}{slice_tag}  params={new.trainable_param_count()}")
                except ValueError as e:
                    skipped.append((full, str(e)))
            else:
                _recurse(child, full)

    _recurse(model)

    for p in model.parameters():
        p.requires_grad_(False)
    AdapterCls = RaSTAHSAdapter if use_hs else RaSTADMAdapter
    for mod in model.modules():
        if isinstance(mod, AdapterCls):
            if use_hs:
                if isinstance(mod.s_L, nn.Parameter):
                    mod.s_L.requires_grad_(True)
                if isinstance(mod.s_R, nn.Parameter):
                    mod.s_R.requires_grad_(True)
            else:
                mod.M.requires_grad_(True)

    if skipped and verbose:
        print(f"  [RaSTA] Skipped {len(skipped)}: {[n for n, _ in skipped[:5]]}")
    if print_stats:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dense_mb = sum(t.numel() * t.element_size() for t in _DENSE_VIEWS.values()) / 1e6
        mode = variant.upper()
        v_tag = f" fixed_V={fixed_V}" if fixed_V is not None else ""
        print(f"\n  [RaSTA-{mode}] Replaced {replaced} | rho={rho}{v_tag} wht={wht_mode}\n"
              f"  [RaSTA-{mode}] Trainable: {trainable:,} / {total:,} "
              f"({100.*trainable/total:.4f}%)\n"
              f"  [RaSTA-{mode}] Shared basis views: {len(_DENSE_VIEWS)} tensors, "
              f"{dense_mb:.1f} MB total (N-9)")
    return model


def _collect_hashes(model: nn.Module) -> Dict[str, str]:
    _dev = torch.device("cpu")
    _dt = torch.float32
    hashes: Dict[str, str] = {}
    for mod in model.modules():
        if not isinstance(mod, (RaSTADMAdapter, RaSTAHSAdapter)):
            continue
        shared = getattr(mod, "shared_lr_basis", False)
        dmax = max(mod.out_features, mod.in_features)
        for key, dim, vmax in (
            (mod.left_key, mod.out_features, mod.left_vocab_max),
            (mod.right_key, mod.in_features, mod.right_vocab_max),
        ):
            hkey = f"{key}|d={dim}" if shared else key
            if hkey not in hashes:
                mat = (get_shared_basis(key, dmax, dim, vmax, _dt, _dev)
                       if shared else get_basis(key, dim, vmax, _dt, _dev))
                hashes[hkey] = hashlib.sha256(mat.detach().numpy().tobytes()).hexdigest()
    return hashes


def _verify_hashes(model: nn.Module, saved: Dict[str, str]) -> None:
    if not saved:
        warnings.warn("No basis_hashes in checkpoint -- cannot verify.", stacklevel=3)
        return
    current = _collect_hashes(model)
    missing = [k for k in saved if k not in current]
    if missing:
        raise RuntimeError(f"Basis keys in checkpoint not present in reconstructed model "
                           f"({len(missing)}): {missing[:3]}. Config/model mismatch.")
    bad = [k for k, h in saved.items() if current[k] != h]
    if bad:
        raise RuntimeError(f"Basis hash mismatch for {len(bad)} key(s): {bad[:3]}.")
    print(f"  [merge] basis_hashes verified: {len(saved)} keys all match")


def save_rasta_adapter(model: nn.Module, save_dir: str, target_modules: List[str],
                       rho: float, variant: Variant, init_scale: float = 0.01,
                       fixed_V: Optional[int] = None, shared_lr_basis: bool = False,
                       freeze_side: Optional[str] = None,
                       training_meta: Optional[dict] = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    state: Dict[str, torch.Tensor] = {}
    AdapterCls = RaSTAHSAdapter if variant == "hs" else RaSTADMAdapter
    for mname, mod in model.named_modules():
        if isinstance(mod, AdapterCls):
            for pname, param in mod.named_parameters(recurse=False):
                if param.requires_grad:
                    state[f"{mname}.{pname}"] = param.detach().cpu()

    _scale = 1.0
    for _m in model.modules():
        if isinstance(_m, AdapterCls):
            _scale = float(getattr(_m, "update_scale", 1.0))
            break
    cfg: Dict = {"rho": rho, "variant": variant, "init_scale": init_scale,
                "update_scale": _scale,
                "fixed_V": fixed_V, "wht_mode": get_wht_mode(),
                "shared_lr_basis": shared_lr_basis, "freeze_side": freeze_side,
                "target_modules": target_modules, "basis_hashes": _collect_hashes(model)}
    if training_meta:
        cfg["_training_meta"] = training_meta

    torch.save(state, os.path.join(save_dir, "adapter_weights.pt"))
    with open(os.path.join(save_dir, "rasta_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    n_params = sum(v.numel() for v in state.values())
    print(f"  [save] {len(state)} tensors ({n_params:,} scalars) -> {save_dir}")


def load_and_merge_rasta(base_model: nn.Module, save_dir: str) -> nn.Module:
    with open(os.path.join(save_dir, "rasta_config.json")) as f:
        cfg = json.load(f)

    adapted = apply_rasta(model=base_model, rho=cfg["rho"], variant=cfg["variant"],
                          init_scale=cfg.get("init_scale", 0.01),
                          update_scale=cfg.get("update_scale", 1.0),
                          fixed_V=cfg.get("fixed_V", None), wht_mode=cfg["wht_mode"],
                          shared_lr_basis=cfg.get("shared_lr_basis", False),
                          freeze_side=cfg.get("freeze_side", None),
                          target_modules=cfg["target_modules"],
                          frozen_dtype=next(base_model.parameters()).dtype,
                          trainable_dtype=torch.float32, print_stats=False, verbose=False)
    _verify_hashes(adapted, cfg.get("basis_hashes", {}))

    state = torch.load(os.path.join(save_dir, "adapter_weights.pt"), map_location="cpu")

    model_params = set(adapted.state_dict().keys())
    matched = [k for k in state if k in model_params]
    unmatched = [k for k in state if k not in model_params]
    print(f"  [merge] adapter_weights: {len(state)} keys, {len(matched)} matched, "
          f"{len(unmatched)} unmatched")
    if unmatched:
        print(f"  [merge] WARNING -- unmatched keys (first 5): {unmatched[:5]}")
    if not matched:
        raise RuntimeError("[merge] No adapter keys matched the adapted model state_dict.")

    adapted.load_state_dict(state, strict=False)

    AdapterCls = RaSTAHSAdapter if cfg["variant"] == "hs" else RaSTADMAdapter

    _delta_norms = []
    with torch.no_grad():
        for mod in adapted.modules():
            if isinstance(mod, AdapterCls):
                _delta_norms.append(mod.get_delta_W().norm().item())
                if len(_delta_norms) >= 4:
                    break
    if _delta_norms:
        print(f"  [merge] ||ΔW||_F sample (first {len(_delta_norms)} layers): "
              f"{[f'{v:.6f}' for v in _delta_norms]}")
        if max(_delta_norms) < 1e-10:
            raise RuntimeError("[merge] ΔW is essentially zero -- adapter weights did not load.")

    def _merge(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, AdapterCls):
                with torch.no_grad():
                    merged_w = child.get_weight().to(child.W_frozen.dtype)
                has_bias = child.bias is not None
                lin = nn.Linear(child.in_features, child.out_features, bias=has_bias,
                                device=merged_w.device, dtype=merged_w.dtype)
                with torch.no_grad():
                    lin.weight.copy_(merged_w)
                    if has_bias:
                        lin.bias.copy_(child.bias.to(merged_w.dtype))
                setattr(parent, name, lin)
            else:
                _merge(child)

    _merge(adapted)
    return adapted
