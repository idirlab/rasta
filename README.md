# RaSTA: Random Subspace Tuning Adaptation

Official implementation of **RaSTA** (**Ra**ndom **S**ubspace **T**uning **A**daptation), a
parameter-efficient fine-tuning method that learns a small mixing core between **frozen random
bases shared across layers**. RaSTA's trainable budget is decoupled from layer width: at its
largest setting it uses *under half* the parameters of rank-one LoRA, and it stays competitive
with LoRA down to roughly a tenth of that.

> **RaSTA: Random Subspace Tuning Adaptation** — [paper link TBD]

**Highlights**

- 🧊 **Frozen, shared, seed-regenerable bases.** No per-weight SVD, no bases in the checkpoint —
  adapters store only the learned core (≈1M scalars for a 7–8B model at ρ = 1).
- 🔀 **Two variants, one knob.** RaSTA-HS (Hadamard scaling, our main variant) and RaSTA-DM
  (dense mixing), both sized by a single compression hyperparameter `ρ`.
- 🪶 **Low memory.** Trains at sequence length 2048 on an 80 GiB GPU where full fine-tuning,
  every LoRA rank, and VeRA run out of memory.
- ⚡ **No inference overhead.** The update merges into the base weight; merged models are
  plain `nn.Linear`.

---

## Method

<p align="center">
  <img src="assets/_RaSTA.png" width="100%" alt="RaSTA in relation to LoRA, VeRA and LoRA-XS">
</p>
<p align="center"><em>
How each method factorizes the additive update to a frozen weight W (m×n).
Frozen components in blue, trainable in orange.
</em></p>

RaSTA expresses the update to a frozen pretrained weight $W \in \mathbb{R}^{m\times n}$ as

$$
\Delta W \;=\; \gamma \, B_L \, \tilde{M} \, B_R^{\top},
$$

where $B_L \in \mathbb{R}^{m\times k}$ and $B_R \in \mathbb{R}^{n\times k}$ are frozen random
Gaussian bases (unit-norm columns) that fix the available left and right directions,
$\tilde{M}$ is a small **mixing core** — entry $\tilde{M}_{ij}$ weights the outer product of the
$i$-th left and $j$-th right basis vector — and $\gamma$ is a fixed update scale. Where LoRA
learns *which* row and column spaces to update, RaSTA keeps those spaces fixed and random, and
learns only *how to combine* directions inside them. The two variants differ in how they spend
the parameter budget on $\tilde{M}$.

### RaSTA-HS — Hadamard scaling (main variant)

$$
\tilde{M} \;=\; \operatorname{diag}(s_L)\; H_V \;\operatorname{diag}(s_R),
\qquad s_L, s_R \in \mathbb{R}^{V}\ \ (2V \text{ trainable parameters})
$$

The core is a **fixed** normalized Walsh–Hadamard matrix $H_V$ sandwiched between two
**learned** diagonal scalings. Each core entry is $\tilde{M}_{ij} = s_{L,i}\,(H_V)_{ij}\,s_{R,j}$,
so:

- **Every left direction is coupled to every right direction.** $H_V$ is dense with
  equal-magnitude entries $\pm 1/\sqrt{V}$, so all $V^2$ direction pairs interact with the same
  strength before the learned scalings act. Contrast VeRA, whose core is *diagonal*: left
  direction $i$ only ever pairs with right direction $i$.
- **The mixing is uniform and non-redundant.** $H_V$ is orthogonal ($H_V H_V^\top = I$), so the
  scalings act on a well-conditioned set of directions rather than a few dominant ones.
- **Many directions for very few parameters.** Because only $2V$ scalars are learned, $V$ can be
  large (e.g. 1024 on a 4096×4096 layer), and since $H_V$ is full-rank the update can reach rank
  up to $V$ — versus rank $r$ for LoRA or $k$ for dense-core methods at a similar budget.
- **No dense matrix is stored or multiplied.** $H_V$ is applied with the fast Walsh–Hadamard
  transform in $\mathcal{O}(V \log V)$, via a fused CUDA kernel when available.

Initialization: $s_L = 0$ and $s_R \sim \mathcal{N}(0, 0.01^2)$, so $\Delta W = 0$ at the start of
training. To our knowledge this is the first use of the Walsh–Hadamard transform *inside the
mixing core* of a frozen-basis adapter (prior work uses fixed transforms over a weight's
coordinates to synthesize updates from sparse spectra).

### RaSTA-DM — dense mixing

$$
\tilde{M} \;=\; M \in \mathbb{R}^{k\times k}\qquad (k^2 \text{ trainable parameters, } M \text{ initialized to } 0)
$$

Every entry of the core is learned: unrestricted pairwise interactions, but among far fewer
directions. RaSTA-DM shares LoRA-XS's dense-core structure and differs **only** in the bases —
shared random instead of a truncated SVD of each weight ($B_L = U_k\Sigma_k$, $B_R = V_k$).
In our experiments the two perform comparably on average, suggesting spectral bases are not
necessary for this family.

### The expressivity–coverage trade-off

The core structures form a spectrum: diagonal (VeRA) → fixed dense coupling with learned
scalings (RaSTA-HS) → fully learned dense coupling (RaSTA-DM / LoRA-XS). At a matched budget,
RaSTA-DM buys **expressivity** (free mixing of a few directions) while RaSTA-HS buys
**coverage** (structured mixing across many). Concretely, for a 4096×4096 layer:

| Method | Update $\Delta W$ | Trainable | Bases | Params (per matrix) | Directions / side | 4096×4096 example |
|---|---|---|---|---|---|---|
| LoRA | $BA$ | $B, A$ | learned | $r(m+n)$ | $r$ | r = 1 → **8,192** params, rank 1 |
| VeRA | $\operatorname{diag}(s_L) B_L \operatorname{diag}(s_R) B_R^\top$ | $s_L, s_R$ | shared random | $m + V$ | $V$ | V = 1024 → **5,120** params |
| LoRA-XS | $B_L M B_R^\top$ | $M$ | per-weight SVD | $k^2$ | $k$ | ρ = 2 → **2,025** params, 45 dirs |
| RaSTA-DM | $\gamma B_L M B_R^\top$ | $M$ | shared random | $k^2$ | $k$ | ρ = 2 → **2,025** params, 45 dirs |
| **RaSTA-HS** | $\gamma B_L \operatorname{diag}(s_L) H_V \operatorname{diag}(s_R) B_R^\top$ | $s_L, s_R$ | shared random | $2V$ | $V$ | ρ = 2 → **2,048** params, 1024 dirs |

### Two design choices

**Size-aware budget `ρ`.** Each $m\times n$ weight gets $\approx \sqrt{mn}/\rho$ trainable
parameters, so larger layers get more parameters (as LoRA does implicitly) and one knob sizes the
whole model. Larger `ρ` = smaller adapter. The widths are

$$
k = \Big\lfloor \big(\sqrt{mn}/\rho\big)^{1/2} \Big\rfloor,
\qquad
V = \min\!\Big(2^{\,\mathrm{round}(\log_2 \frac{\sqrt{mn}}{2\rho})},\; 2^{\lfloor \log_2 \min(m,n)\rfloor}\Big),
$$

giving $k^2 \approx 2V \approx \sqrt{mn}/\rho$. $V$ is a power of two (required by $H_V$) and
capped so directions never outnumber the layer's rank. At ρ = 1 the budget is
$\sqrt{mn} \le (m+n)/2$ — at most half of rank-one LoRA.

**Update scaling `γ = √ρ`.** Shrinking the budget also shrinks the magnitude of the unscaled
update; $\gamma = \sqrt{\rho}$ compensates, so a **single learning rate per variant works at every
budget** (analogous in spirit to rank-stabilized LoRA). With scaling, loss curves at different ρ
become near-parallel; without it, larger ρ trains visibly slower — especially for RaSTA-DM:

<p align="center">
  <img src="assets/loss_curves_llama2-7b.png" width="70%" alt="Training loss with and without update scaling">
</p>
<p align="center"><em>
Training loss (EMA) on Llama-2-7B for RaSTA-HS (left) and RaSTA-DM (right), with γ = √ρ (solid)
and without (dashed). ρ = 1 is identical in both cases.
</em></p>

**Basis sharing and regeneration.** Bases are sampled deterministically from a seed key
(side, dimension, width) and shared by every layer with the same basis shape. Adapter
checkpoints store only the learned core plus the seed keys; bases are regenerated (and verified by
hash) when reloading or merging.

---

## Results

Across math, commonsense, code, and GLUE, on Llama-2-7B, Mistral-7B, Llama-3-8B and
RoBERTa-large, both variants stay competitive with low-rank LoRA and degrade gracefully as the
budget shrinks. RaSTA-HS is the strongest frozen-basis method at every budget: its ρ = 2
configuration matches or exceeds RaSTA-DM and LoRA-XS at ρ = 1 with roughly half the parameters.

<p align="center">
  <img src="assets/rho_relative_all.png" width="90%" alt="Frozen-basis methods across budgets">
</p>
<p align="center"><em>
Fraction of the best frozen-basis score retained at each budget, per task (normalized within
each model/task cell, averaged over models). Dashed lines: LoRA r = 1 and VeRA.
</em></p>

**Llama-3-8B, instruction tuning** (accuracy / pass@1; CS = commonsense 8-task average):

| Method | Params | GSM8K | MATH | HumanEval | MBPP | CS Avg. |
|---|---:|---:|---:|---:|---:|---:|
| LoRA r = 1 | 2.62M | 71.27 | 21.60 | 46.34 | 67.72 | 85.94 |
| VeRA V = 1024 | 1.61M | 73.31 | 21.60 | 49.39 | 63.76 | 85.75 |
| LoRA-XS ρ = 1 | 1.12M | 68.76 | 21.80 | 48.17 | 60.32 | 85.34 |
| RaSTA-DM ρ = 1 | 1.12M | 69.60 | 19.00 | 46.95 | 62.96 | 84.16 |
| **RaSTA-HS ρ = 1** | 1.18M | 71.11 | 20.80 | 47.56 | 66.67 | **86.33** |
| **RaSTA-HS ρ = 2** | 590K | 69.98 | 20.40 | 47.56 | 63.49 | 85.83 |
| **RaSTA-HS ρ = 4** | 295K | 68.46 | 17.20 | 48.78 | 66.67 | 84.92 |

**RoBERTa-large, GLUE** (8-task average; params exclude the classification head):

| Method | Params | Avg. |
|---|---:|---:|
| LoRA r = 1 | 98K | 87.56 |
| VeRA V = 256 | 61.4K | 85.81 |
| LoRA-XS ρ = 1 | 49.2K | 86.26 |
| RaSTA-DM ρ = 1 | 49.2K | 84.91 |
| **RaSTA-HS ρ = 1** | 49.2K | **87.13** |
| **RaSTA-HS ρ = 4** | 12.3K | 85.83 |

All numbers are single runs at seed 42; in preliminary multi-seed runs we saw 1–2 point spreads
on GSM8K, so differences of that size are within run-to-run variation. Full per-model and
per-dataset tables are in the paper.

---

## Efficiency

Single optimization step on Mistral-7B (batch 4, bf16 base, fp32 trainable parameters, AdamW;
median of 30 timed steps after 10 warm-up). "—" = out of memory on an 80 GiB H100.

| Method | Peak mem @ seq 512 | Step time @ seq 512 | Peak mem @ seq 2048 |
|---|---:|---:|---:|
| Full fine-tuning | 67.6 GiB | 208 ms | — |
| LoRA r = 1 / 8 / 64 | 32.5 / 32.7 / 35.1 GiB | 251 / 274 / 298 ms | — |
| VeRA V = 1024 | 35.3 GiB | 701 ms | — |
| LoRA-XS ρ = 1 | 23.4 GiB | 198 ms | 51.6 GiB |
| RaSTA-DM ρ = 1 | 23.0 GiB | 202 ms | 51.3 GiB |
| RaSTA-HS ρ = 1 / 2 / 4 | 27.7 / 25.3 / 24.1 GiB | 304 / 241 / 211 ms | 69.3 / 60.2 / 55.6 GiB |

Both RaSTA variants use less memory than every LoRA and VeRA configuration, and the gap widens
with workload — the advantage is largest in the long-context, large-batch regime where the memory
ceiling binds.

<p align="center">
  <img src="assets/memory_len.png" width="32%" alt="Peak memory vs sequence length">
  <img src="assets/memory_batch.png" width="32%" alt="Peak memory vs batch size">
  <img src="assets/time.png" width="32%" alt="Step time vs sequence length">
</p>
<p align="center"><em>
Mistral-7B: peak memory vs. sequence length (batch 4), peak memory vs. batch size (seq 512),
and step time vs. sequence length (batch 4). Dotted line: 80 GiB ceiling.
</em></p>

---

## Repository layout

```
.
├── rasta.py                  # RaSTA implementation (both variants, save/merge utilities)
├── utils/
│   ├── train_rasta.py        # trainer for RaSTA-DM / RaSTA-HS
│   ├── train_loraxs.py       # trainer for LoRA-XS (within RaSTA's budget/scaling framework)
│   ├── train_lora.py         # trainer for LoRA
│   ├── train_vera.py         # trainer for VeRA
│   └── ...                   # shared training/eval utilities
├── Math/                     # math task: data module + GSM8K/MATH evaluation
├── Code/                     # code task: data module + HumanEval/MBPP (EvalPlus) evaluation
├── Commonsense/              # commonsense task: data module + 8-dataset evaluation
├── Glue/                     # GLUE: train_glue.py (RoBERTa-large, train+eval in one stage)
├── assets/                   # figures used in this README
├── run_math.sh               # single-run driver: train + eval, one method/model/config
├── run_code.sh
├── run_commonsense.sh
├── run_glue.sh
└── requirements.txt
```

## Installation

```bash
git clone <repo-url> && cd <repo>
pip install -r requirements.txt
```

Main dependencies: PyTorch (bf16-capable GPU recommended), `transformers`, `datasets`,
`peft >= 0.10` (VeRA support), and `evalplus` (code evaluation only).

**Optional but recommended for RaSTA-HS:**

```bash
pip install fast-hadamard-transform
```

This enables the fused Walsh–Hadamard CUDA kernel (up to ~2× faster whole-model steps for
RaSTA-HS). Without it, RaSTA-HS **automatically falls back** to a pure-PyTorch butterfly
implementation — identical results, slower. The kernel is self-checked against the reference
implementation at first use.

## Quick start

Each task has a single self-contained driver:

```bash
bash run_<task>.sh <method> <model> <config> [lr] [seed]
```

```bash
# RaSTA-HS at rho=1 on Mistral-7B, math (train on MetaMathQA, eval GSM8K + MATH)
bash run_math.sh rasta-hs mistralai/Mistral-7B-v0.1 rho1

# RaSTA-DM at rho=2 on Llama-3-8B, code (train on CodeFeedback, eval HumanEval + MBPP)
bash run_code.sh rasta-dm meta-llama/Meta-Llama-3-8B rho2

# LoRA r=8 baseline on Llama-2-7B, math
bash run_math.sh lora meta-llama/Llama-2-7b-hf r8

# Commonsense (Llama-3-8B)
bash run_commonsense.sh rasta-hs meta-llama/Meta-Llama-3-8B rho1

# GLUE (RoBERTa-large): bash run_glue.sh <task> <method> <config>
bash run_glue.sh rte rasta-hs rho1
```

- `method` ∈ `rasta-hs | rasta-dm | lora-xs | lora | vera`
- `config` is `rho<value>` for rasta-hs / rasta-dm / lora-xs, and `r<value>` for lora / vera
  (for VeRA, `r<value>` sets the basis width V, e.g. `r1024`)
- `lr` defaults to the per-method value used in the paper (see below); `seed` defaults to 42
- Results land in `results/<run_name>/eval_results.json`

Hardware knobs are environment variables: `N_GPUS` (default 2, via `torchrun`), `GLOBAL_BATCH`
(default 128), `PER_DEVICE_BATCH` (default 8; use 4 for RaSTA-HS on 7–8B models). Gradient
accumulation is derived automatically. The paper's runs fit on a single H100-80GB.

## Using RaSTA in your own code

```python
import torch
from transformers import AutoModelForCausalLM
from rasta import apply_rasta, save_rasta_adapter, load_and_merge_rasta

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                             torch_dtype=torch.bfloat16)

# Wrap target linear layers with RaSTA adapters (freezes everything else).
target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model = apply_rasta(model, rho=1.0, variant="hs", target_modules=target)
# variant="dm" for the dense-mixing variant; the update scale γ=√ρ is applied automatically.

# ... train the (few) parameters with requires_grad=True ...

# Save only the trained core parameters (+ config and basis hashes) — bases are regenerated.
save_rasta_adapter(model, "my_adapter/", target_modules=target, rho=1.0, variant="hs")

# Later: rebuild from a fresh base model, verify bases, and merge into plain Linear layers.
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                            torch_dtype=torch.bfloat16)
merged = load_and_merge_rasta(base, "my_adapter/")   # no inference-time overhead
```

Notes:

- `wht_mode="fused"` (default) uses the CUDA kernel when available and silently falls back to
  the PyTorch reference otherwise.
- Bases are cached and shared across layers of compatible shape, so they cost memory once per
  shape, not once per layer.

## Reproducing the paper

**Grid.** All main results are single runs at seed 42. Instruction tuning uses models
`{Llama-2-7B, Mistral-7B, Llama-3-8B}` (commonsense: Llama-3-8B only); GLUE uses RoBERTa-large.

| Method | Instruction tuning | GLUE |
|---|---|---|
| RaSTA-HS / RaSTA-DM / LoRA-XS | ρ ∈ {1, 2, 4} | ρ ∈ {1, 2, 4} |
| LoRA (α = 2r) | r ∈ {1, 8, 64} | r ∈ {1, 2, 8} |
| VeRA | V = 1024 | V = 256 |

LoRA-XS uses the same per-layer allocation and update scale as RaSTA-DM, so the two differ only
in their frozen bases.

**Learning rates** (fixed per method across models and budgets; the scripts' defaults):

| Method | Math / Code | Commonsense / GLUE |
|---|---|---|
| RaSTA-HS | 3e-2 | 2e-2 |
| RaSTA-DM | 4e-3 | 2e-3 |
| LoRA-XS | 4e-3 | 2e-3 |
| LoRA | 2e-4 | 2e-4 |
| VeRA | 2e-2 | 2e-2 |

**Protocol.**

- *Instruction tuning:* adapters on all attention + MLP linear projections; 1 epoch, effective
  batch 128, AdamW, cosine schedule with 0.03 warmup, bf16 base weights with fp32 trainable
  parameters, Alpaca prompt format, greedy decoding at evaluation.
- *GLUE:* adapters on query/value projections of RoBERTa-large; fp32, sequence length 128,
  batch 32, AdamW with a linear schedule and 0.06 warmup. The classification head trains in
  every run at a fixed lr of 5e-4. `train_glue.py` evaluates the dev set each epoch and reports
  the best-epoch metric (per-task epoch counts are set by its defaults).
- RaSTA and LoRA-XS use `γ = √ρ` throughout.

To reproduce a full table, loop the driver over the grid, e.g.:

```bash
for model in mistralai/Mistral-7B-v0.1 meta-llama/Llama-2-7b-hf meta-llama/Meta-Llama-3-8B; do
  for cfg in rho1 rho2 rho4; do
    bash run_math.sh rasta-hs "$model" "$cfg"
  done
done
```

### Data

| Task | Train | Eval | Setup |
|---|---|---|---|
| Math | MetaMathQA (100k samples) | GSM8K, MATH | auto-downloaded from the HF Hub |
| Code | CodeFeedback python split (via `fxmeng/pissa-dataset`) | HumanEval, MBPP via EvalPlus | auto-downloaded; `pip install evalplus` |
| Commonsense | Commonsense-170K | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA | included in the repo under `Commonsense/data/commonsense/` |
| GLUE | per-task GLUE train sets | per-task dev sets | auto-downloaded via `datasets` |

## Citation

```bibtex
@article{rasta2026,
  title  = {RaSTA: Random Subspace Tuning Adaptation},
  author = {TBD},
  year   = {2026}
}
```
