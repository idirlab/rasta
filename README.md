# RaSTA: Random Subspace Tuning Adaptation

Official implementation of **RaSTA** (**Ra**ndom **S**ubspace **T**uning **A**daptation), a
parameter-efficient fine-tuning method that learns a small mixing core between frozen random
bases shared across layers.

> **RaSTA: Random Subspace Tuning Adaptation** — [paper link TBD]

## Method at a glance

RaSTA expresses the additive update to a frozen pretrained weight `W (m×n)` as

```
ΔW = γ · B_L · M̃ · B_Rᵀ
```

where `B_L (m×k)` and `B_R (n×k)` are **frozen random Gaussian bases shared across compatible
layers**, and only the small core `M̃` is trained. In relation to its predecessors:

```
LoRA:       ΔW = B A                                (trains B, A;  r(m+n) params)
VeRA:       ΔW = Λ_b B Λ_d A                        (frozen shared random bases; trains 2 scaling vectors)
LoRA-XS:    ΔW = B M A                              (frozen per-weight SVD bases; trains dense core M)
RaSTA-DM:   ΔW = γ B_L M B_Rᵀ                       (frozen shared random bases; trains dense core M; k² params)
RaSTA-HS:   ΔW = γ B_L diag(s_L) H_V diag(s_R) B_Rᵀ (fixed Walsh–Hadamard core; trains s_L, s_R; 2V params)
```

Two design choices tie the family together:

- **Size-aware budget `ρ`** — each `m×n` weight gets a budget of `≈ √(mn)/ρ` trainable
  parameters (`k² ≈ 2V ≈ √(mn)/ρ`), so larger layers get more parameters and one knob
  controls the whole model. Larger `ρ` = smaller adapter.
- **Update scaling `γ = √ρ`** — keeps the effective optimization step comparable across
  budgets, so a single learning rate per method works at every `ρ` (analogous to
  rank-stabilized LoRA, transposed from rank to budget).

Because the bases are generated deterministically from seeds, adapter checkpoints store only
the tiny core parameters; bases are regenerated (and verified by hash) at merge time. Merged
models are plain `nn.Linear` — **no inference-time overhead**.

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
implementation — identical results, slower. The implementation self-checks the kernel against
the reference at first use.

## Quick start

Each task has a single self-contained driver:
`bash run_<task>.sh <method> <model> <config> [lr] [seed]`

```bash
# RaSTA-HS at rho=1 on Mistral-7B, math task (train on MetaMathQA, eval GSM8K + MATH)
bash run_math.sh rasta-hs mistralai/Mistral-7B-v0.1 rho1

# RaSTA-DM at rho=2 on Llama-3-8B, code task (train on CodeFeedback, eval HumanEval/MBPP)
bash run_code.sh rasta-dm meta-llama/Meta-Llama-3-8B rho2

# LoRA r=8 baseline on Llama-2-7B, math task
bash run_math.sh lora meta-llama/Llama-2-7b-hf r8

# Commonsense (Llama-3-8B; requires local data, see below)
bash run_commonsense.sh rasta-hs meta-llama/Meta-Llama-3-8B rho1

# GLUE (RoBERTa-large): bash run_glue.sh <task> <method> <config>
bash run_glue.sh rte rasta-hs rho1
```

- `method` ∈ `rasta-hs | rasta-dm | lora-xs | lora | vera`
- `config` is `rho<value>` for rasta-hs / rasta-dm / lora-xs, and `r<value>` for lora / vera
- `lr` defaults to the per-method value used in the paper (see table below); `seed` defaults to 42
- Results land in `results/<run_name>/eval_results.json`

Hardware knobs are env vars: `N_GPUS` (default 2, via `torchrun`), `GLOBAL_BATCH` (default
128), `PER_DEVICE_BATCH` (default 8; use 4 for RaSTA-HS on 7–8B models). Gradient
accumulation is derived automatically. The paper's runs fit on a single H100-80GB.

## Reproducing the paper

**Grid.** All main results are single runs at one fixed seed (42). The instruction-tuning
grid is: models `{Llama-2-7B, Mistral-7B, Llama-3-8B}` × methods × configs below
(commonsense uses Llama-3-8B only; GLUE uses RoBERTa-large):

| Method | Instruction tuning | GLUE |
|---|---|---|
| RaSTA-HS / RaSTA-DM / LoRA-XS | ρ ∈ {1, 2, 4} | ρ ∈ {1, 2, 4} |
| LoRA | r ∈ {1, 8, 64} | r ∈ {1, 2, 8} |
| VeRA | r = 1024 | r = 256 |

**Learning rates** (fixed per method; the scripts' defaults):

| Method | Math / Code | Commonsense / GLUE |
|---|---|---|
| RaSTA-HS | 3e-2 | 2e-2 |
| RaSTA-DM | 4e-3 | 2e-3 |
| LoRA-XS | 4e-3 | 2e-3 |
| LoRA | 2e-4 | 2e-4 |
| VeRA | 2e-2 | 2e-2 |

**Protocol.** Instruction tuning: adapters on all attention + MLP linear projections; 1 epoch,
effective batch 128, AdamW, cosine schedule with 0.03 warmup, bf16 base weights with fp32
trainable parameters, Alpaca prompt format, greedy decoding at evaluation. GLUE: adapters on
query/value projections of RoBERTa-large; fp32, sequence length 128, batch 32, linear schedule
with 0.06 warmup; the classification head trains in every run at a fixed lr of 5e-4;
`train_glue.py` evaluates the dev set each epoch and reports the best-epoch metric
(per-task epoch counts are enforced by its defaults). RaSTA and LoRA-XS use `γ = √ρ`
throughout.

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
| Commonsense | Commonsense-170K | BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA | **local files required**, see below |
| GLUE | per-task GLUE train sets | per-task dev sets | auto-downloaded via `datasets` |

For commonsense, download `commonsense_170k.json` and the eight datasets' `test.json` files
from the [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) repository into:

```
Commonsense/data/commonsense/
├── commonsense_170k.json
├── boolq/test.json
├── piqa/test.json
├── social_i_qa/test.json
├── hellaswag/test.json
├── winogrande/test.json
├── ARC-Challenge/test.json
├── ARC-Easy/test.json
└── openbookqa/test.json
```

(or point `DATA_DIR` at an equivalent directory).

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
# variant="dm" for the dense-mixing variant; update scale γ=√ρ is applied automatically.

# ... train the (few) parameters with requires_grad=True ...

# Save only the trained core parameters (+ config and basis hashes) — bases are regenerated.
save_rasta_adapter(model, "my_adapter/", target_modules=target, rho=1.0, variant="hs")

# Later: rebuild the adapted model from a fresh base model, verify bases, merge to plain Linear.
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                            torch_dtype=torch.bfloat16)
merged = load_and_merge_rasta(base, "my_adapter/")   # no inference-time overhead
```

Notes:

- `wht_mode="fused"` (default) uses the CUDA kernel when available and silently falls back to
  the PyTorch reference otherwise.
- Bases are cached and shared across layers of compatible shape; adapter checkpoints contain
  only the core parameters, so they are tiny (e.g. ~1M scalars at ρ=1 on a 7B model).

## Efficiency

Both variants train with substantially lower peak memory than LoRA and VeRA at matched
settings — at sequence length 2048 on an 80 GiB device, RaSTA-DM and LoRA-XS (and larger-ρ
RaSTA-HS) fit where full fine-tuning, every LoRA rank, and VeRA run out of memory. See the
paper's efficiency section and appendix for the full time/memory grid and the fused-vs-reference
Walsh–Hadamard comparison.

