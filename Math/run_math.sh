#!/usr/bin/env bash
# run_math.sh — train and evaluate one method/model/config on the math task.
#
# Usage:
#   bash run_math.sh <method> <model> <config> [lr] [seed]
#
#   method: rasta-hs | rasta-dm | lora-xs | lora | vera
#   model:  any HF model id (e.g. mistralai/Mistral-7B-v0.1)
#   config: rho<value> for rasta-hs/rasta-dm/lora-xs (e.g. rho1)
#           r<value>   for lora/vera                (e.g. r8)
#   lr:     optional, defaults to the value used in the paper
#   seed:   optional, default 42
#
# Example:
#   bash run_math.sh rasta-hs mistralai/Mistral-7B-v0.1 rho1
#
# Requires: <repo_root>/utils/{train_rasta,train_loraxs,train_lora,train_vera}.py
#           <repo_root>/Math/{data_math.py,eval_math.py}
# Trains on MetaMathQA, evaluates on MATH + GSM8K.

set -euo pipefail

METHOD="$1"; MODEL="$2"; CONFIG="$3"
SEED="${5:-42}"

# Default learning rates from the paper (one universal value per method).
declare -A DEFAULT_LR=(
    [rasta-hs]="3e-2" [rasta-dm]="4e-3"
    [lora-xs]="4e-3"  [lora]="2e-4" [vera]="2e-2"
)
LR="${4:-${DEFAULT_LR[$METHOD]:-}}"
if [ -z "$LR" ]; then
    echo "unknown method: $METHOD" >&2
    echo "expected one of: ${!DEFAULT_LR[*]}" >&2
    exit 1
fi

# ── Fixed training config (kept identical across models/methods for comparability) ──
N_GPUS="${N_GPUS:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"     # use 4 for rasta-hs on 7-8B models
GRAD_ACCUM=$(( GLOBAL_BATCH / (PER_DEVICE_BATCH * N_GPUS) ))
N_TRAIN_SAMPLES="${N_TRAIN_SAMPLES:-100000}"
N_EVAL_SAMPLES="${N_EVAL_SAMPLES:-999999}"     # 999999 = full eval set
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LEN=512
LR_SCHEDULER=cosine
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RUN_DIR="results/${METHOD}_$(basename "$MODEL")_${CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# ── Build training config ─────────────────────────────────────────────────
CFG_FILE="$(mktemp /tmp/run_math_XXXXXX.json)"
python3 - "$CFG_FILE" << PY
import json
cfg = {
    "model": "$MODEL", "output_dir": "$RUN_DIR",
    "learning_rate": $LR, "seeds": [$SEED],
    "lr_scheduler_type": "$LR_SCHEDULER", "warmup_ratio": $WARMUP_RATIO,
    "weight_decay": $WEIGHT_DECAY, "num_train_epochs": $EPOCHS,
    "max_train_samples": $N_TRAIN_SAMPLES, "max_seq_len": $MAX_SEQ_LEN,
    "per_device_train_batch_size": $PER_DEVICE_BATCH,
    "gradient_accumulation_steps": $GRAD_ACCUM,
    "data_module": "Math/data_math.py", "trainer_bf16": True,
}
case = "$METHOD"
if case in ("rasta-hs", "rasta-dm"):
    cfg["rho"] = float("$CONFIG".removeprefix("rho"))
    cfg["variant"] = "hs" if case == "rasta-hs" else "dm"
    cfg["basis_construction"] = "gaussian"
elif case == "lora-xs":
    cfg["rho"] = float("$CONFIG".removeprefix("rho"))
elif case in ("lora", "vera"):
    cfg["rank"] = int("$CONFIG".removeprefix("r"))
json.dump(cfg, open("$CFG_FILE", "w"))
PY

# ── Train ──────────────────────────────────────────────────────────────────
declare -A TRAIN_MODULE=(
    [rasta-hs]=utils.train_rasta [rasta-dm]=utils.train_rasta
    [lora-xs]=utils.train_loraxs [lora]=utils.train_lora [vera]=utils.train_vera
)
echo "Training $METHOD on $MODEL ($CONFIG, lr=$LR, seed=$SEED)..."
torchrun --nproc_per_node="$N_GPUS" -m "${TRAIN_MODULE[$METHOD]}" --config "$CFG_FILE"
rm -f "$CFG_FILE"

# ── Evaluate ───────────────────────────────────────────────────────────────
echo "Evaluating on MATH + GSM8K..."
python3 Math/eval_math.py \
    --checkpoint "$RUN_DIR/merged" --task both \
    --batch_size 32 --num_samples "$N_EVAL_SAMPLES" \
    --max_new_tokens_gsm8k 512 --max_new_tokens_math 2048 \
    --output_file "$RUN_DIR/eval_results.json"

echo "Done. Results in $RUN_DIR/eval_results.json"
