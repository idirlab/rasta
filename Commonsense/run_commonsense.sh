#!/usr/bin/env bash
# run_commonsense.sh — train and evaluate one method/model/config on
# the commonsense-170k task.
#
# Usage:
#   bash run_commonsense.sh <method> <model> <config> [lr] [seed]
#
#   method: rasta-hs | rasta-dm | lora-xs | lora | vera
#   config: rho<value> for rasta-hs/rasta-dm/lora-xs; r<value> for lora/vera
#
# Trains on the full Commonsense-170K split (1 epoch), evaluates on the
# 8 commonsense datasets (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande,
# ARC-Challenge, ARC-Easy, OpenBookQA).
#
# Expects <repo_root>/Commonsense/data/commonsense/commonsense_170k.json
# and .../commonsense/<dataset>/test.json for each dataset above.

set -euo pipefail

METHOD="$1"; MODEL="$2"; CONFIG="$3"
SEED="${5:-42}"

declare -A DEFAULT_LR=(
    [rasta-hs]="2e-2" [rasta-dm]="2e-3"
    [lora-xs]="2e-3"  [lora]="2e-4" [vera]="2e-2"
)
LR="${4:-${DEFAULT_LR[$METHOD]:-}}"
if [ -z "$LR" ]; then
    echo "unknown method: $METHOD (expected one of: ${!DEFAULT_LR[*]})" >&2
    exit 1
fi

N_GPUS="${N_GPUS:-2}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-8}"     # use 4 for rasta-hs on 7-8B models
GRAD_ACCUM=$(( GLOBAL_BATCH / (PER_DEVICE_BATCH * N_GPUS) ))
MAX_SEQ_LEN=256
LR_SCHEDULER=cosine
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
EVAL_BATCH=16
DATA_DIR="${DATA_DIR:-Commonsense/data/commonsense}"
DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RUN_DIR="results/${METHOD}_$(basename "$MODEL")_${CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# ── Build training config ─────────────────────────────────────────────────
CFG_FILE="$(mktemp /tmp/run_cs_XXXXXX.json)"
python3 - "$CFG_FILE" << PY
import json
cfg = {
    "model": "$MODEL", "output_dir": "$RUN_DIR",
    "learning_rate": $LR, "seeds": [$SEED], "num_train_epochs": 1,
    "lr_scheduler_type": "$LR_SCHEDULER", "warmup_ratio": $WARMUP_RATIO,
    "weight_decay": $WEIGHT_DECAY, "max_train_samples": None,
    "max_seq_len": $MAX_SEQ_LEN,
    "per_device_train_batch_size": $PER_DEVICE_BATCH,
    "gradient_accumulation_steps": $GRAD_ACCUM,
    "data_module": "Commonsense/data_commonsense.py", "trainer_bf16": True,
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

declare -A TRAIN_MODULE=(
    [rasta-hs]=utils.train_rasta [rasta-dm]=utils.train_rasta
    [lora-xs]=utils.train_loraxs [lora]=utils.train_lora [vera]=utils.train_vera
)
echo "Training $METHOD on $MODEL ($CONFIG, lr=$LR, seed=$SEED)..."
torchrun --nproc_per_node="$N_GPUS" -m "${TRAIN_MODULE[$METHOD]}" --config "$CFG_FILE"
rm -f "$CFG_FILE"

# ── Evaluate on all 8 datasets ──────────────────────────────────────────────
EVAL_OUT="$RUN_DIR/commonsense_eval"; mkdir -p "$EVAL_OUT"
for ds in "${DATASETS[@]}"; do
    echo "Evaluating $ds..."
    python3 Commonsense/eval_commonsense.py \
        --model_path "$RUN_DIR/merged" --dataset "$ds" \
        --batch_size "$EVAL_BATCH" --output_dir "$EVAL_OUT" --data_dir "$DATA_DIR"
done

python3 - "$RUN_DIR" "$EVAL_OUT" << 'PY'
import json, sys, os
rd, out = sys.argv[1], sys.argv[2]
ds_all = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande",
          "ARC-Challenge", "ARC-Easy", "openbookqa"]
results = {}
for ds in ds_all:
    p = os.path.join(out, f"{ds}_summary.json")
    if os.path.exists(p):
        results[ds] = {"accuracy": json.load(open(p))["accuracy"]}
if len(results) == len(ds_all):
    results["average"] = {"accuracy": sum(v["accuracy"] for v in results.values()) / len(ds_all)}
json.dump({"results": results}, open(os.path.join(rd, "eval_results.json"), "w"), indent=2)
print("[eval]", {k: round(v["accuracy"], 4) for k, v in results.items()})
PY

echo "Done. Results in $RUN_DIR/eval_results.json"
