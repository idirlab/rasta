#!/usr/bin/env bash
# run_code.sh — train and evaluate one method/model/config on the code task.
#
# Usage:
#   bash run_code.sh <method> <model> <config> [lr] [seed]
#
#   method: rasta-hs | rasta-dm | lora-xs | lora | vera
#   model:  any HF model id
#   config: rho<value> for rasta-hs/rasta-dm/lora-xs (e.g. rho1)
#           r<value>   for lora/vera                (e.g. r8)
#
# Trains on the pissa-dataset python split (1 epoch, full split),
# evaluates HumanEval(+) and MBPP(+) via EvalPlus.
#
# Requires: evalplus (pip install evalplus)
#           <repo_root>/utils/train_{rasta,loraxs,lora,vera}.py
#           <repo_root>/Code/{data_code.py,generate.py,code_process.py,score_code.py}

set -euo pipefail

METHOD="$1"; MODEL="$2"; CONFIG="$3"
SEED="${5:-42}"

declare -A DEFAULT_LR=(
    [rasta-hs]="3e-2" [rasta-dm]="4e-3"
    [lora-xs]="4e-3"  [lora]="2e-4" [vera]="2e-2"
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
MAX_SEQ_LEN=512
LR_SCHEDULER=cosine
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_NEW_TOKENS=1024
EVAL_BATCH=16

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RUN_DIR="results/${METHOD}_$(basename "$MODEL")_${CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

# ── Build training config ─────────────────────────────────────────────────
CFG_FILE="$(mktemp /tmp/run_code_XXXXXX.json)"
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
    "data_module": "Code/data_code.py", "trainer_bf16": True,
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

# ── Evaluate: generate -> clean/split -> score -> consolidate ──────────────
echo "Generating completions..."
python3 Code/generate.py --base_model "$RUN_DIR/merged" \
    --types humaneval mbpp --output_file "$RUN_DIR/responses.jsonl" \
    --batch_size "$EVAL_BATCH" --max_new_tokens "$MAX_NEW_TOKENS"

python3 Code/code_process.py --path "$RUN_DIR/responses.jsonl"

for ds in humaneval mbpp; do
    echo "Scoring $ds..."
    python3 Code/score_code.py --dataset "$ds" --samples "$RUN_DIR/$ds.jsonl"
done

python3 - "$RUN_DIR" << 'PY'
import json, sys, os
rd = sys.argv[1]
results = {}
for ds in ("humaneval", "mbpp"):
    p = os.path.join(rd, f"{ds}_pass1.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    results[ds] = {"accuracy": d["pass@1_base"]}
    results[f"{ds}_plus"] = {"accuracy": d["pass@1"]}
plus = ("humaneval_plus", "mbpp_plus")
if all(k in results for k in plus):
    results["average"] = {"accuracy": sum(results[k]["accuracy"] for k in plus) / 2}
json.dump({"results": results}, open(os.path.join(rd, "eval_results.json"), "w"), indent=2)
print("[eval]", {k: round(v["accuracy"], 4) for k, v in results.items()})
PY

echo "Done. Results in $RUN_DIR/eval_results.json"
