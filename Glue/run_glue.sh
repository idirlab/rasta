#!/usr/bin/env bash
# run_glue.sh — train and evaluate one method/config on one GLUE task.
#
# Usage:
#   bash run_glue.sh <task> <method> <config> [lr] [seed]
#
#   task:   rte | mrpc | stsb | cola | sst2 | qnli | qqp | mnli
#   method: rasta-hs | rasta-dm | lora-xs | lora | vera
#   config: rho<value> for rasta-hs/rasta-dm/lora-xs; r<value> for lora/vera
#
# Model: roberta-large. Adapter lr is set below; the classification
# head trains at a fixed lr of 5e-4 (train_glue.py default, not swept).
# train_glue.py handles per-epoch dev eval and writes eval_results.json.

set -euo pipefail

TASK="$1"; METHOD="$2"; CONFIG="$3"
SEED="${5:-42}"
MODEL="${MODEL:-roberta-large}"

declare -A DEFAULT_LR=(
    [rasta-hs]="2e-2" [rasta-dm]="2e-3"
    [lora-xs]="2e-3"  [lora]="2e-4" [vera]="2e-2"
)
LR="${4:-${DEFAULT_LR[$METHOD]:-}}"
if [ -z "$LR" ]; then
    echo "unknown method: $METHOD (expected one of: ${!DEFAULT_LR[*]})" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RUN_DIR="results/glue_${TASK}_${METHOD}_${CONFIG}_seed${SEED}"
mkdir -p "$RUN_DIR"

CFG_FILE="$(mktemp /tmp/run_glue_XXXXXX.json)"
python3 - "$CFG_FILE" << PY
import json
cfg = {
    "task": "$TASK", "method": "$METHOD", "model": "$MODEL",
    "learning_rate": $LR, "seeds": [$SEED], "output_dir": "$RUN_DIR",
}
tok = "$CONFIG"
if tok.startswith("rho"):
    cfg["rho"] = float(tok[3:])
elif tok.startswith("r"):
    cfg["rank"] = int(tok[1:])
else:
    raise SystemExit(f"bad config token {tok!r}")
json.dump(cfg, open("$CFG_FILE", "w"), indent=2)
PY

echo "Training $METHOD on GLUE/$TASK ($CONFIG, lr=$LR, seed=$SEED)..."
python3 Glue/train_glue.py --config "$CFG_FILE"
rm -f "$CFG_FILE" "$RUN_DIR"/_cfg_*.json 2>/dev/null || true
rm -rf "$RUN_DIR/hf_tmp"

echo "Done. Results in $RUN_DIR/eval_results.json"
python3 -c '
import json, sys
r = json.load(open(sys.argv[1] + "/eval_results.json"))["results"][sys.argv[2]]
print("dev accuracy:", r["accuracy"])' "$RUN_DIR" "$TASK"
