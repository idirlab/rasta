"""
Code/score_code.py — scores a HumanEval/MBPP completions file against
EvalPlus's ground truth WITHOUT requiring every problem in the
benchmark to be present.  Port of the user's subset scorer with two
changes marked NEW below.

Why this exists: `evalplus.evaluate` (the CLI) hard-asserts
`len(completion_id) == len(problems)` — it refuses to run unless your
samples file covers every single problem.  Fine for a full run, breaks
immediately for a smoke on e.g. 4 samples.  This script calls
EvalPlus's own correctness-checking internals (the same
`untrusted_check` / ground-truth cache the CLI uses under the hood)
directly, one task at a time, and reports pass@1 over whatever subset
you actually generated.  Full runs are just the subset that happens to
cover everything, so this is the ONE scoring path for smokes and reals
alike (it also replaces the old pad-then-restrict machinery:
prep_evalplus_samples.py + distill_code_scores.py are retired).

Completion-vs-solution ambiguity (unchanged): an instruction-tuned
model asked to "write a Python script" often returns a complete
function (signature included), not a bare continuation of the
benchmark's original prompt.  Concatenating the benchmark prompt in
front of a complete function would produce a syntactically broken
duplicate `def` line.  So each sample is checked for
`def <entry_point>(` — present: scored standalone; absent: the
benchmark prompt is prepended (EvalPlus's own default for
continuations).

NEW vs the uploaded version:
  1. Emits BOTH pass@1 variants: `pass@1_base` (base tests only) and
     `pass@1` (base AND plus tests — the strict EvalPlus "plus" number;
     key name kept for backward compatibility with anything reading
     the old files).  The base/plus statuses were already computed
     separately; only the strict one was reported before.  The paper
     reports both (HumanEval/HumanEval+ etc.).
  2. evalplus imports moved inside score(), so this module is
     importable (sanity checks, preflights) without evalplus.
"""
import argparse
import json
import os


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_solution(problem: dict, row: dict) -> str:
    if "solution" in row:
        return row["solution"]  # code_process.py: complete function
    entry_point = problem["entry_point"]
    completion = row["completion"]
    if f"def {entry_point}(" in completion:
        return completion  # complete function — don't double-prepend
    return problem["prompt"] + completion  # bare continuation


def score(dataset: str, samples_path: str):
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    from evalplus.eval import PASS, untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    if dataset == "humaneval":
        problems = get_human_eval_plus()
        ground_truth = get_groundtruth(problems,
                                       get_human_eval_plus_hash(), [])
    else:
        problems = get_mbpp_plus()
        ground_truth = get_groundtruth(problems, get_mbpp_plus_hash(),
                                       MBPP_OUTPUT_NOT_NONE_TASKS)

    rows = load_jsonl(samples_path)
    n_pass_base, n_pass, n_total, failed_ids = 0, 0, 0, []

    for row in rows:
        task_id = row["task_id"]
        if task_id not in problems:
            print(f"  skipping {task_id}: not found in {dataset}")
            continue
        problem = problems[task_id]
        solution = build_solution(problem, row)

        base_status, _ = untrusted_check(
            dataset, solution, problem["base_input"],
            problem["entry_point"],
            expected=ground_truth[task_id]["base"], atol=problem["atol"],
            ref_time=ground_truth[task_id]["base_time"], fast_check=True)
        plus_status, _ = untrusted_check(
            dataset, solution, problem["plus_input"],
            problem["entry_point"],
            expected=ground_truth[task_id]["plus"], atol=problem["atol"],
            ref_time=ground_truth[task_id]["plus_time"], fast_check=True)

        base_ok = base_status == PASS
        ok = base_ok and plus_status == PASS
        n_pass_base += int(base_ok)
        n_pass += int(ok)
        n_total += 1
        if not ok:
            failed_ids.append(task_id)

    base_rate = n_pass_base / n_total if n_total else 0.0
    pass_rate = n_pass / n_total if n_total else 0.0
    n_benchmark_total = len(problems)
    print(f"\n{dataset}: base {n_pass_base}/{n_total} = {base_rate:.1%}   "
          f"plus-strict {n_pass}/{n_total} = {pass_rate:.1%}")
    print(f"({n_benchmark_total - n_total}/{n_benchmark_total} benchmark "
          f"problems not attempted — expected for a partial/smoke run)")
    if failed_ids:
        print(f"failed task_ids (plus-strict): {failed_ids}")

    out_dir = os.path.dirname(os.path.abspath(samples_path))
    out_path = os.path.join(out_dir, f"{dataset}_pass1.json")
    payload = {
        "dataset":           dataset,
        "pass@1":            pass_rate,     # strict: base AND plus (EvalPlus "+")
        "pass@1_base":       base_rate,     # NEW: base tests only
        "n_pass":            n_pass,
        "n_pass_base":       n_pass_base,
        "n_attempted":       n_total,
        "n_benchmark_total": n_benchmark_total,
        "partial_run":       n_total < n_benchmark_total,
        "failed_task_ids":   failed_ids,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["humaneval", "mbpp"],
                   required=True)
    p.add_argument("--samples", required=True,
                   help="<run_dir>/humaneval.jsonl or mbpp.jsonl")
    args = p.parse_args()
    score(args.dataset, args.samples)


if __name__ == "__main__":
    main()
