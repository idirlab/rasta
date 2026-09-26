"""
Code/code_process.py — extracts clean Python code from generated
completions and splits them into humaneval.jsonl / mbpp.jsonl, ready
for scoring.  Verbatim port; the one change vs the uploaded version is
that evalplus is imported lazily inside main(), so the module stays
importable (for sanity checks / preflights) on machines without
evalplus installed.

Original design notes preserved:
1. human_eval's write_jsonl/stream_jsonl inlined (drops a
   security-sensitive dependency used for two five-line functions).
2. Writes a `solution` field instead of `completion` whenever the
   cleaned code already contains `def <entry_point>(` — i.e. the model
   returned a complete, self-contained function rather than a bare
   continuation.  Blindly prepending the benchmark prompt in front of
   an already-complete function produces a syntactically broken
   duplicate `def` line and silently tanks pass@1.
"""
import argparse
import json
import os


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def clean_completion(text: str) -> str:
    text = text.replace("\r", "")
    if "```python" in text:
        start = text.index("```python")
        text = text[start:].replace("```python", "", 1).strip()
        if "\n```" in text:
            text = text[: text.index("\n```")].strip()
    for marker in ('if __name__ == "__main__":', "# Example usage",
                   "assert"):
        if marker in text:
            text = text[: text.index(marker)].strip()
    return text


def main():
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="responses.jsonl produced by generate.py")
    args = parser.parse_args()

    humaneval_problems = get_human_eval_plus()
    mbpp_problems = get_mbpp_plus()
    humaneval, mbpp = [], []
    for row in read_jsonl(args.path):
        if row["type"] not in ("humaneval", "mbpp"):
            continue
        row["task_id"] = str(row["answer"])
        cleaned = clean_completion(row["output"])
        problems = (humaneval_problems if row["type"] == "humaneval"
                    else mbpp_problems)
        entry_point = problems.get(row["task_id"], {}).get("entry_point", "")
        if entry_point and f"def {entry_point}(" in cleaned:
            row["solution"] = cleaned      # complete, self-contained fn
        else:
            row["completion"] = cleaned    # bare continuation
        (humaneval if row["type"] == "humaneval" else mbpp).append(row)

    out_dir = os.path.dirname(args.path) or "."
    humaneval_path = os.path.join(out_dir, "humaneval.jsonl")
    mbpp_path = os.path.join(out_dir, "mbpp.jsonl")
    write_jsonl(humaneval_path, humaneval)
    write_jsonl(mbpp_path, mbpp)
    print(f"{len(humaneval)} HumanEval rows -> {humaneval_path}")
    print(f"{len(mbpp)} MBPP rows -> {mbpp_path}")


if __name__ == "__main__":
    main()
