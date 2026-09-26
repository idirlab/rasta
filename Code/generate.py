"""
Code/generate.py — batched generation for the code eval (replaces
vLLM; verbatim port with three additions marked NEW below).

Why no vLLM: it needs a specific CUDA/torch build, compiles custom
kernels on install, and regularly breaks across transformers versions.
For eval runs (not high-throughput serving), plain model.generate() in
a batched loop is slower but has none of those failure modes.

Left padding is used here (not training's right padding) because
batched causal-LM generation requires it: with right padding the model
would see padding tokens positioned *before* the real continuation
point for shorter sequences in the batch, corrupting position handling.

NEW vs the original:
  * --types: filter test rows by their "type" column (humaneval / mbpp)
    so the LR-selection probes can generate ONLY the probe benchmark's
    rows; winners generate the held-out type separately.
  * peft imported lazily (only when --adapter_path is given), so the
    script runs in environments without peft — our pipeline always
    evaluates <run_dir>/merged with no adapter.
  * prompt_utils resolved relative to this file, so the script works
    from any CWD.
"""
import argparse
import importlib.util
import json
import os

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "code_prompt_utils", os.path.join(_here, "prompt_utils.py"))
_pu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pu)
format_prompt = _pu.format_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None,
                   help="Saved LoRA adapter dir. Omit to evaluate the "
                        "model at --base_model as-is (our pipeline: "
                        "<run_dir>/merged, no adapter).")
    p.add_argument("--data_path", default="fxmeng/pissa-dataset")
    p.add_argument("--sub_task", nargs="+", default=["python"])
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--prompt_template", default="none",
                   help="Must match the training-side value")
    p.add_argument("--types", nargs="+", default=None,
                   help="NEW: only generate rows whose 'type' is in this "
                        "list (e.g. --types humaneval). Default: all.")
    p.add_argument("--output_file", default="responses.jsonl")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_samples_per_type", type=int, default=None,
                   help="Cap rows per 'type' (quick sanity runs).")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    parts = [load_dataset(args.data_path, data_dir=task,
                          split=args.dataset_split)
             for task in args.sub_task]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    queries = [format_prompt(x, args.prompt_template)
               for x in dataset["instruction"]]
    answers = dataset["output"]
    types = dataset["type"]

    if args.types is not None:
        allowed = set(args.types)
        keep = [i for i, t in enumerate(types) if t in allowed]
        queries = [queries[i] for i in keep]
        answers = [answers[i] for i in keep]
        types = [types[i] for i in keep]
        print(f"Type filter {sorted(allowed)} -> {len(keep)} rows")

    if args.max_samples_per_type is not None:
        seen, keep = {}, []
        for i, t in enumerate(types):
            if seen.get(t, 0) < args.max_samples_per_type:
                keep.append(i)
                seen[t] = seen.get(t, 0) + 1
        queries = [queries[i] for i in keep]
        answers = [answers[i] for i in keep]
        types = [types[i] for i in keep]
        print(f"Capped to {args.max_samples_per_type}/type -> {dict(seen)} "
              f"({len(keep)} rows total)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    with open(args.output_file, "w") as f:
        for i in tqdm(range(0, len(queries), args.batch_size),
                      desc="Generating"):
            batch_q = queries[i:i + args.batch_size]
            batch_a = answers[i:i + args.batch_size]
            batch_t = types[i:i + args.batch_size]

            inputs = tokenizer(batch_q, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024
                               ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False,   # greedy == vLLM temperature=0
                    pad_token_id=tokenizer.pad_token_id)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new_tokens,
                                           skip_special_tokens=True)
            for q, text, a, t in zip(batch_q, texts, batch_a, batch_t):
                f.write(json.dumps({"type": t, "query": q, "output": text,
                                    "answer": a}) + "\n")

    print(f"Wrote {len(queries)} generations to {args.output_file}")


if __name__ == "__main__":
    main()
