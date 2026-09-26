#!/usr/bin/env python3
"""
Commonsense/eval_commonsense.py — evaluate a MERGED model on the 8
commonsense-reasoning datasets.  Ported unchanged from
commonsense_evaluate.py (method-agnostic: every trainer saves a plain
full checkpoint under <run_dir>/merged, and all methods are scored on
byte-identical prompts).

For each dataset this writes, into --output_dir:
    <dataset>_predictions.json  — per-example predictions
    <dataset>_summary.json      — {dataset, correct, total, accuracy}

cs_common.sh's eval_cs_run drives this per dataset, then consolidates
the summaries into <run_dir>/eval_results.json (math-compatible shape:
results = {<dataset>: {accuracy, correct, total}, ..., average:
{accuracy}}), which is what cleanup_run gates on and
utils/aggregate_results.py reads.

Usage:
    python Commonsense/eval_commonsense.py \
        --model_path <run_dir>/merged \
        --dataset boolq \
        --batch_size 16 \
        --output_dir <run_dir>/commonsense_eval \
        [--max_samples 5]         # sanity: first N test examples only
"""
import copy
import json
import os
import re
import argparse

import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = ["boolq", "piqa", "social_i_qa", "hellaswag",
            "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to the MERGED model dir "
                             "(<run_dir>/merged), saved via save_pretrained.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_dir", default="commonsense_eval",
                        help="Where to write <dataset>_predictions.json and "
                             "<dataset>_summary.json for this run.")
    parser.add_argument("--data_dir", default="Commonsense/data/commonsense",
                        help="Root holding <dataset>/test.json for each dataset.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--no_bf16", action="store_true", default=False,
                        help="use fp16 instead of bf16 (bf16 default, "
                             "matching training).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, only evaluate the first N examples "
                             "(quick sanity check).")
    return parser.parse_args()


# ── Prompt: copied VERBATIM from commonsense_evaluate.py ──────────────────────
# (byte-identical to the response-prefix produced by the training side's
#  generate_prompt; S18 in utils/sanity_check.py pins both)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    file_path = os.path.join(args.data_dir, args.dataset, "test.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, "r"))
    if args.max_samples is not None:
        json_data = json_data[:args.max_samples]
    return json_data


def create_batch(dataset, batch_size):
    batches = []
    num_batch = (len(dataset) // batch_size
                 if len(dataset) % batch_size == 0
                 else len(dataset) // batch_size + 1)
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def load_model(args):
    """Load the merged model directly — no PEFT wrapper. The tokenizer is
    read from the same dir (saved at merge time) so it matches training; we still
    force pad_token_id=0 / left padding to be identical to the original
    commonsense_evaluate.py generation setup."""
    use_bf16 = ((not args.no_bf16) and torch.cuda.is_available()
                and torch.cuda.is_bf16_supported())
    if (not args.no_bf16) and not use_bf16:
        print("bf16 requested but not supported on this GPU/torch build "
              "-- falling back to fp16.")
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0  # unk, deliberately != eos (matches training)

    def _load(attn_impl):
        kwargs = dict(trust_remote_code=True)
        if device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = compute_dtype
        return AutoModelForCausalLM.from_pretrained(
            args.model_path, attn_implementation=attn_impl, **kwargs)

    try:
        model = _load(args.attn_implementation)
    except Exception as e:
        print(f"Failed to load with attn_implementation="
              f"'{args.attn_implementation}' ({e}); falling back to 'eager'.")
        model = _load("eager")

    if device == "cpu":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def extract_answer(dataset: str, sentence: str) -> str:
    sentence_ = sentence.strip().lower()  # lower-cased for robust matching
    if dataset == "boolq":
        pred = re.findall(r"true|false", sentence_)
    elif dataset == "piqa":
        pred = re.findall(r"solution1|solution2", sentence_)
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
    elif dataset == "hellaswag":
        pred = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
    elif dataset == "winogrande":
        pred = re.findall(r"option1|option2", sentence_)
    else:
        pred = []
    return pred[0] if pred else ""


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model = load_model(args)

    def evaluate(instructions, input=None, temperature=0.1, top_p=0.75,
                 top_k=40, num_beams=4, max_new_tokens=32, **kwargs):
        prompts = [generate_prompt(ins, input) for ins in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature, top_p=top_p, top_k=top_k,
            num_beams=num_beams, **kwargs)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)

    total_batches = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total_batches)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]
        outputs = evaluate(instructions, max_new_tokens=args.max_new_tokens)
        for data, output in zip(batch, outputs):
            label = data.get("answer")
            predict = extract_answer(args.dataset, output)
            flag = (label == predict)
            if flag:
                correct += 1
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)
        acc = correct / current if current else 0.0
        print(f"\r{args.dataset}: {idx + 1}/{total_batches} | "
              f"accuracy {correct}/{current} = {acc:.4f}")
        pred_path = os.path.join(args.output_dir,
                                 f"{args.dataset}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()

    total = len(output_data)
    accuracy = correct / total if total else 0.0
    summary = {
        "dataset":  args.dataset,
        "correct":  correct,
        "total":    total,
        "accuracy": accuracy,
    }
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{args.dataset}] accuracy = {accuracy:.4f} "
          f"({correct}/{total})  →  {summary_path}")


if __name__ == "__main__":
    main()
