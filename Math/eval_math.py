"""
eval_math.py — Evaluate a merged model on GSM8K and MATH-500.

Usage:
    python eval_math.py --checkpoint ./path/to/merged \
                        --task both \
                        --batch_size 8 \
                        --output_file ./path/to/eval_results.json
"""

from __future__ import annotations

import argparse, json, logging, math, os, re, sys
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import sympy as sp
    from sympy import Eq, FiniteSet, Interval
    from sympy.parsing.sympy_parser import (
        convert_xor, implicit_multiplication_application,
        parse_expr, standard_transformations,
    )
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("eval_math")

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)

_SYMPY_TRANSFORMS = ()
if _HAS_SYMPY:
    _SYMPY_TRANSFORMS = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--task",                default="both", choices=["gsm8k", "math", "both"])
    p.add_argument("--batch_size",          type=int, default=8)
    p.add_argument("--max_new_tokens_gsm8k",type=int, default=512)
    p.add_argument("--max_new_tokens_math", type=int, default=2048)
    p.add_argument("--max_seq_len",         type=int, default=512)
    p.add_argument("--output_file",         type=str, default=None)
    p.add_argument("--num_samples",         type=int, default=None,
                   help="Limit examples per task (for debugging).")
    p.add_argument("--save_generations",    action="store_true")
    return p.parse_args()


# ── extraction ────────────────────────────────────────────────────────────────

def _extract_boxed_content(text: str) -> Optional[str]:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start, depth, out = idx + len(r"\boxed{"), 1, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1; out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = re.search(r"[Tt]he answer is:?\s*([\-]?\d[\d,\.]*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_math_answer(text: str) -> Optional[str]:
    """Used only on model outputs, not reference solutions."""
    boxed = _extract_boxed_content(text)
    if boxed:
        return boxed.strip()
    for pat in [r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*([^\n]+)",
                r"[Aa]nswer\s*:?\s*([^\n]+)"]:
        m = re.search(pat, text)
        if m:
            s = m.group(1).strip().rstrip(".").strip("$").strip()
            if s:
                return s
    lines = [x.strip() for x in text.strip().split("\n") if x.strip()]
    return lines[-1].rstrip(".").strip("$").strip() if lines else None


# ── normalisation & equivalence ───────────────────────────────────────────────

def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        if not math.isfinite(f):
            return s.lower()
        return str(int(f)) if f == int(f) else f"{f:.12g}"
    except Exception:
        return s.lower()


def _strip_outer_delimiters(s: str) -> str:
    s = s.strip()
    for a, b in [("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")]:
        changed = True
        while changed:
            changed = False
            if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
                s = s[len(a):-len(b)].strip(); changed = True
    return s


def _normalize_math_text(s: str) -> str:
    s = _strip_outer_delimiters(s)
    for tok in (",", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:"):
        s = s.replace(tok, "")
    s = (s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
          .replace("−", "-").replace("–", "-").replace("^", "**"))
    return re.sub(r"\s+", "", s)


def _latex_frac_to_plain(s: str) -> str:
    pat = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s; s = pat.sub(r"((\1)/(\2))", s)
    return s


def _latex_sqrt_to_plain(s: str) -> str:
    pat = re.compile(r"\\sqrt\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s; s = pat.sub(r"sqrt(\1)", s)
    return s


def _parse_math_expr(s: str):
    if not _HAS_SYMPY:
        return None
    s = _normalize_math_text(s)
    s = _latex_frac_to_plain(s)
    s = _latex_sqrt_to_plain(s)
    for k, v in {r"\mathbb{R}": "Reals", r"\mathbb{Z}": "Integers",
                 r"\mathbb{Q}": "Rationals", r"\mathbb{N}": "Naturals"}.items():
        s = s.replace(k, v)
    s = (s.replace(r"\cdot", "*").replace(r"\times", "*")
          .replace(r"\pi", "pi").replace(r"\infty", "oo")
          .replace("{", "(").replace("}", ")"))
    local_dict = {
        "pi": sp.pi, "oo": sp.oo, "sqrt": sp.sqrt, "sin": sp.sin,
        "cos": sp.cos, "tan": sp.tan, "log": sp.log, "ln": sp.log,
        "exp": sp.exp, "Abs": sp.Abs, "Reals": sp.S.Reals,
        "Integers": sp.S.Integers, "Rationals": sp.S.Rationals, "Naturals": sp.S.Naturals,
    }
    try:
        return parse_expr(s, transformations=_SYMPY_TRANSFORMS, local_dict=local_dict, evaluate=True)
    except Exception:
        return None


def _interval_like(s: str):
    if not _HAS_SYMPY:
        return None
    s = _normalize_math_text(_latex_frac_to_plain(_latex_sqrt_to_plain(s)))
    m = re.compile(r"^([\(\[])([^,\]]+),([^,\]]+)([\)\]])$").match(s)
    if not m:
        return None
    lb, a, b, rb = m.groups()
    ae, be = _parse_math_expr(a), _parse_math_expr(b)
    if ae is None or be is None:
        return None
    try:
        return Interval(ae, be, left_open=(lb == "("), right_open=(rb == ")"))
    except Exception:
        return None


def _set_like(s: str):
    if not _HAS_SYMPY:
        return None
    s = _normalize_math_text(s)
    if not (s.startswith("{") and s.endswith("}")):
        return None
    inner = s[1:-1]
    if inner == "":
        return FiniteSet()
    exprs = [_parse_math_expr(p) for p in inner.split(",")]
    if any(e is None for e in exprs):
        return None
    try:
        return FiniteSet(*exprs)
    except Exception:
        return None


def gsm8k_answers_match(pred: Optional[str], gold: str) -> bool:
    return pred is not None and normalize_number(pred) == normalize_number(gold)


def math_answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    pred_s = _strip_outer_delimiters(pred.strip())
    gold_s = _strip_outer_delimiters(gold.strip())

    if normalize_number(pred_s) == normalize_number(gold_s):
        return True
    if _normalize_math_text(pred_s) == _normalize_math_text(gold_s):
        return True
    if not _HAS_SYMPY:
        return False

    pi, gi = _interval_like(pred_s), _interval_like(gold_s)
    if pi is not None and gi is not None:
        return pi == gi

    ps, gs = _set_like(pred_s), _set_like(gold_s)
    if ps is not None and gs is not None:
        return ps == gs

    pe, ge = _parse_math_expr(pred_s), _parse_math_expr(gold_s)
    if pe is not None and ge is not None:
        try:
            if sp.simplify(pe - ge) == 0:
                return True
        except Exception:
            pass
        try:
            if sp.simplify(Eq(pe, ge)) is True:
                return True
        except Exception:
            pass
    return False


# ── data loading ──────────────────────────────────────────────────────────────

def load_gsm8k(num_samples: Optional[int] = None) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    rows = []
    for ex in ds:
        m    = re.search(r"####\s*([\-]?\d[\d,\.]*)", ex["answer"])
        gold = m.group(1).replace(",", "").strip() if m else ex["answer"]
        rows.append({"question": ex["question"], "gold": gold, "task": "gsm8k"})
    return rows


def load_math500(num_samples: Optional[int] = None) -> List[Dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))
    rows = []
    for ex in ds:
        gold = _extract_boxed_content(ex["solution"]) or ex["solution"]
        gold = _strip_outer_delimiters(gold).strip()
        rows.append({
            "question": ex["problem"], "gold": gold, "task": "math",
            "level": ex.get("level", ""), "type": ex.get("type", ""),
        })
    return rows


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint: str):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token    = tok.eos_token
    tok.padding_side = "left"
    last_exc = None
    for attn in ("sdpa", None):
        try:
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
            if attn is not None:
                kwargs["attn_implementation"] = attn
            model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
            model.eval()
            return model, tok
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load {checkpoint}: {last_exc}")


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    examples: List[Dict], model, tokenizer,
    batch_size: int, max_new_tokens: int, max_seq_len: int,
    task: str, save_generations: bool = False,
) -> Dict:
    device  = next(model.parameters()).device
    total   = len(examples)
    correct = 0
    records = []

    for start in range(0, total, batch_size):
        batch   = examples[start: start + batch_size]
        prompts = [PROMPT.format(instruction=ex["question"]) for ex in batch]
        enc     = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq_len)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )

        texts = tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        for ex, txt in zip(batch, texts):
            txt = txt if txt.strip() else ""
            if task == "gsm8k":
                pred = extract_gsm8k_answer(txt)
                ok   = gsm8k_answers_match(pred, ex["gold"])
            else:
                pred = extract_math_answer(txt)
                ok   = math_answers_match(pred, ex["gold"])
            correct += int(ok)
            row = {"question": ex["question"], "gold": ex["gold"], "predicted": pred, "correct": ok}
            if "level" in ex: row["level"] = ex["level"]
            if "type"  in ex: row["type"]  = ex["type"]
            if save_generations: row["generated"] = txt
            records.append(row)

        done = min(start + batch_size, total)
        if done % max(batch_size * 10, 50) == 0 or done == total:
            logger.info(f"  [{done}/{total}]  acc={correct/done:.4f}")

    result = {"accuracy": correct / total if total else 0.0,
              "correct": correct, "total": total}
    if save_generations:
        result["per_example"] = records
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    out_path = args.output_file or os.path.join(args.checkpoint, "eval_results.json")
    tasks    = ["math", "gsm8k"] if args.task == "both" else [args.task]
    task_tokens = {"gsm8k": args.max_new_tokens_gsm8k, "math": args.max_new_tokens_math}

    model, tok = load_model_and_tokenizer(args.checkpoint)
    results    = {}

    for task in tasks:
        examples = load_gsm8k(args.num_samples) if task == "gsm8k" else load_math500(args.num_samples)
        logger.info(f"Evaluating {task.upper()} ({len(examples)} examples, "
                    f"max_new_tokens={task_tokens[task]})")
        res = evaluate(examples, model, tok, args.batch_size, task_tokens[task],
                       args.max_seq_len, task, args.save_generations)
        results[task] = res
        logger.info(f"{task.upper()} accuracy: {res['accuracy']:.4f} "
                    f"({res['correct']}/{res['total']})")

    payload = {
        "checkpoint": args.checkpoint,
        "eval_protocol": (
            "0-shot greedy Alpaca prompt + CoT trigger; "
            "GSM8K: 'The answer is: N' + last-number fallback; "
            "MATH-500: \\boxed{} extraction + sympy equivalence; "
            "gold via _extract_boxed_content on reference solutions."
        ),
        "generation_config": {"do_sample": False,
                              "max_new_tokens_gsm8k": task_tokens.get("gsm8k"),
                              "max_new_tokens_math":  task_tokens.get("math")},
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved: {out_path}")
    print(json.dumps({
        "gsm8k_accuracy": results.get("gsm8k", {}).get("accuracy", float("nan")),
        "math_accuracy":  results.get("math",  {}).get("accuracy", float("nan")),
    }))


if __name__ == "__main__":
    main()
