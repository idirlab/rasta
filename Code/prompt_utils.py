"""
Code/prompt_utils.py — shared prompt-template logic (VERBATIM port).

Why this file exists:
In the original PiSSA repo, train.py wraps every instruction in an
Alpaca-style template before tokenizing, but utils/gen_vllm.py (used at
eval time) feeds the dataset's raw "instruction" column straight to the
model with NO template applied.  That's fine as long as the dataset's
"instruction" column is itself raw text.  It is NOT fine for the
`python` split of fxmeng/pissa-dataset: each row's "instruction" field
already contains the fully-rendered Alpaca prompt (header,
"### Instruction:", "### Response:", everything).  So the original
train.py double-wraps the prompt during training while gen_vllm.py uses
it single-wrapped at eval time — a real train/inference mismatch.

Fix: both the training data module (data_code.py) and generate.py call
format_prompt() with the SAME template setting, so training and
generation always agree.  Default is "none" (use the instruction column
as-is), the correct setting for the `python` split.  S20 in
utils/sanity_check.py pins the template bytes and the "none" default.
"""

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def format_prompt(instruction: str, template: str) -> str:
    if template == "none":
        return instruction
    if template == "alpaca":
        return ALPACA_TEMPLATE.format(instruction=instruction)
    raise ValueError(f"Unknown prompt template: {template!r} "
                     "(expected 'none' or 'alpaca')")
