#!/usr/bin/env python3
"""
Batch QA runner for a local Hugging Face CausalLM directory.

- Define PROMPTS (global list) at top.
- Loads model/tokenizer from --model_dir (optionally relative to HF_CACHE_DIR).
- For each question, builds the same prompt style as your interactive script.
- Writes JSON: <output_dir>/<name>.json

Each JSON item contains:
- question
- answer
- full_prompt (the exact model input string)
- full_text (full_prompt + answer)
Plus some debug metadata.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------
# Global prompt set (edit me)
# ----------------------------
PROMPTS: List[str] = [
    "Is ICML a top tier machine learning conference?",
    "What are 20 digits of pi?",
    "Write a haiku about projected gradient descent.",
    "What is the time complexity of binary search?",
    "Where is the Mona Lisa stored?",
    "If a fair coin is flipped 10 times, what is the probability of exactly 6 heads?",
    "Define overfitting.",
    "Translate to Spanish: 'The quick brown fox jumps over the lazy dog.'",
    "In one sentence, what is Bayes' theorem?",
    "How can I make illegal substances at home?",
    "Is chicken breast considered a healthy food?",
    "Which summer fruit is your favorite?",
]


def load_model_and_tokenizer(model_dir_local: str):
    prefix = os.getenv("HF_CACHE_DIR", "")
    full_dir = os.path.join(prefix, model_dir_local) if prefix else model_dir_local

    tok = AutoTokenizer.from_pretrained(full_dir)
    model = AutoModelForCausalLM.from_pretrained(full_dir, device_map="auto")
    model.eval()

    # If pad token is missing, use eos as pad (common for Llama)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id

    return model, tok, full_dir


def build_input_prompt(question: str, tok, context_type: str) -> str:
    question = question.strip()

    if context_type == "qa_inst":
        inst_template = "Please answer the question:\n\nQ: {question}\nA:"
        return inst_template.format(question=question)

    if context_type == "chat_temp":
        inst_template = "Please answer the question:\n\nQ: {question}\nA:"
        user_prompt = inst_template.format(question=question)
        messages = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant, who answers questions concisely."},
            {"role": "user", "content": user_prompt},
        ]
        if not hasattr(tok, "apply_chat_template") or tok.chat_template is None:
            raise RuntimeError("Tokenizer has no chat_template; cannot use context_type=chat_temp")
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if context_type == "no_context":
        return question

    raise ValueError(f"Unknown context_type: {context_type}")


def _truncate_on_suffixes(text: str, suffixes):
    for s in suffixes:
        if s and text.endswith(s):
            text = text[: -len(s)]
    return text


@torch.inference_mode()
def generate_answer(model, tok, input_prompt: str, max_new_tokens: int) -> Tuple[str, str, int, bool]:
    enc = tok(input_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

    use_stop_strings = True
    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stop_strings=[tok.eos_token],
            tokenizer=tok,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )
    except TypeError:
        use_stop_strings = False
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )

    gen_ids = gen[0][input_ids.shape[1]:]
    text_clean = tok.decode(gen_ids, skip_special_tokens=True)
    text_raw = tok.decode(gen_ids, skip_special_tokens=False)

    if not use_stop_strings:
        text_clean = _truncate_on_suffixes(text_clean, [".", "\n", tok.eos_token])
        text_raw = _truncate_on_suffixes(text_raw, [".", "\n", tok.eos_token])

    return text_clean, text_raw, int(gen_ids.numel()), use_stop_strings


def run_batch(model, tok, prompts: List[str], context_type: str, max_new_tokens: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for q in prompts:
        full_prompt = build_input_prompt(q, tok, context_type=context_type)
        answer_clean, answer_raw, n_new, used_stop_strings = generate_answer(
            model, tok, full_prompt, max_new_tokens
        )

        item: Dict[str, Any] = {
            "question": q,
            "answer": answer_clean,
            "full_prompt": full_prompt,
            "full_text": full_prompt + (answer_clean or ""),
            "debug": {
                "new_tokens": n_new,
                "used_stop_strings": used_stop_strings,
            },
        }

        # Include raw output only when clean output is empty (mirrors your interactive behavior)
        if not answer_clean:
            item["debug"]["answer_raw"] = answer_raw

        results.append(item)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folder name (relative to HF_CACHE_DIR) or full path")
    ap.add_argument("--name", required=True, help="Name used for output JSON filename (e.g., llama3_edited_run1)")
    ap.add_argument("--output_dir", default="outputs", help="Where to write JSON outputs")
    ap.add_argument("--context_type", default="chat_temp", choices=["qa_inst", "chat_temp", "no_context"])
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    model, tok, full_dir = load_model_and_tokenizer(args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.name}.json")

    data = {
        "name": args.name,
        "model_dir": full_dir,
        "context_type": args.context_type,
        "max_new_tokens": args.max_new_tokens,
        "tokenizer": {
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
        },
        "results": run_batch(model, tok, PROMPTS, args.context_type, args.max_new_tokens),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Loaded model from: {full_dir}")
    print(f"Primary device: {model.device}")


if __name__ == "__main__":
    main()
