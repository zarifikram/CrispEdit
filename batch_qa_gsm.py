#!/usr/bin/env python3
"""
GSM8K sample generation -> JSON (similar structure to your prior model sample generation).

- Deterministically samples N examples from GSM8K with --sample_seed so the same set can be reused across methods.
- Builds prompts with the same qa_inst template (default).
- Optionally appends a CoT trigger ("Let's think step by step.") to better match gsm8k_cot style.
- Saves outputs to: <output_dir>/<name>.json

Each result item contains ONLY:
  - src                           (GSM8K question text)
  - original_answer_to_src        (GSM8K gold final answer, extracted)
  - edited_answer_to_src          (None for GSM8K; kept for schema compatibility)
  - model_output
  - prompt_with_QA
  - prompt_with_full_template
  - prompt_answer_with_full_template
"""

import os
import json
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

COT = False


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

    # For decoder-only models, left padding is typically safer for batching.
    if getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"

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
            {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
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

    use_stop_strings = False
    stop_strings = [tok.eos_token] if tok.eos_token else None

    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            tokenizer=tok,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )
        use_stop_strings = True
    except TypeError:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
            use_cache=False,
        )

    gen_ids = gen[0][input_ids.shape[1] :]
    text_clean = tok.decode(gen_ids, skip_special_tokens=True)
    text_raw = tok.decode(gen_ids, skip_special_tokens=False)

    if not use_stop_strings:
        text_clean = _truncate_on_suffixes(text_clean, [".", "\n", tok.eos_token])
        text_raw = _truncate_on_suffixes(text_raw, [".", "\n", tok.eos_token])

    return text_clean, text_raw, int(gen_ids.numel()), use_stop_strings


def extract_gsm8k_final(answer_field: str) -> str:
    """
    GSM8K 'answer' commonly contains rationale + '#### <final>'.
    Return the final string after #### if present, else the whole string stripped.
    """
    if not isinstance(answer_field, str):
        return ""
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    return answer_field.strip()


def load_gsm8k(split: str, local_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads GSM8K examples into a list of dicts containing at least:
      - question
      - answer
    """
    if local_path:
        # Expect a JSONL file with fields similar to HF GSM8K (question/answer).
        data: List[Dict[str, Any]] = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    # Otherwise use HF datasets (requires datasets installed + dataset available/cache)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install datasets` "
            "or pass --gsm8k_local_path to load from a local JSONL."
        ) from e

    ds = load_dataset("gsm8k", "main", split=split)
    return [dict(x) for x in ds]


def sample_indices(n_total: int, n_samples: int, seed: int) -> List[int]:
    if n_samples > n_total:
        raise ValueError(f"Requested n_samples={n_samples} but dataset has only {n_total} items")
    rng = random.Random(seed)
    return rng.sample(range(n_total), n_samples)


def main():
    ap = argparse.ArgumentParser()

    # Model + output
    ap.add_argument("--model_dir", required=True, help="Folder name (relative to HF_CACHE_DIR) or full path")
    ap.add_argument("--name", required=True, help="Name used for output JSON filename (e.g., methodA_gsm8k_samples)")
    ap.add_argument("--output_dir", default="outputs/gsm8k", help="Where to write JSON outputs")

    # GSM8K loading / sampling
    ap.add_argument("--gsm8k_split", default="test", choices=["train", "test"], help="HF GSM8K split")
    ap.add_argument("--gsm8k_local_path", default=None, help="Optional local GSM8K JSONL path (bypasses HF datasets)")
    ap.add_argument("--n_samples", type=int, default=10, help="How many GSM8K problems to sample")
    ap.add_argument("--sample_seed", type=int, default=0, help="Seed for deterministic sampling")

    # Prompting / generation
    ap.add_argument("--context_type", default="chat_temp", choices=["qa_inst", "chat_temp", "no_context"])
    ap.add_argument("--max_new_tokens", type=int, default=256)

    args = ap.parse_args()

    model, tok, full_dir = load_model_and_tokenizer(args.model_dir)
    gsm8k = load_gsm8k(args.gsm8k_split, local_path=args.gsm8k_local_path)

    idxs = sample_indices(len(gsm8k), args.n_samples, args.sample_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.name}.json")

    results: List[Dict[str, Any]] = []
    for i in idxs:
        ex = gsm8k[i]
        q = (ex.get("question") or "").strip()
        a_full = (ex.get("answer") or "").strip()

        # "src" corresponds to the main prompt text
        src = q + ("\n\nLet's think step by step." if COT else "")

        # For GSM8K there is no "edited answer"; keep it explicitly null.
        original_answer = extract_gsm8k_final(a_full)
        edited_answer = None

        full_prompt = build_input_prompt(src, tok, args.context_type)
        answer_clean, answer_raw, n_new, used_stop_strings = generate_answer(
            model, tok, full_prompt, args.max_new_tokens
        )

        prompt_with_QA = f"Q: {src}\nA:"

        item: Dict[str, Any] = {
            "src": src,
            "original_answer_to_src": original_answer,
            "edited_answer_to_src": edited_answer,
            "model_output": answer_clean,

            "prompt_with_QA": prompt_with_QA,
            "prompt_with_full_template": full_prompt,
            "prompt_answer_with_full_template": full_prompt + (answer_clean or ""),
        }

        # Optional debug info (comment out if you want *only* the 6 fields above)
        item["_meta"] = {
            "dataset_index": i,
            "split": args.gsm8k_split,
            "used_stop_strings": used_stop_strings,
            "new_tokens": n_new,
        }
        if not answer_clean:
            item["_meta"]["model_output_raw"] = answer_raw

        results.append(item)

    payload = {
        "name": args.name,
        "model_dir": full_dir,
        "gsm8k_split": args.gsm8k_split,
        "gsm8k_local_path": args.gsm8k_local_path,
        "n_samples": args.n_samples,
        "sample_seed": args.sample_seed,
        "sampled_indices": idxs,
        "context_type": args.context_type,
        "cot": bool(COT),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Loaded model from: {full_dir}")
    print(f"Primary device: {model.device}")
    print(f"Sample seed={args.sample_seed}, indices={idxs}")


if __name__ == "__main__":
    main()
