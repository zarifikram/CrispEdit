#!/usr/bin/env python3
import os
import json
import argparse
import random
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # stop_strings support varies by Transformers version
    use_stop_strings = False
    stop_strings = [tok.eos_token] if tok.eos_token else None

    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stop_strings=[tok.eos_token, "\n", "."],
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


def load_zsre(zsre_path: str) -> List[Dict[str, Any]]:
    with open(zsre_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {zsre_path}, got {type(data)}")
    return data


def sample_indices(n_total: int, n_samples: int, seed: int) -> List[int]:
    if n_samples > n_total:
        raise ValueError(f"Requested n_samples={n_samples} but dataset has only {n_total} items")
    rng = random.Random(seed)
    return rng.sample(range(n_total), n_samples)


def run_one_prompt(model, tok, question: str, context_type: str, max_new_tokens: int) -> Dict[str, Any]:
    full_prompt = build_input_prompt(question, tok, context_type=context_type)
    answer_clean, answer_raw, n_new, used_stop_strings = generate_answer(
        model, tok, full_prompt, max_new_tokens
    )

    out: Dict[str, Any] = {
        "question": question,
        "full_prompt": full_prompt,
        "answer": answer_clean,
        "full_text": full_prompt + (answer_clean or ""),
        "debug": {
            "new_tokens": n_new,
            "used_stop_strings": used_stop_strings,
        },
    }
    if not answer_clean:
        out["debug"]["answer_raw"] = answer_raw
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folder name (relative to HF_CACHE_DIR) or full path")
    ap.add_argument("--name", required=True, help="Name used for output JSON filename (e.g., methodA_run1)")
    ap.add_argument("--output_dir", default="outputs_zsre_no_ctx", help="Where to write JSON outputs")

    ap.add_argument("--zsre_path", default="data/zsre_mend_3k.json", help="Path to ZSRE JSON file")
    ap.add_argument("--n_samples", type=int, default=10, help="Number of cases to sample")
    ap.add_argument("--sample_seed", type=int, default=0, help="Seed for deterministic sampling")

    # Default per your request
    ap.add_argument("--context_type", default="no_context", choices=["qa_inst", "chat_temp", "no_context"])
    ap.add_argument("--max_new_tokens", type=int, default=50)
    args = ap.parse_args()

    model, tok, full_dir = load_model_and_tokenizer(args.model_dir)
    zsre = load_zsre(args.zsre_path)

    idxs = sample_indices(len(zsre), args.n_samples, args.sample_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.name}.json")

    cases_out: List[Dict[str, Any]] = []
    for i in idxs:
        ex = zsre[i]

        src_q = ex.get("src", "")

        # "original" vs "edited" answers in ZSRE-MEND:
        # - pred is the pre-edit model prediction (often equals original target)
        # - answers[0] is the dataset's original gold answer (often matches pred)
        # - alt is the edited target answer
        original_answer = ex.get("pred") or (ex.get("answers") or [None])[0]
        edited_answer = ex.get("alt")

        # Run model on src only
        gen = run_one_prompt(model, tok, src_q, args.context_type, args.max_new_tokens)
        # gen has: gen["answer"], gen["full_prompt"], gen["full_text"], etc.

        # If you specifically want "prompt with QA" (just Q/A without the leading instruction)
        prompt_qa = f"Q: {src_q}\nA:"

        cases_out.append(
            {
                "src": src_q,
                "original_answer_to_src": original_answer,
                "edited_answer_to_src": edited_answer,
                "model_output": gen["answer"],

                # Requested prompt strings
                "prompt_with_QA": prompt_qa,
                "prompt_with_full_template": gen["full_prompt"],
                "prompt_answer_with_full_template": gen["full_text"],
            }
        )

    payload: Dict[str, Any] = {
        "name": args.name,
        "model_dir": full_dir,
        "zsre_path": args.zsre_path,
        "n_samples": args.n_samples,
        "sample_seed": args.sample_seed,
        "sampled_indices": idxs,
        "context_type": args.context_type,
        "max_new_tokens": args.max_new_tokens,
        "tokenizer": {
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
        },
        "results": cases_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Loaded model from: {full_dir}")
    print(f"Primary device: {model.device}")
    print(f"Sample seed={args.sample_seed}, indices={idxs}")


if __name__ == "__main__":
    main()
