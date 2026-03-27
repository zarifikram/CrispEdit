#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_model_and_tokenizer(edited_model_dir_local: str):
    prefix = os.getenv("HF_CACHE_DIR", "")
    full_dir = os.path.join(prefix, edited_model_dir_local) if prefix else edited_model_dir_local

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
        # Keep your original behavior: apply_chat_template(tokenize=False) then tokenize the string
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
    # Remove trailing suffixes exactly like your original logic
    for s in suffixes:
        if s and text.endswith(s):
            text = text[: -len(s)]
    return text


@torch.inference_mode()
def generate_answer(model, tok, input_prompt: str, max_new_tokens: int):
    enc = tok(input_prompt, return_tensors="pt")
    # device expects cuda:N style; model is device-mapped, so use model.device for input tensors
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

    # Try to use stop_strings if your Transformers version supports it.
    # If not supported, fallback to post-truncation.
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

    # Decoder-only: slice off the prompt
    gen_ids = gen[0][input_ids.shape[1]:]

    # Two decodes: clean and raw for debugging
    text_clean = tok.decode(gen_ids, skip_special_tokens=True)
    text_raw = tok.decode(gen_ids, skip_special_tokens=False)

    # If stop_strings wasn’t used by generate(), emulate your old behavior:
    if not use_stop_strings:
        # Remove trailing '.' '\n' eos_token if present (original logic)
        text_clean = _truncate_on_suffixes(text_clean, [".", "\n", tok.eos_token])
        text_raw = _truncate_on_suffixes(text_raw, [".", "\n", tok.eos_token])

    return text_clean, text_raw, int(gen_ids.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited_model_dir", required=True, help="Folder name (relative to HF_CACHE_DIR) or full path")
    ap.add_argument("--context_type", default="chat_temp", choices=["qa_inst", "chat_temp", "no_context"])
    ap.add_argument("--max_new_tokens", type=int, default=50)
    args = ap.parse_args()

    model, tok, full_dir = load_model_and_tokenizer(args.edited_model_dir)
    print(f"Loaded: {full_dir}")
    print(f"Model device (primary): {model.device}")
    print(f"eos_token={tok.eos_token!r} eos_token_id={tok.eos_token_id} pad_token={tok.pad_token!r} pad_token_id={tok.pad_token_id}")
    print("Type ':q' to quit.\n")

    while True:
        try:
            q = input("QA> ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", "quit", "exit"):
            break

        prompt = build_input_prompt(q, tok, args.context_type)
        out_clean, out_raw, n_new = generate_answer(model, tok, prompt, args.max_new_tokens)

        print("\n--- Prompt ---")
        print(prompt)
        print("\n--- Output (skip_special_tokens=True) ---")
        print(out_clean if out_clean else "[EMPTY]")
        print(f"\n[debug] new_tokens={n_new}")
        if not out_clean:
            print("\n--- Output RAW (skip_special_tokens=False) ---")
            print(out_raw if out_raw else "[EMPTY RAW]")
        print("\n" + ("-" * 60) + "\n")


if __name__ == "__main__":
    main()
