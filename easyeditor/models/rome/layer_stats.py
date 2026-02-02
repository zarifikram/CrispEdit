import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import random
from datasets import Dataset
from typing import List

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally, make_loader
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = os.getenv("HF_DATASETS_DIR")

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )

def get_eval_txt_and_tgt(txt, tgt, sample_size=None, is_augmented=False):
    if sample_size and sample_size < len(txt):
        indices = random.sample(range(len(txt)), sample_size)
        txt_eval = [txt[i] for i in indices] if not is_augmented else ["Please answer the question:\n\nQ: " + txt[i] + "\nA:" for i in indices]
        tgt_eval = [tgt[i] for i in indices] if not is_augmented else [tgt[i].strip() for i in indices]
    else:
        txt_eval = [t for t in txt] if not is_augmented else ["Please answer the question:\n\nQ: " + t + "\nA:" for t in txt]
        tgt_eval = [t for t in tgt] if not is_augmented else [t.strip() for t in tgt]
    return txt_eval, tgt_eval

def get_in_and_out_dim_from_layer(layer, layer_name):
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        in_dim = layer.in_features
        out_dim = layer.out_features
    elif hasattr(layer, 'weight'):
        # Conv1D or other layers with weight attribute
        weight_shape = layer.weight.shape
        if len(weight_shape) == 2:
            # For Conv1D in GPT-2: weight is (in_features, out_features)
            # But Conv1D transposes, so we need to check
            if hasattr(layer, 'nf'):  # GPT-2 Conv1D has nf attribute
                out_dim = layer.nf
                in_dim = weight_shape[0]
            else:
                # Standard case
                out_dim, in_dim = weight_shape
        else:
            raise ValueError(f"Layer {layer_name} has unexpected weight shape: {weight_shape}")
    else:
        raise ValueError(f"Layer {layer_name} does not have recognizable dimension attributes")
    return in_dim, out_dim
    
def get_num_positions_from_model(model):
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096
    return npos

def get_max_length_from_model(model):
    if hasattr(model.config, 'n_positions'):
        maxlen = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        maxlen = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        maxlen = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        maxlen = model.config.seq_length
    else:
        raise NotImplementedError
            
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            maxlen = model.config.sliding_window or 4096
        else:
            maxlen = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
        maxlen = 4096

    return maxlen        

def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_wiki_ds(ds_name)
        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    npos = get_num_positions_from_model(model)

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")
    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat

def layer_stats_kfac(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_wiki_ds(ds_name)
        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        maxlen = 2048  
        maxlen = 512  
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 1 # Examine this many dataset texts at once
    npos = get_num_positions_from_model(model)

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    
    # Compute KFAC matrices A and B
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz"
    filename = stats_dir / file_extension
    
    print(f"Computing KFAC matrices A and B locally....")
    
    if filename.exists() and not force_recompute:
        print(f"Loading cached KFAC matrices from {filename}")
        loaded = torch.load(filename)
        return loaded['A'], loaded['B']
    
    ds = get_ds()
    if progress is None:
        progress = lambda x: x
    
    # Get layer to determine dimensions
    layer_name = layer_name.split(".weight")[0] if ".weight" in layer_name else layer_name
    layer = dict(model.named_modules())[layer_name]
    in_dim, out_dim = get_in_and_out_dim_from_layer(layer, layer_name)
    
    A = torch.zeros((in_dim, in_dim), dtype=dtype, device=model.device)
    B = torch.zeros((out_dim, out_dim), dtype=dtype, device=model.device)
    N = 0
    total_tokens = 0
    
    captured_input = None
    captured_grad_output = None

    def save_input_hook(module, input_tuple, output):
        nonlocal captured_input
        captured_input = input_tuple[0].detach()
        
        output.requires_grad_(True)
        
        def capture_grad(grad):
            nonlocal captured_grad_output
            captured_grad_output = grad.detach()
        
        # Register a hook on the tensor to capture the gradient w.r.t the output
        output.register_hook(capture_grad)

    h_input = layer.register_forward_hook(save_input_hook)
    
        
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    
    batch_count = -(-(sample_size or len(ds)) // batch_size)

    # remember the grads before torch.no_grad() so we can restore them later
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = param.requires_grad
        param.requires_grad = False

    model.requires_grad_(False)
    # model.gradient_checkpointing_enable()
    
    with torch.enable_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                labels = batch['input_ids'].clone()
                labels[labels == 0] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100, 
                    reduction='sum'
                )
                loss.backward()
                
                with torch.no_grad():
                    feat_in = captured_input.detach().to(model.device)
                    grad_out = captured_grad_output.detach().to(model.device)

                    feat_in = feat_in[:, :-1, :] 
                    grad_out = grad_out[:, :-1, :]

                    valid_mask = (shift_labels != -100) # Shape: (Batch, Seq_Len-1)
                    
                    input_flat = feat_in[valid_mask].to(dtype=dtype)       # (N_valid, Dim_in)
                    grad_flat = grad_out[valid_mask].to(dtype=dtype)       # (N_valid, Dim_out)

                    A.addmm_(input_flat.T, input_flat)
                    B.addmm_(grad_flat.T, grad_flat)
                    
                    current_valid_tokens = input_flat.shape[0]
                    total_tokens += current_valid_tokens
                    N += batch['input_ids'].size(0)

                    del feat_in, grad_out, input_flat, grad_flat
                
                if sample_size is not None and N >= sample_size:
                    break
            
            if sample_size is not None and N >= sample_size:
                break
    
    h_input.remove()
    
    A /= total_tokens
    B /= total_tokens

    if not force_recompute:
        print(f"Saving KFAC matrices to {filename}")
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'A': A, 'B': B, 'N': total_tokens}, filename)
    
    # restore them now
    for name, param in model.named_parameters():
        param.requires_grad = grads[name]
        
    A, B = A.to(model.device), B.to(model.device)
    return A, B

def layer_stats_kfac_one_pass(
    model,
    tokenizer,
    layer_names: List[str],
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
    force_recompute=False,
):
    # --- 1. Setup paths and check cache for ALL layers ---
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]
    stats_dir = Path(stats_dir)
    
    results = {}
    missing_layers = []

    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"

    # Check which layers are already cached
    for layer_name in layer_names:
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz"
        filename = stats_dir / file_extension
        
        if filename.exists() and not force_recompute:
            loaded = torch.load(filename, map_location='cpu')
            results[layer_name] = (loaded['A'].to(dtype=dtype), loaded['B'].to(dtype=dtype), loaded['N'])
        else:
            missing_layers.append(layer_name)

    if not missing_layers:
        return results

    # print(f"Recalculating KFAC for {len(missing_layers)} layers: {missing_layers}")

    # --- 2. Dataset and Batching Logic (Restored from original) ---
    def get_ds():
        raw_ds = load_wiki_ds(ds_name)

        maxlen = get_max_length_from_model(model) # Ensure this helper is imported
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        # Hardcoded overrides from your original snippet
        maxlen = 2048
        maxlen = 512 
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 1
    npos = get_num_positions_from_model(model) # Ensure this helper is imported

    # FIX: This was missing in the previous version, causing the TypeError
    if batch_tokens is None:
        batch_tokens = npos * 3 
        
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix

    ds = get_ds()


    # --- 3. Initialize Matrices and Hooks for MISSING layers ---
    matrices = {}
    handles = []
    
    # State storage for hooks
    captured_inputs = {}
    captured_grads = {}

    def get_hook(l_name):
        def save_input_hook(module, input_tuple, output):
            captured_inputs[l_name] = input_tuple[0].detach()
            output.requires_grad_(True)
            def capture_grad(grad):
                captured_grads[l_name] = grad.detach()
            output.register_hook(capture_grad)
        return save_input_hook

    for layer_name in missing_layers:
        # Resolve module
        target_name = layer_name.split(".weight")[0] if ".weight" in layer_name else layer_name
        module = dict(model.named_modules())[target_name]
        
        # Init A and B
        in_dim, out_dim = get_in_and_out_dim_from_layer(module, target_name)
        matrices[layer_name] = {
            "A": torch.zeros((in_dim, in_dim), dtype=dtype, device=model.device),
            "B": torch.zeros((out_dim, out_dim), dtype=dtype, device=model.device)
        }
        
        # Register Hook
        handles.append(module.register_forward_hook(get_hook(layer_name)))

    # --- 4. Training Loop ---
    
    # Restore the 'stat' object logic
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    
    loader = tally(
        stat,
        ds,
        cache=None, # We don't use the cache arg here because we manage multiple files manually
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    
    batch_count = -(-(sample_size or len(ds)) // batch_size)

    # Backup gradients state
    grads = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters(): p.requires_grad = False
    model.requires_grad_(False)
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    
    N = 0
    total_tokens = 0

    with torch.enable_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                labels = batch['input_ids'].clone()
                labels[labels == 0] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                loss.backward()

                # UPDATE STEP: Iterate over all active layers
                with torch.no_grad():
                    # Check if we captured anything (safeguard)
                    if not captured_inputs:
                        continue

                    # Mask is shared across layers for the same batch
                    valid_mask = (shift_labels != -100)
                    current_valid_tokens = valid_mask.sum().item()

                    for layer_name in missing_layers:
                        if layer_name not in captured_inputs or layer_name not in captured_grads:
                            continue
                        
                        feat_in = captured_inputs[layer_name].to(model.device)
                        grad_out = captured_grads[layer_name].to(model.device)
                        
                        # Truncate to match valid mask logic (seq_len - 1)
                        feat_in = feat_in[:, :-1, :]
                        grad_out = grad_out[:, :-1, :]

                        input_flat = feat_in[valid_mask].to(dtype=dtype)
                        grad_flat = grad_out[valid_mask].to(dtype=dtype)
                        
                        matrices[layer_name]["A"].addmm_(input_flat.T, input_flat)
                        matrices[layer_name]["B"].addmm_(grad_flat.T, grad_flat)

                        # if A or B has null, breakpoint
                        if torch.any(torch.isnan(matrices[layer_name]["A"])) or torch.any(torch.isinf(matrices[layer_name]["A"])):
                            print(f"NaN or Inf detected in A matrix of layer {layer_name}")
                            breakpoint()

                    # Update counters (only once per batch)
                    total_tokens += current_valid_tokens
                    N += batch['input_ids'].size(0)

                    # Clear captures for next batch
                    captured_inputs.clear()
                    captured_grads.clear()
                
                if sample_size is not None and N >= sample_size: break
            if sample_size is not None and N >= sample_size: break

    # --- 5. Cleanup and Save ---
    for h in handles: h.remove()
    for name, param in model.named_parameters(): 
        param.requires_grad = grads[name]

    for layer_name in missing_layers:
        A = matrices[layer_name]["A"] / total_tokens
        B = matrices[layer_name]["B"] / total_tokens
        
        # Save individually to match original file structure
        # if force_recompute then we skip saving
        if not force_recompute:
            file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_kfac{size_suffix}.npz"
            filename = stats_dir / file_extension
            filename.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'A': A, 'B': B, 'N': total_tokens}, filename)
        
        results[layer_name] = (A, B, total_tokens)

    return results

def layer_stats_kfac_with_txt_tgt_old(
    model,
    tokenizer,
    layer_names: List[str],
    txt,
    tgt,
    sample_size=None,
    model_name=None,
    precision=None,
):
    # --- 1. Setup paths and check cache for ALL layers ---
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]
    
    results = {}

    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    # print(f"Recalculating KFAC for {len(layer_names)} layers: {layer_names} for given txt/tgt")

    batch_size = 128
    layer_to_cov_cache = {}
    total_tokens = 0


    # --- 3. Initialize Matrices and Hooks for MISSING layers ---
    handles = []
    
    # State storage for hooks
    captured_inputs = {}
    captured_grads = {}

    def get_hook(l_name):
        def save_input_hook(module, input_tuple, output):
            captured_inputs[l_name] = input_tuple[0].detach()
            output.requires_grad_(True)
            def capture_grad(grad):
                captured_grads[l_name] = grad.detach()
            output.register_hook(capture_grad)
        return save_input_hook

    for layer_name in layer_names:
        # Resolve module
        target_name = layer_name.split(".weight")[0] if ".weight" in layer_name else layer_name
        module = dict(model.named_modules())[target_name]

        in_dim, out_dim = get_in_and_out_dim_from_layer(module, target_name)
        layer_to_cov_cache[layer_name] = {
            "A": torch.zeros((in_dim, in_dim), dtype=dtype, device=model.device),
            "B": torch.zeros((out_dim, out_dim), dtype=dtype, device=model.device)
        }
        
        handles.append(module.register_forward_hook(get_hook(layer_name)))
    
    # Backup gradients state
    grads = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters(): p.requires_grad = False
    model.requires_grad_(False)
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    txt_eval, tgt_eval = get_eval_txt_and_tgt(txt, tgt, sample_size)
    
    with torch.enable_grad():
        # possibly the worst code i've ever written in a while...
        for txt_edit, tgt_edit in tqdm(zip(chunks(txt_eval, batch_size), chunks(tgt_eval, batch_size)), total=len(txt_eval)//batch_size):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt_edit, tgt_edit)]
            encodings = tokenizer(inputs_targets, return_tensors="pt", padding=True).to(model.device)
            labels = encodings["input_ids"].clone()
            attention_mask = encodings["attention_mask"]
            
            for i, prompt in enumerate(txt_edit):
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
                # Set prompt tokens to -100 so they are ignored by CrossEntropy and your valid_mask
                labels[i, :prompt_len] = -100
                
            check_labels_masking(labels, tgt_edit, tokenizer)

            labels[labels == 0] = -100
            labels[labels == tokenizer.pad_token_id] = -100
                    
            model.zero_grad(set_to_none=True)
            outputs = model(**encodings, use_cache=False)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            loss.backward()

            # UPDATE STEP: Iterate over all active layers
            with torch.no_grad():
                # Check if we captured anything (safeguard)
                assert captured_inputs is not None, "Did not really capture anything. Double check?"

                # Mask is sharedsh across layers for the same batch
                valid_mask = (shift_labels != -100)
                # valid_mask = attention_mask[:, :-1].bool()
                current_valid_tokens = valid_mask.sum().item()

                for layer_name in layer_names:
                    if layer_name not in captured_inputs or layer_name not in captured_grads:
                        continue

                    feat_in = captured_inputs[layer_name].to(model.device)
                    grad_out = captured_grads[layer_name].to(model.device)

                    # Truncate to match valid mask logic (seq_len - 1)
                    feat_in = feat_in[:, :-1, :]
                    grad_out = grad_out[:, :-1, :]
                    kfac_dtype = layer_to_cov_cache[layer_name]["A"].dtype
                    input_flat = feat_in[valid_mask].to(dtype=kfac_dtype)
                    grad_flat = grad_out[valid_mask].to(dtype=kfac_dtype)
                    layer_to_cov_cache[layer_name]["A"].addmm_(input_flat.T, input_flat)
                    layer_to_cov_cache[layer_name]["B"].addmm_(grad_flat.T, grad_flat)
                    # print(f"Layer {layer_name}: Added {input_flat.shape[0]} tokens to KFAC computation. trace of B is now {torch.trace(layer_to_cov_cache[layer_name]['B']).item()}")
                # Update counters (only once per batch)
                total_tokens += current_valid_tokens

                # Clear captures for next batch
                captured_inputs.clear()
                captured_grads.clear()
                    
    # --- 5. Cleanup and Save ---
    for h in handles: h.remove()
    for name, param in model.named_parameters(): 
        param.requires_grad = grads[name]

    for layer_name in layer_names:
        cov_cache = layer_to_cov_cache.pop(layer_name)
        layer_to_cov_cache[layer_name] = (cov_cache["A"].to("cpu")/total_tokens, cov_cache["B"].to("cpu")/total_tokens, total_tokens)
        del cov_cache
        torch.cuda.empty_cache()

    return layer_to_cov_cache

def layer_stats_kfac_with_txt_tgt(
    model,
    tokenizer,
    layer_names: List[str],
    txt,
    tgt,
    to_collect,
    add_pretrain_data = False,
    pretrain_sample_size = 100,
    sample_size=None,
    model_name=None,
    precision=None,
    batch_tokens=None,
    progress=tqdm,
):
    # --- 1. Setup paths and check cache for ALL layers ---
    def get_ds(txt, tgt):
        all_texts = [t + g for t, g in zip(txt, tgt)]
        if add_pretrain_data:
            raw_ds_pretrain = load_wiki_ds("wikipedia")["train"]
            pretrain_texts = get_shuffled_subset_texts(
                raw_ds_pretrain, 
                sample_size=pretrain_sample_size,
                seed=69
            )
            print(f"Edit txts {len(all_texts)} + Pretrain txts {len(pretrain_texts)}")
            all_texts = all_texts + pretrain_texts

        raw_ds = create_text_dataset(all_texts)
        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        maxlen = 2048  
        maxlen = 512  
        return TokenizedDataset(raw_ds, tokenizer, maxlen=maxlen)
    
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]
    
    npos = get_num_positions_from_model(model) # Ensure this helper is imported

    if batch_tokens is None:
        batch_tokens = npos * 3 
        
    txt_eval, tgt_eval = get_eval_txt_and_tgt(txt, tgt, sample_size, is_augmented=False)
    ds = get_ds(txt_eval, tgt_eval)

    results = {}

    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    batch_size = 8 # Tune batch size for memory capacity
    layer_to_cov_cache = {}
    total_tokens = 0


    # --- 3. Initialize Matrices and Hooks for MISSING layers ---
    handles = []
    
    # State storage for hooks
    captured_inputs = {}
    captured_grads = {}

    def get_hook(l_name):
        def save_input_hook(module, input_tuple, output):
            captured_inputs[l_name] = input_tuple[0].detach()
            output.requires_grad_(True)
            def capture_grad(grad):
                captured_grads[l_name] = grad.detach()
            output.register_hook(capture_grad)
        return save_input_hook

    for layer_name in layer_names:
        # Resolve module
        target_name = layer_name.split(".weight")[0] if ".weight" in layer_name else layer_name
        module = dict(model.named_modules())[target_name]

        in_dim, out_dim = get_in_and_out_dim_from_layer(module, target_name)
        layer_to_cov_cache[layer_name] = {
            "A": torch.zeros((in_dim, in_dim), dtype=dtype, device=model.device),
            "B": torch.zeros((out_dim, out_dim), dtype=dtype, device=model.device)
        }
        
        handles.append(module.register_forward_hook(get_hook(layer_name)))
    
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    
    loader = tally(
        stat,
        ds,
        cache=None, # We don't use the cache arg here because we manage multiple files manually
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=False,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)


    # Backup gradients state
    grads = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters(): p.requires_grad = False
    model.requires_grad_(False)
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    with torch.enable_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                labels = batch['input_ids'].clone()
                labels[labels == 0] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                loss.backward()

                with torch.no_grad():
                    if not captured_inputs:
                        continue
                    valid_mask = (shift_labels != -100)
                    current_valid_tokens = valid_mask.sum().item()

                    for layer_name in layer_names:
                        if layer_name not in captured_inputs or layer_name not in captured_grads:
                            continue

                        feat_in = captured_inputs[layer_name].to(model.device)
                        grad_out = captured_grads[layer_name].to(model.device)

                        # Truncate to match valid mask logic (seq_len - 1)
                        feat_in = feat_in[:, :-1, :]
                        grad_out = grad_out[:, :-1, :]
                        kfac_dtype = layer_to_cov_cache[layer_name]["A"].dtype
                        input_flat = feat_in[valid_mask].to(dtype=kfac_dtype)
                        grad_flat = grad_out[valid_mask].to(dtype=kfac_dtype)
                        layer_to_cov_cache[layer_name]["A"].addmm_(input_flat.T, input_flat)
                        layer_to_cov_cache[layer_name]["B"].addmm_(grad_flat.T, grad_flat)
                
                total_tokens += current_valid_tokens

                # Clear captures for next batch
                captured_inputs.clear()
                captured_grads.clear()
                    
    # --- 5. Cleanup and Save ---
    for h in handles: h.remove()
    for name, param in model.named_parameters(): 
        param.requires_grad = grads[name]

    for layer_name in layer_names:
        cov_cache = layer_to_cov_cache.pop(layer_name)
        layer_to_cov_cache[layer_name] = (cov_cache["A"].to("cpu")/total_tokens, cov_cache["B"].to("cpu")/total_tokens, total_tokens)
        del cov_cache
        torch.cuda.empty_cache()

    return layer_to_cov_cache

def layer_stats_kfac_fisher_with_txt_tgt(
    model,
    tokenizer,
    layer_names: List[str],
    txt,
    tgt,
    sample_size=None,
    model_name=None,
    precision=None,
    temperature=1.0, # Added temperature argument (default 1.0)
):
    # --- 1. Setup paths and check cache for ALL layers ---
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]
    
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)

    # print(f"Computing Fisher KFAC (KL/Sampling) for {len(layer_names)} layers")

    batch_size = 1
    layer_to_cov_cache = {}
    total_tokens = 0

    # --- 2. Initialize Matrices and Hooks ---
    handles = []
    
    # State storage for hooks
    captured_inputs = {}
    captured_grads = {}

    def get_hook(l_name):
        def save_input_hook(module, input_tuple, output):
            captured_inputs[l_name] = input_tuple[0].detach()
            output.requires_grad_(True)
            def capture_grad(grad):
                captured_grads[l_name] = grad.detach()
            output.register_hook(capture_grad)
        return save_input_hook

    for layer_name in layer_names:
        # Resolve module
        target_name = layer_name.split(".weight")[0] if ".weight" in layer_name else layer_name
        module = dict(model.named_modules())[target_name]

        # Helper to get dims (assuming this function exists in your scope)
        # If not, you might need to import it or rely on module.weight.shape
        try:
            in_dim, out_dim = get_in_and_out_dim_from_layer(module, target_name)
        except NameError:
            # Fallback if helper function is missing
            if hasattr(module, 'weight'):
                out_dim, in_dim = module.weight.shape
            elif hasattr(module, 'in_features'):
                in_dim, out_dim = module.in_features, module.out_features
            else:
                raise ValueError(f"Could not determine dims for {layer_name}")

        layer_to_cov_cache[layer_name] = {
            "A": torch.zeros((in_dim, in_dim), dtype=dtype, device=model.device),
            "B": torch.zeros((out_dim, out_dim), dtype=dtype, device=model.device)
        }
        
        handles.append(module.register_forward_hook(get_hook(layer_name)))
    
    # Backup gradients state
    grads = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters(): p.requires_grad = False
    model.requires_grad_(False)

    # Data Sampling
    txt_eval, tgt_eval = get_eval_txt_and_tgt(txt, tgt, sample_size)

    with torch.enable_grad():
        for txt_edit, tgt_edit in tqdm(zip(chunks(txt_eval, batch_size), chunks(tgt_eval, batch_size)), total=len(txt_eval)//batch_size):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt_edit, tgt_edit)]
            encodings = tokenizer(inputs_targets, return_tensors="pt", padding=True).to(model.device)
            labels = encodings["input_ids"].clone()
            
            # Masking setup
            for i, prompt in enumerate(txt_edit):
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
                labels[i, :prompt_len] = -100

            labels[labels == 0] = -100
            labels[labels == tokenizer.pad_token_id] = -100
                    
            model.zero_grad(set_to_none=True)
            outputs = model(**encodings, use_cache=False)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            shift_logits = logits[..., :-1, :].contiguous() # [B, Seq-1, Vocab]
            shift_labels = labels[..., 1:].contiguous()     # [B, Seq-1]

            scaled_logits = shift_logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1) # [B, S, V]

            probs_flat = probs.view(-1, probs.size(-1))
            sampled_indices = torch.multinomial(probs_flat, 1).view(probs.size(0), probs.size(1)) # [B, S]

            grad_z = -probs / temperature 
        
            grad_z.scatter_add_(-1, sampled_indices.unsqueeze(-1), 
                                torch.ones_like(grad_z) * (1.0 / temperature))

            valid_mask = (shift_labels != -100) # [B, S]
            current_valid_tokens = valid_mask.sum().item()
            
            grad_z = grad_z * valid_mask.unsqueeze(-1).to(grad_z.dtype)

            shift_logits.backward(gradient=grad_z, retain_graph=False)
            

            with torch.no_grad():
                assert captured_inputs is not None, "Did not really capture anything. Double check?"

                for layer_name in layer_names:
                    if layer_name not in captured_inputs or layer_name not in captured_grads:
                        continue

                    feat_in = captured_inputs[layer_name].to(model.device)
                    grad_out = captured_grads[layer_name].to(model.device)

                    # Truncate to match valid mask logic (seq_len - 1)
                    feat_in = feat_in[:, :-1, :]
                    grad_out = grad_out[:, :-1, :]
                    kfac_dtype = layer_to_cov_cache[layer_name]["A"].dtype
                    
                    input_flat = feat_in[valid_mask].to(dtype=kfac_dtype)
                    grad_flat = grad_out[valid_mask].to(dtype=kfac_dtype)
                    
                    layer_to_cov_cache[layer_name]["A"].addmm_(input_flat.T, input_flat)
                    layer_to_cov_cache[layer_name]["B"].addmm_(grad_flat.T, grad_flat)

                # Update counters
                total_tokens += current_valid_tokens

                # Clear captures
                captured_inputs.clear()
                captured_grads.clear()
                    
    # --- 3. Cleanup and Save ---
    for h in handles: h.remove()
    for name, param in model.named_parameters(): 
        param.requires_grad = grads[name]

    for layer_name in layer_names:
        cov_cache = layer_to_cov_cache.pop(layer_name)
        # Avoid division by zero
        denom = total_tokens if total_tokens > 0 else 1
        layer_to_cov_cache[layer_name] = (cov_cache["A"].to("cpu")/denom, cov_cache["B"].to("cpu")/denom, total_tokens)
        # layer_to_cov_cache[layer_name] = (cov_cache["A"].to("cpu")/denom, torch.ones_like(cov_cache["B"].to("cpu")/denom), total_tokens)
        del cov_cache
        torch.cuda.empty_cache()

    return layer_to_cov_cache

def calculate_request_loss(model, tokenizer, txt, tgt, sample_size=1):
    """
    Calculates the average cross-entropy loss per target token.
    Ignores the prompt (txt) tokens and padding in the loss calculation.
    """
    total_loss = 0.0
    total_tokens = 0

    # randomly sample sample_size examples from txt and tgt
    txt_eval, tgt_eval = get_eval_txt_and_tgt(txt, tgt, sample_size, is_augmented=False)
    # Ensure model is in eval mode and we don't store unnecessary gradients
    model.eval()
    
    batch_size = 1
    with torch.no_grad():
        for txt_edit, tgt_edit in tqdm(zip(chunks(txt_eval, batch_size), chunks(tgt_eval, batch_size)), total=len(txt_eval)//batch_size, disable=sample_size < 50):
            inputs_targets = [t + g for t, g in zip(txt_edit, tgt_edit)]
            encodings = tokenizer(inputs_targets, return_tensors="pt", padding=True).to(model.device)
            
            labels = encodings["input_ids"].clone()
            
            for i, prompt in enumerate(txt_edit):
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
                labels[i, :prompt_len] = -100

            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(**encodings)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            current_valid_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += current_valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    model.train()
    return avg_loss
    
def calculate_cache_loss(
    model,
    tokenizer,
    ds_name,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_wiki_ds(ds_name)

        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        # maxlen = 2048  
        maxlen = 512  
        return TokenizedDataset(raw_ds["val"], tokenizer, maxlen=maxlen)

    batch_size = 1 # Examine this many dataset texts at once
    npos = get_num_positions_from_model(model)

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"

    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.rsplit("/")[-1]

    ds = get_ds()
    if progress is None:
        progress = lambda x: x

    loader = make_loader(
        ds,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=False,
        random_sample=1,
        num_workers=2,
    )
                         
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    total_loss_sum = 0.0
    total_valid_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                
                labels = batch['input_ids'].clone()
                labels[labels == 0] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                
                model.zero_grad(set_to_none=True)
                outputs = model(**batch, use_cache=False)                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100, 
                    reduction='sum'
                )
                
                valid_token_count = (shift_labels != -100).sum().item()
                total_loss_sum += loss.item()
                total_valid_tokens += valid_token_count

    model.train()
    
    # Prevent division by zero if something went wrong
    if total_valid_tokens == 0:
        return 0.0
        
    final_avg_loss = total_loss_sum / total_valid_tokens
    return final_avg_loss

if __name__ == "__main__":
    main()

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def check_labels_masking(labels, tgt_batch, tokenizer):
    """
    Asserts that the 'labels' tensor contains ONLY the 'tgt_batch' tokens 
    (i.e., prompts are masked with -100).
    """
    # Iterate through every sample in the current batch
    for i in range(len(tgt_batch)):
        
        # 1. Get the single row of labels and remove -100s
        row_labels = labels[i]
        valid_ids = row_labels[row_labels != -100]
        
        # 2. Decode back to text
        decoded_target = tokenizer.decode(valid_ids, skip_special_tokens=True)
        
        # 3. Compare with the expected string
        # Using .strip() handles potential leading-space weirdness from tokenization
        expected = tgt_batch[i].strip()
        actual = decoded_target.strip()
        
        assert actual == expected, (
            f"Label Masking Failed at batch index {i}!\n"
            f"Expected (Target): '{expected}'\n"
            f"Actual (Decoded):  '{actual}'"
        )

def create_text_dataset(text_list):
    data_dict = {
        "id": [str(i) for i in range(len(text_list))],
        "url": ["http://placeholder-url.com"] * len(text_list),
        "title": ["Placeholder Title"] * len(text_list),
        "text": text_list
    }
    
    return Dataset.from_dict(data_dict)

def load_wiki_ds(ds_name):
    raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    raw_ds = raw_ds["train"].train_test_split(test_size=0.001, seed=69, shuffle=True)
    raw_ds['val'] = raw_ds.pop("test")
    return raw_ds

def get_shuffled_subset_texts(dataset, sample_size, seed=42):
    shuffled_ds = dataset.shuffle(seed=seed)
    return shuffled_ds[:sample_size]['text']