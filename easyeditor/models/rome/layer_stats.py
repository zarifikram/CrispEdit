import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally, make_loader
CACHE_DIR = "/data0/zikram/huggingface/datasets" # TO-DO: CHANGE TO YOURS

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
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            cache_dir=CACHE_DIR,
        )
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
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        raw_ds = raw_ds["train"].train_test_split(test_size=0.001, seed=69, shuffle=True)
        raw_ds['val'] = raw_ds.pop("test")

        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        maxlen = 2048  
        maxlen = 512  
        print(f"Max length is {maxlen}")
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    batch_size = 8 # Examine this many dataset texts at once
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
    model.gradient_checkpointing_enable()
    
    with torch.enable_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, model.device)
                labels = batch['input_ids'].clone()
                labels[labels == 0] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                
                model.zero_grad()
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
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        raw_ds = raw_ds["train"].train_test_split(test_size=0.001, seed=69, shuffle=True)
        raw_ds['val'] = raw_ds.pop("test")

        maxlen = get_max_length_from_model(model)

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        
        # maxlen = 2048  
        maxlen = 512  
        return TokenizedDataset(raw_ds["val"], tokenizer, maxlen=maxlen)

    batch_size = 4 # Examine this many dataset texts at once
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
        pin_memory=True,
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
                
                model.zero_grad()
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
