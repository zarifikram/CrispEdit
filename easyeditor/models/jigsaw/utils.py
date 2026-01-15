from ..rome.layer_stats import layer_stats_kfac, layer_stats_kfac_one_pass, layer_stats_kfac_with_txt_tgt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Jigsaw_hparams import JigsawHyperParams
from typing import Dict, Tuple
from dotenv import load_dotenv
import os

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")

def get_rank_and_threshold_by_energy_ratio(eigenvalues, percent=0.9):
    total_energy = torch.sum(eigenvalues)
    sorted_eigvals, _ = torch.sort(eigenvalues, descending=True)
    cumulative_energy = torch.cumsum(sorted_eigvals, dim=0)
    energy_ratio = cumulative_energy / total_energy

    rank = torch.searchsorted(energy_ratio, percent).item() + 1  # +1 for 0-based index
    threshold = sorted_eigvals[rank-1] if rank - 1 < len(sorted_eigvals) else 0.0
    return rank, threshold

def calculate_projection_cache_with_kfac(A, B, energy_threhold=0.9):
    # we will get A_inv, B_inv, U_A, U_B, M
    Sa, Ua = torch.linalg.eigh(A) 
    Sb, Ub = torch.linalg.eigh(B)

    M = torch.outer(Sa, Sb)
    rank, null_threshold = get_rank_and_threshold_by_energy_ratio(M.view(-1), percent=energy_threhold)
    M = M < null_threshold
    print(f"Rank is {rank} out of {A.shape[0]*B.shape[0]} total, null threshold: {null_threshold}")

    A_inv = None #torch.linalg.inv(A)
    B_inv = None #torch.linalg.inv(B)

    return {'Ua': Ua, 'Ub': Ub, 'M': M, }#'A_inv': A_inv, 'B_inv': B_inv}

def get_cov_ab(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    A, B = layer_stats_kfac(
        model,
        tok,
        layer_name,
        STATS_DIR,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        force_recompute=force_recompute,
    )

    return A, B

def calculate_projection_cache_by_layer(model, tok, layer, hparams, force_recompute):
    A, B = get_cov_ab(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    )

    # if model is not llama switch it up
    if hparams.model_name not in ["Llama3-8B","phi-1.5"]:
        A, B = B, A

    null_threshold = hparams.energy_threshold
    P_cache = calculate_projection_cache_with_kfac(A, B, energy_threhold=null_threshold)
    for key in P_cache:
        P_cache[key] = P_cache[key].to(model.device).to(model.dtype)
    return P_cache

def get_weights(
    model: AutoModelForCausalLM,
    hparams: JigsawHyperParams,
) -> Dict[str, torch.Tensor]:
    """
    Retrieves the weights that will be changed by the FT algorithm.
    :return: List of (name, weight tensor) tuples
    """
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and "bias" not in n
    }
    return weights

def calculate_projection_caches(model, tok, hparams, force_recompute=False):
    weights = get_weights(model, hparams)
    proj_map = {}
    
    # 1. Identify all target layers
    layer_name_map = {}
    for layer_num in hparams.layers:
        # Format the name: e.g., "layers.5.mlp.down_proj"
        layer_name = hparams.rewrite_module_tmp.format(layer_num)
        layer_name_map[layer_num] = layer_name

    target_layers = list(layer_name_map.values())

    # 2. Get covariance stats for ALL layers in ONE pass
    print(f"Retrieving covariance statistics for {len(target_layers)} layers...")
    
    stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=target_layers,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute
    )

    # 3. Compute projections for each layer
    for layer_num in hparams.layers:
        layer_name = layer_name_map[layer_num]
        A, B = stats_dict[layer_name]

        # Apply Model-Specific Logic (moved from calculate_projection_cache_by_layer)
        # If model is not llama/phi, swap A and B
        if hparams.model_name not in ["Llama3-8B", "phi-1.5"]:
            A, B = B, A

        # Calculate Projection
        null_threshold = hparams.energy_threshold
        # Ensure correct device/dtype
        P_cache = calculate_projection_cache_with_kfac(A, B, energy_threhold=null_threshold)
        
        for key in P_cache:
            P_cache[key] = P_cache[key].to(model.device).to(model.dtype)
            
        # Store in map
        proj_map[weights[layer_name]] = P_cache

    return proj_map

def update_projection_caches_with_request(weight_to_projection_cache, txt, tgt, model, tok, hparams):
    new_stats_dict = layer_stats_kfac_with_txt_tgt(
        model,
        tok,
        layer_names = [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers],
        txt=txt,
        tgt=tgt
    )

    weights = get_weights(model, hparams)

    for layer_num in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer_num)
        A_new, B_new, num_samples_new = new_stats_dict[layer_name]

        old_P_cache = weight_to_projection_cache[weights[layer_name]]
        A, B, num_samples_old = old_P_cache['A'], old_P_cache['B'], old_P_cache['num_samples']

        A_updated = (A * num_samples_old + A_new * num_samples_new) / (num_samples_old + num_samples_new)
        B_updated = (B * num_samples_old + B_new * num_samples_new) / (num_samples_old + num_samples_new)
        num_samples_updated = num_samples_old + num_samples_new

        null_threshold = hparams.energy_threshold
        P_cache_updated = calculate_projection_cache_with_kfac(A_updated, B_updated, energy_threhold=null_threshold)
        for key in P_cache_updated:
            P_cache_updated[key] = P_cache_updated[key].to(model.device).to(model.dtype)

        P_cache_updated['num_samples'] = num_samples_updated
        weight_to_projection_cache[weights[layer_name]] = P_cache_updated

    return weight_to_projection_cache


