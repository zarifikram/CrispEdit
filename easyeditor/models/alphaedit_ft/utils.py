from ..rome.layer_stats import layer_stats_kfac
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .AlphaEditFT_hparams import AlphaEditFTHyperParams
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
    # Sb, Ub = torch.linalg.eigh(B)

    rank, null_threshold = get_rank_and_threshold_by_energy_ratio(Sa, percent=energy_threhold)
    print(f"Rank is {rank} out of {A.shape[0]} total, null threshold: {null_threshold}")

    P_A = Ua[:, rank: ] @ Ua[:, rank:].T
    # B_inv = torch.linalg.inv(B)

    return {'P_A': P_A}

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
    hparams: AlphaEditFTHyperParams,
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
    for i, layer in enumerate(hparams.layers):
        proj_layer_cache = calculate_projection_cache_by_layer(model, tok, layer, hparams, force_recompute)
        rewrite_module_name = hparams.rewrite_module_tmp.format(layer)
        proj_map[weights[rewrite_module_name]] = proj_layer_cache

    return proj_map