import gc
import wandb

from easyeditor.models.crispedit.projected_adam import ProjectedAdam
from easyeditor.models.crispedit.projected_sgd import ProjectedSGD
from ..rome.layer_stats import layer_stats_kfac, layer_stats_kfac_one_pass, layer_stats_kfac_with_txt_tgt, layer_stats_kfac_fisher_with_txt_tgt, calculate_cache_loss, calculate_request_loss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .CrispEdit_hparams import CrispEditHyperParams
from typing import Dict, List, Tuple, Union
from dotenv import load_dotenv
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType
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

def calculate_projection_cache_with_kfac(A, B, energy_threshold=0.9):
    Sa, Ua = torch.linalg.eigh(A)
    Sb, Ub = torch.linalg.eigh(B)

    M = torch.outer(Sa, Sb)
    rank, null_threshold = get_rank_and_threshold_by_energy_ratio(M.view(-1), percent=energy_threshold)
    M = M < null_threshold
    print(f"Rank is {rank} out of {A.shape[0]*B.shape[0]} total, null threshold: {null_threshold}")

    return {'Ua': Ua, 'Ub': Ub, 'M': M}

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

    energy_threshold = hparams.energy_threshold
    P_cache = calculate_projection_cache_with_kfac(A, B, energy_threhold=energy_threshold)
    for key in P_cache:
        P_cache[key] = P_cache[key].to(model.device).to(model.dtype)
    return P_cache

def get_weights(
    model: AutoModelForCausalLM,
    hparams: CrispEditHyperParams,
    bias: bool,
    to_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    bias = False # always ignore bias for now
    weights = {
        n: (p.detach().cpu().clone() if to_cpu else p)
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and (bias or ("bias" not in n))
    }
    return weights

def calculate_cov_cache_with_old_data(model, tok, hparams, force_recompute=False) -> Dict[str, Dict]:
    if hparams.no_crisp:
        return None
    
    layer_to_cov_cache = {}
    
    layer_name_map = {}
    for layer_num in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer_num)
        layer_name_map[layer_num] = layer_name

    target_layers = list(layer_name_map.values())
    
    stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=target_layers,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples,
        precision=hparams.mom2_dtype,
        force_recompute=force_recompute
    )

    for layer_num in hparams.layers:
        layer_name = layer_name_map[layer_num]
        A, B, num_samples = stats_dict.pop(layer_name)

        cov_cache = {'A': A.to("cpu", dtype=torch.float32), 'B': B.to("cpu", dtype=torch.float32), 'num_samples': num_samples}
            
        layer_to_cov_cache[layer_name] = cov_cache
        del A, B

    return layer_to_cov_cache

# def tighten_projection_caches(weight_to_projection_cache, hparams, weights, device):
#     layer_to_projection_cache = {}
#     for layer_num in hparams.layers:
#         layer_name = hparams.rewrite_module_tmp.format(layer_num)
#         P_cache = weight_to_projection_cache.pop(weights[layer_name])
#         A, B, num_samples = P_cache['A'], P_cache['B'], P_cache['num_samples']
#         del P_cache
#         torch.cuda.empty_cache()

#         A, B = A * num_samples, B * num_samples  # Scale back to sum of squares
#         layer_to_projection_cache[layer_name] = {'B': A.to(device), 'A': B.to(device), 'num_samples': num_samples}

#     return layer_to_projection_cache

def calculate_cov_cache_with_request(txt, tgt, model, tok, hparams):
    if hparams.no_crisp:
        return None
    
    layer_to_cov_cache = {}
    cov_stats_dict = layer_stats_kfac_with_txt_tgt(
        model,
        tok,
        layer_names = [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers],
        txt=txt,
        tgt=tgt,
        precision=hparams.mom2_dtype,
        sample_size=hparams.edit_n_samples,
        to_collect=["mom2"],
        add_pretrain_data=(hparams.edit_cache_style == "mix"),
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    for layer_num in hparams.layers:
        layer_name = hparams.rewrite_module_tmp.format(layer_num)
        A, B, num_samples = cov_stats_dict.pop(layer_name)

        A = A.to(model.device, non_blocking=True)
        B = B.to(model.device, non_blocking=True)
    
        cov_cache = {'A': A.to("cpu", dtype=torch.float32), 'B': B.to("cpu", dtype=torch.float32), 'num_samples': num_samples}
        layer_to_cov_cache[layer_name] = cov_cache

        del A, B
        torch.cuda.empty_cache()

    return layer_to_cov_cache

def cache_weights_to_cpu(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(weights, dict):
        return {
            name: param.detach().cpu().clone()
            for name, param in weights.items()
        }    
    else:
        raise ValueError("Input must be a torch.Tensor or a dict of Tensors.")
    
def is_weights_changed(current_weights, cached_weights, threshold: float) -> bool:
    for name, param in current_weights.items():
        cached_param = cached_weights[name]
        change = torch.norm(param.detach().cpu() - cached_param) / torch.norm(cached_param)
        if change > threshold:
            print(f"Weight {name} changed by {change:.4f}, exceeding threshold {threshold}.")
            return True
    return False

def recalculate_cov_cache_if_weights_changed(model, tok, hparams, current_weights_cpu, layer_to_cov_cache) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], bool]:
    if not hparams.recalculate_cache or hparams.no_crisp: ### Early exit if not recalculating or we are not using CrispEdit
        return current_weights_cpu, layer_to_cov_cache, False
    
    weights = get_weights(model, hparams, bias=True)
    if not is_weights_changed(weights, current_weights_cpu, hparams.recalculate_weight_threshold):
        return current_weights_cpu, layer_to_cov_cache, False
        
    del layer_to_cov_cache, weights
    gc.collect()
    torch.cuda.empty_cache()
    layer_to_cov_cache = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=True
    )
    
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)

    return current_weights_cpu, layer_to_cov_cache, True

def calculate_old_loss(model, tok, hparams):
    if hparams.disable_old_loss_check:
        return {}
    with torch.no_grad():
        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100
        )
    return {"Task 1 Loss": old_task_loss}

def calculate_old_edit_loss(txt_chunks, tgt_chunks, model, tok):
    if len(txt_chunks) == 0:
        return {}
    mets = {}
    with torch.no_grad():
        for i, (txt, tgt) in enumerate(zip(txt_chunks, tgt_chunks)):
            request_loss = calculate_request_loss(
                model,
                tok,
                txt,
                tgt,
                sample_size=10
            )
            mets.update({f"OLD_EDIT_LOSS/Old Edit Loss Chunk {i}": request_loss})
    avg_loss = sum(mets.values()) / len(mets)
    mets.update({"Task 2 Loss": avg_loss})
    return mets

def build_optimizer_with_cov_caches(model, hparams, layer_to_cov_caches: List[Dict[str, Dict]], opt = None):
    if hparams.no_crisp and opt is not None:
        return opt

    if hparams.no_crisp:
        weights = get_weights(model, hparams, bias=True)
        return torch.optim.Adam(
            [v for _, v in weights.items()],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
        )
    if layer_to_cov_caches:
        combined_layer_to_cov_cache = combine_layer_to_cov_caches(layer_to_cov_caches)
        weight_to_projection_cache = calculate_projection_caches_from_cov_caches(model, hparams, combined_layer_to_cov_cache)
    else:
        weight_to_projection_cache = None
    
    if opt is not None:
        opt.reset_cache(weight_to_projection_cache)
        return opt
    
    weights = get_weights(model, hparams, bias=True)
    # return ProjectedSGD(
    #     [v for _, v in weights.items()],
    #     projection_cache_map = weight_to_projection_cache,
    #     lr=hparams.lr,
    # )
    return ProjectedAdam(
        [v for _, v in weights.items()],
        projection_cache_map = weight_to_projection_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    
def combine_layer_to_cov_caches(layer_to_cov_caches: List[Dict[str, Dict]], normalize_trace_with_first=False) -> Dict[str, Dict]:
    if len(layer_to_cov_caches) == 1:
        return layer_to_cov_caches[0]
    combined_layer_to_cov_caches = {}
    for layer_name in layer_to_cov_caches[0].keys():
        A_list = [layer_to_cov[layer_name]['A'] for layer_to_cov in layer_to_cov_caches]
        B_list = [layer_to_cov[layer_name]['B'] for layer_to_cov in layer_to_cov_caches]
        num_samples_list = [layer_to_cov[layer_name]['num_samples'] for layer_to_cov in layer_to_cov_caches]
        combined_A = sum([A * num_sample for A, num_sample in zip(A_list, num_samples_list)]) / sum(num_samples_list)
        combined_B = sum([B * num_sample for B, num_sample in zip(B_list, num_samples_list)]) / sum(num_samples_list)
        combined_num_samples = sum(num_samples_list)

        combined_layer_to_cov_caches[layer_name] = {
            'A': combined_A,
            'B': combined_B,
            'num_samples': combined_num_samples
        }
    print(f"Combined samples {num_samples_list}")
    return combined_layer_to_cov_caches

def calculate_projection_caches_from_cov_caches(model, hparams, layer_to_cov_caches, energy_threshold=None):
    weight_to_projection_cache = {}
    weights = get_weights(model, hparams, bias=False)
    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache['A'].to(model.device)
        B = cov_cache['B'].to(model.device)
        energy_threshold = hparams.energy_threshold if energy_threshold is None else energy_threshold
        projection_cache = calculate_projection_cache_with_kfac(A, B, energy_threshold=energy_threshold)
        weight_to_projection_cache[weights[layer_name]] = projection_cache
    return weight_to_projection_cache

def get_weights_to_projection_cache(model, opt, hparams):
    weights_to_projection_cache = opt.param_groups[0]['projection_cache_map']
    weights = get_weights(model, hparams, bias=False)
    layers = [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]
    layer_to_projection_cache = {layer: weights_to_projection_cache[weights[layer]] for layer in layers}
    return layer_to_projection_cache

def wrap_model_with_lora_and_return_opt(model, hparams):
    if hparams.lora_type == "lora":
        lora_config = LoraConfig
    elif hparams.lora_type == "adalora":
        lora_config = AdaLoraConfig
    peft_config = lora_config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.lora_rank,
            lora_alpha=hparams.lora_alpha, 
            lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
    )
    peft_model = get_peft_model(model, peft_config)
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    return peft_model, opt

def update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams):
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer