import random
from copy import deepcopy
from typing import Any, Dict, List
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from utils import chunks

from easyeditor.models.jigsaw.Jigsaw_hparams import JigsawHyperParams
from easyeditor.models.jigsaw.utils import cache_weights_to_cpu, calculate_cov_cache_with_old_data, calculate_cov_cache_with_request, build_optimizer_with_cov_caches, recalculate_cov_cache_if_weights_changed, combine_layer_to_cov_caches, log_old_loss, get_weights
from easyeditor.models.jigsaw import ProjectedAdam

def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: JigsawHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    device = model.device
    
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    
    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )

    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)
    
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    log_old_loss(model, tok, hparams)
    
    loss_meter = AverageMeter()
    for it in trange(hparams.num_steps):
        loss_meter.reset()

        random.shuffle(requests)
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        # split into batches
        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(inputs_targets, return_tensors="pt", padding=True).to(device)

            labels = encodings["input_ids"].clone()

            labels[labels == tok.pad_token_id] = -100
            for i, prompt in enumerate(txt):
                prompt_len = len(tok(prompt, add_special_tokens=True)["input_ids"])
                labels[i, :prompt_len] = -100
            opt.zero_grad(set_to_none=True)
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
                
            loss_meter.update(loss.item(), n=labels.size(0))
            
            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
                current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                    model,
                    tok,
                    hparams,
                    current_weights_cpu,
                    layer_to_cov_cache_old,
                )
                if should_recalculate:
                    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old], opt=opt)
                print(f"Step {it} Batch Loss: {loss.item()}")

        log_old_loss(model, tok, hparams)
        wandb.log({f"FT Loss": loss_meter.avg})
        
        if loss_meter.avg < 1e-2:
            break
    
    return model

def execute_ft_sequential(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: JigsawHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    device = model.device
    
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )
    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    weights = get_weights(model, hparams, bias=True)
    current_weights_cpu = cache_weights_to_cpu(weights)
    
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    log_old_loss(model, tok, hparams)
    layer_to_cov_cache_data = None
    
    loss_meter = AverageMeter()
    random.shuffle(requests)
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    # split into batches
    for txt_edit, tgt_edit in zip(
        chunks(texts, hparams.num_edits), chunks(targets, hparams.num_edits)
    ):
        for it in trange(hparams.num_steps):
            loss_meter.reset()
            for txt, tgt in zip(
                chunks(txt_edit, hparams.batch_size), chunks(tgt_edit, hparams.batch_size)
            ):
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                encodings = tok(inputs_targets, return_tensors="pt", padding=True).to(device)

                labels = encodings["input_ids"].clone()

                labels[labels == tok.pad_token_id] = -100
                for i, prompt in enumerate(txt):
                    prompt_len = len(tok(prompt, add_special_tokens=True)["input_ids"])
                    labels[i, :prompt_len] = -100
                opt.zero_grad()
                outputs = model(**encodings, labels=labels)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN or Inf detected in loss")
                    breakpoint()
                if loss.item() >= 1e-2:
                    loss.backward()
                    opt.step()
                    current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                        model,
                        tok,
                        hparams,
                        current_weights_cpu,
                        layer_to_cov_cache_old,
                    )
                    if should_recalculate:
                        opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old] if layer_to_cov_cache_data is None else [layer_to_cov_cache_data, layer_to_cov_cache_old], opt=opt)

                loss_meter.update(loss.item(), n=labels.size(0))
            if loss_meter.avg < 1e-2:
                break

        layer_to_cov_cache_data_new = calculate_cov_cache_with_request(
            txt_edit,
            tgt_edit,
            model,
            tok,
            hparams,
        )
        layer_to_cov_cache_data = layer_to_cov_cache_data_new if layer_to_cov_cache_data is None else combine_layer_to_cov_caches([layer_to_cov_cache_data, layer_to_cov_cache_data_new])
        opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_data, layer_to_cov_cache_old], opt=opt)

        log_old_loss(model, tok, hparams)

    return model

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_requests_for_safeedit(requests: List[Dict]) -> List[Dict]:
    # just a simple way to make safeedit dataset work with old jigsaw code
    if "target_new" in requests[0]:
        return requests

    new_requests = []
    for request in requests:
        new_request = {
            'prompt': request['question'],
            'target_new': request['target_unsafe'],
        }
        new_requests.append(new_request)
    return new_requests
