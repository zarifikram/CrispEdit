import random
from copy import deepcopy
from typing import Any, Dict, List
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from utils import chunks

from easyeditor.models.jigsaw.Jigsaw_hparams import JigsawHyperParams
from easyeditor.models.jigsaw.utils import calculate_projection_caches
from easyeditor.models.jigsaw import ProjectedAdam
from easyeditor.models.rome.layer_stats import calculate_cache_loss



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
    
    weight_to_projection_cache = calculate_projection_caches(
        model, tok, hparams, force_recompute=False
    )
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Configure optimizer / gradients
    opt = ProjectedAdam(
        [v for _, v in weights.items()],
        projection_cache_map = weight_to_projection_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    old_task_loss = calculate_cache_loss(
        model,
        tok,
        hparams.mom2_dataset,
        sample_size=100
    )

    wandb.log({"Task 1 Loss": old_task_loss})
    
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
            opt.zero_grad()
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
                
            loss_meter.update(loss.item(), n=labels.size(0))
            
            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100
        )
        wandb.log({f"FT Loss": loss_meter.avg, "Task 1 Loss": old_task_loss})
        
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
    
    weight_to_projection_cache = calculate_projection_caches(
        model, tok, hparams, force_recompute=False
    )
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Configure optimizer / gradients
    opt = ProjectedAdam(
        [v for _, v in weights.items()],
        projection_cache_map = weight_to_projection_cache,
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    old_task_loss = calculate_cache_loss(
        model,
        tok,
        hparams.mom2_dataset,
        sample_size=100
    )

    wandb.log({"Task 1 Loss": old_task_loss})
    
    loss_meter = AverageMeter()
    random.shuffle(requests)
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    # split into batches
    for txt, tgt in zip(
        chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
    ):
        loss_meter.reset()
        for it in trange(hparams.num_steps):
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
            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
                
            loss_meter.update(loss.item(), n=labels.size(0))
            if loss_meter.avg < 1e-2:
                break

        # now get A and B for new samples
        # Update A, B. Calculate P caches
        # And update Adam.

        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100
        )
        wandb.log({"Task 1 Loss": old_task_loss})
        
    
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
