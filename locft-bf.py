import random
import numpy as np
from tqdm import trange
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List
from utils import print_time, prepare_requests_from_data_type, save_model_and_tokenizer, chunks
from easyeditor.models.crispedit.utils import update_model_and_tokenizer_with_appropriate_padding_token
import os
from dotenv import load_dotenv
load_dotenv()
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
import argparse
import wandb 

from crispedit import AverageMeter

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.ft.ft_hparams import FTHyperParams
from easyeditor.models.rome.layer_stats import calculate_cache_loss

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit', help='WandB project name.')
    args = parser.parse_args()
    return args

def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    device = model.device

    # This is from the original code. Likely this is incorrect, but we keep it for consistency.
    if tok.padding_side != "left":
        tok.padding_side = "left"

    # Corrected padding side
    # if tok.padding_side != "right":
    #     tok.padding_side = "right"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
            # requests[i]["target_new"] += tok.eos_token # Bad idea to add eos token. Kept for reference.
    
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    for name, w in model.named_parameters():
        w.requires_grad = name in weights
    
    old_task_loss = calculate_cache_loss(
                model,
                tok,
                "wikipedia",
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
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
            num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
            num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
            prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
            prompt_target_len = inputs_targets['input_ids'].size(1)
            label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
            
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            
            logits = model(**inputs_targets).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(bs, -1)
            loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
            loss = loss.mean()
                
            # print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

        old_task_loss = calculate_cache_loss(
            model,
            tok,
            "wikipedia",
            sample_size=100
        )
        wandb.log({f"FT Loss": loss_meter.avg, "Task 1 Loss": old_task_loss})

        if loss_meter.avg < 1e-2:
            break
    
    return model

def print_time(process_name):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f'{process_name}: {formatted_time}')


if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    hparams = FTHyperParams.from_hparams(f"./hparams/FT/{args.model}") 
    hparams.batch_size = args.batch_size

    save_model_name = f"{args.model}_{hparams.alg_name}_{args.data_type}"
    print(f"Model will be saved to BASE_DIR/{save_model_name}")
    wandb.init(project=args.wandb_project, name=save_model_name, config=vars(hparams))


    MODEL_NAME = hparams.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR, device_map='auto')
    device = model.device

    # set appropriate padding token
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams)

    print_time("Begin FT Time")
    edited_model = execute_ft(model, tokenizer, requests, hparams)
    print_time("End FT Time")

    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)