import random
import numpy as np
from tqdm import trange
from copy import deepcopy
from typing import Any, Dict, List
from easyeditor.models.crispedit.utils import update_model_and_tokenizer_with_appropriate_padding_token
from utils import print_time, prepare_requests_from_data_type, save_model_and_tokenizer, chunks
import os
from dotenv import load_dotenv
load_dotenv()
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import wandb

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.alphaedit_ft.AlphaEditFT_hparams import AlphaEditFTHyperParams
from easyeditor.models.alphaedit_ft.utils import calculate_projection_caches
from easyeditor.models.alphaedit_ft import ProjectedAdam
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
    parser.add_argument('--cache_sample_num', type=int, default=1000, help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--energy_threshold', type=float, default=0.9, help='Energy threshold for projection matrix computation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit', help='WandB project name.')
    args = parser.parse_args()
    return args

def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditFTHyperParams,
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

        if loss_meter.avg < 5e-2:
            break
    
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

if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    hparams = AlphaEditFTHyperParams.from_hparams(f"./hparams/AlphaEditFT/{args.model}") 
    hparams.batch_size = args.batch_size
    hparams.energy_threshold = args.energy_threshold
    hparams.mom2_n_samples = args.cache_sample_num
    save_model_name = f"{args.model}_{hparams.alg_name}_{args.data_type}_{args.energy_threshold}"
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