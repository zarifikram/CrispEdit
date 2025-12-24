import random
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional, Union
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = "7, 4, 5, 6"
os.environ["HF_DATASETS_CACHE"] = "/data0/zikram/huggingface/datasets"
import argparse
import ast

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.ft.ft_hparams import FTHyperParams
from easyeditor.evaluate.evaluate import compute_edit_quality
from easyeditor.editors.utils import _prepare_requests, summary_metrics
from easyeditor.models.rome.layer_stats import calculate_cache_loss

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

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
    device = torch.device(f'cuda:{hparams.device}')
    if tok.padding_side != "left":
        tok.padding_side = "left"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if (hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD") and hparams.objective_optimization == "target_new": # used to be LLM-judge
            # requests[i]["target_new"] += tok.eos_token
            requests[i]["target_new"] += ""
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    print(f"Weights to be updated: {list(weights.keys())}")
    
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
    print(f"Old task loss: {old_task_loss}")

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        # Shuffle requests for each epoch
        random.shuffle(requests)
        # Define inputs
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
        print(f"Old task loss: {old_task_loss}")
        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break
    
    return model


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


def print_time(process_name):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f'{process_name}: {formatted_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--layer', required=True, type=int)
    parser.add_argument('--rewrite_module', required=True, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--save_model_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki'])
    parser.add_argument('--edited_model_dir', required=False, type=str, default=None, help='Path to edited model for evaluation. If None, we train a model and evaluate it.')
    parser.add_argument('--eval_num', required=False, type=int, default=3000, help='Number of evaluation instances to use.')
    args = parser.parse_args()
    
    # load data 
    data_file = {"zsre": "zsre_mend_eval_3k",
                 "counterfact": "counterfact-edit_3k",
                 "wiki": "wiki_big_edit_3k"}[args.data_type]
    data = json.load(open(f"./data/{data_file}.json", 'r', encoding='utf-8'))
    
    # process data
    datatype = args.data_type
    if datatype == 'counterfact':
        prompts = [d['prompt'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase_prompt'] for d in data]
        target_new = [d['target_new'] for d in data]
        locality_prompts = [d['locality_prompt'] for d in data]
        locality_ans = [d['locality_ground_truth'] for d in data]
    elif datatype == 'zsre':
        prompts = [d['src'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase'] for d in data]
        target_new = [d['alt'] for d in data]
        locality_prompts = [d['loc'] for d in data]
        locality_ans = [d['loc_ans'] for d in data]
    
    ground_truth = ['<|endoftext|>' for d in data]  
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    # prepare requests
    requests = _prepare_requests(prompts, target_new, ground_truth, None, rephrase_prompts, locality_inputs)
    hparams = FTHyperParams.from_hparams(f"./hparams/FT/{args.model}") 
    hparams.evaluation_type = "WILD"
    hparams.device = args.device
    # hparams.layers = [args.layer]
    hparams.rewrite_module_tmp = args.rewrite_module
    hparams.batch_size = args.batch_size

    MODEL_NAME = hparams.model_name
    if args.edited_model_dir is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/data0/zikram/huggingface/hub/")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="/data0/zikram/huggingface/hub/", device_map='auto')
    else:
        edited_model_dir = "/data0/zikram/huggingface/hub/" + args.edited_model_dir
        tokenizer = AutoTokenizer.from_pretrained(edited_model_dir)
        model = AutoModelForCausalLM.from_pretrained(edited_model_dir, device_map='auto')
    device = model.device


    # set appropriate padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    

    all_metrics = []

    print_time("Begin FT Time")
    
    if args.edited_model_dir is not None:
        edited_model = model
    else:
        edited_model = execute_ft(model, tokenizer, requests, hparams)

    print_time("End FT Time")

    if args.edited_model_dir is None:
        save_directory = "/data0/zikram/huggingface/hub/" + args.save_model_dir
        edited_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

    print_time("Begin Post Edit Eval Time")

    requests = random.sample(requests, len(requests))
    if args.eval_num is not None:
        requests = requests[:args.eval_num]

    for i, request in enumerate(tqdm(requests)):

        metrics = {
            'case_id': i,
            "requested_rewrite": request,
            "pre": {},
            "post": compute_edit_quality(edited_model, MODEL_NAME, hparams, tokenizer, request, hparams.device),
        }
        all_metrics.append(metrics)

        print(f"{i} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[i]}")

    summary_metrics(all_metrics)   

    print_time("End Post Edit Eval Time") 