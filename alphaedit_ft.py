import random
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional, Union
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1087"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1087"
os.environ["HF_DATASETS_CACHE"] = "/data0/zikram/huggingface/datasets"
import argparse
import ast

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor.models.alphaedit_ft.AlphaEditFT_hparams import AlphaEditFTHyperParams
from easyeditor.evaluate.evaluate import compute_edit_quality
from easyeditor.editors.utils import _prepare_requests, summary_metrics
from easyeditor.models.alphaedit_ft.utils import calculate_projection_caches, get_weights
from easyeditor.models.alphaedit_ft import ProjectedAdam
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
    hparams: AlphaEditFTHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    """
    device = torch.device(f'cuda:{hparams.device}')
    
    # if tok.padding_side != "left":
    #     tok.padding_side = "left"
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    weight_to_projection_cache = calculate_projection_caches(
        model, tok, hparams, force_recompute=False
    )
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if (hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD") and hparams.objective_optimization == "target_new": # used to be LLM-judge
            # requests[i]["target_new"] += tok.eos_token
            requests[i]["target_new"] += ""
        # print(
        #     f"Executing FT algo for: "
        #     f"[{requests[i]['prompt']}] -> [{requests[i]['target_new']}]"
        # )
    
    # Retrieve weights that user desires to change
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


    print(f"Weights to be updated: {list(weights.keys())}")
    old_task_loss = calculate_cache_loss(
        model,
        tok,
        hparams.mom2_dataset,
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
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
            # txt_ids = tok(txt, return_tensors="pt").to(model.device)

            # 2. Create labels from input_ids
            labels = encodings["input_ids"].clone()

            # 3. Mask the prompt and padding using -100
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

            

        print(f"Total loss {loss_meter.avg}")
        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100
        )
        print(f"Old task loss: {old_task_loss}")

        if loss_meter.avg < 5e-2:
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
    parser.add_argument('--layer', required=False, type=int)
    parser.add_argument('--rewrite_module', required=True, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--save_model_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki'])
    parser.add_argument('--edited_model_dir', required=False, type=str, default=None, help='Path to edited model for evaluation. If None, we train a model and evaluate it.')
    parser.add_argument('--eval_num', required=False, type=int, default=3000, help='Number of evaluation instances to use.')
    parser.add_argument('--cache_sample_num', type=int, default=1000, help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--energy_threshold', type=float, default=0.9, help='Energy threshold for projection matrix computation.')
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
    hparams = AlphaEditFTHyperParams.from_hparams(f"./hparams/AlphaEditFT/{args.model}") 
    hparams.evaluation_type = "WILD"
    hparams.device = args.device
    # hparams.layers = [args.layer]
    hparams.rewrite_module_tmp = args.rewrite_module
    hparams.batch_size = args.batch_size
    hparams.energy_threshold = args.energy_threshold
    hparams.mom2_n_samples = args.cache_sample_num
    

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

    # before evaluation, always make sure tokenizer padding side is correct
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
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