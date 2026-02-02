import random
import numpy as np
import os
from easyeditor.models.crispedit.utils import update_model_and_tokenizer_with_appropriate_padding_token
from crispedit import *
from dotenv import load_dotenv
load_dotenv()
from utils import (
    print_time, 
    prepare_requests_from_data_type, 
    save_model_and_tokenizer, 
)

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import wandb

import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from easyeditor.models.crispedit.CrispEdit_hparams import CrispEditHyperParams

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'zsre10k', 'counterfact', 'wiki', 'safeedit_train', 'safeedit_test'])
    parser.add_argument('--cache_sample_num', type=int, default=1000, help='Number of samples to use for caching projection matrices.')
    parser.add_argument('--edit_sample_num', type=int, default=1000, help='Number of samples to use for calculating old loss during editing.')
    parser.add_argument('--energy_threshold', type=float, default=0.9, help='Energy threshold for projection matrix computation.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--num_edits', type=int, default=100, help='Sequential edit batch')
    parser.add_argument('--sequential_edit', action='store_true', help='Whether to use sequential editing or not. Default is False.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit', help='WandB project name.')
    parser.add_argument('--recalculate_cache', action='store_true', help='Whether to recalculate the projection caches. Default is False.')
    parser.add_argument('--recalculate_weight_threshold', type=float, default=0.25, help='Threshold for recalculating weight projection caches. [0.0-1.0]')
    parser.add_argument('--no_crisp', action='store_true', help='Disable CrispEdit optimization even if available.')
    parser.add_argument('--disable_old_loss_check', action='store_true', help='Disable old loss check to speed up sequential editing.')
    parser.add_argument('--edit_cache_style', type=str, default='mix', choices=['sequential', 'mix', 'disable'], help='Style of cache data to use during sequential editing. Sequential: cache from current chunk and combine with previous chunk cache. Mix: sample data from pretrain and previous chunks. Disable: do not use cache data.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging.')
    # LoRA-specific args can be added here if needed
    parser.add_argument('--perform_lora', action='store_true', help='Whether to use LoRA for editing.')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank if LoRA is used.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha if LoRA is used.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout if LoRA is used.')
    parser.add_argument('--lora_type', type=str, default='lora', choices=['lora', 'adalora'], help='Type of LoRA to use.')
    parser.add_argument('--target_modules', type=list, default=["q_proj", "v_proj"], help='Target modules for LoRA adaptation.')
    args = parser.parse_args()
    return args

def get_hparams(args):
    hparams = CrispEditHyperParams.from_hparams(f"./hparams/CrispEdit/{args.model}")
    hparams.batch_size = args.batch_size
    hparams.energy_threshold = args.energy_threshold
    hparams.mom2_n_samples = args.cache_sample_num
    hparams.edit_n_samples = args.edit_sample_num
    hparams.recalculate_cache = args.recalculate_cache
    hparams.recalculate_weight_threshold = args.recalculate_weight_threshold
    hparams.no_crisp = args.no_crisp
    hparams.disable_old_loss_check = args.disable_old_loss_check
    hparams.edit_cache_style = args.edit_cache_style
    hparams.perform_lora = args.perform_lora

    assert not (not args.no_crisp and args.perform_lora), "We don't currently support using CrispEdit and LoRA together. Please set --no_crisp if you want to use LoRA."
    if hparams.perform_lora and args.sequential_edit:
        print("Warning: We suggest using edit.py for LoRA-based sequential editing instead of this one.")

    if hparams.perform_lora:
        hparams.lora_rank = args.lora_rank
        hparams.lora_alpha = args.lora_alpha
        hparams.lora_dropout = args.lora_dropout
        hparams.lora_type = args.lora_type
        hparams.target_modules = args.target_modules

    if args.sequential_edit:
        assert args.num_edits >= args.batch_size, "Makes no sense to have a batch_size bigger than number of edits..."
        hparams.num_edits = args.num_edits
    return hparams

def calculate_model_name(args, hparams):
    if args.perform_lora:
        # it's basically lora ft
        name = f"{args.model}_LoRA_FT_{args.data_type}"
    elif args.no_crisp:
        # it's basically ft
        name = f"{args.model}_FT_{args.data_type}"
    else:
        name = f"{args.model}_{hparams.alg_name}_{args.data_type}_{args.energy_threshold}_{hparams.mom2_n_samples}"
    
    if args.sequential_edit:
        name += f"_sequential_{args.num_edits}"
    
    if hparams.recalculate_cache:
        name += f"_recalc_cache_{args.recalculate_weight_threshold}_edit_sample_{hparams.edit_n_samples}"
    if args.sequential_edit:
        name += f"_edit_cache_{hparams.edit_cache_style}"
        
    return name.replace('.', '_')

if __name__ == "__main__":
    args = get_arguments()
    requests = prepare_requests_from_data_type(args.data_type)
    requests = setup_requests_for_safeedit(requests)
    hparams = get_hparams(args)

    
    save_model_name = calculate_model_name(args, hparams)
    print(f"Model will be saved to BASE_DIR/{save_model_name}")
    wandb.init(project=args.wandb_project, name=save_model_name, config=vars(hparams), mode="disabled" if args.no_wandb else "online")

    MODEL_NAME = hparams.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR, device_map='cuda', torch_dtype=torch.float32 if args.no_crisp else torch.bfloat16)
    device = model.device

    # set appropriate padding token
    model, tokenizer = update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams)
    
    print_time("Begin FT Time")
    if args.sequential_edit:
        edited_model = execute_ft_sequential(model, tokenizer, requests, hparams)
    else:
        edited_model = execute_ft(model, tokenizer, requests, hparams)
    print_time("End FT Time")

    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)