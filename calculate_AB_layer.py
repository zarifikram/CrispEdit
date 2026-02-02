from easyeditor.models.crispedit.utils import calculate_projection_caches
from easyeditor.models.crispedit.CrispEdit_hparams import CrispEditHyperParams
from dotenv import load_dotenv
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import wandb

import argparse
import torch

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# get model name, tokenizer, and layer number from args

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--cache_sample_num', type=int, default=1000, help='Number of samples to use for caching projection matrices.')
    parser.add_argument("--layer", required=True, type=int, help="Layer number to compute A, B for.")
    parser.add_argument('--wandb_project', type=str, default='CrispEdit_CACHE_CALC', help='WandB project name.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_arguments()
    hparams = CrispEditHyperParams.from_hparams(f"./hparams/CrispEdit/{args.model}")
    
    hparams.mom2_n_samples = args.cache_sample_num
    hparams.layers = [args.layer]

    hparams.energy_threshold = 0.9

    wand_name = f"calc_AB_{hparams.model_name.replace('/', '_')}_layer{args.layer}_samples{args.cache_sample_num}"
    wandb.init(project=args.wandb_project, name=wand_name, config=vars(hparams))

    MODEL_NAME = hparams.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR, device_map='auto')
    device = model.device

    # set appropriate padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    layer = int(args.layer)  # specify the layer number you want to compute A, B for

    
    device = model.device
    
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    weight_to_projection_cache = calculate_projection_caches(model, tokenizer, hparams, force_recompute=False)
    print(f"Projection caches computed for model {MODEL_NAME} at layer {layer}.")
    # print weight_to_projection_cache keys
    for key in weight_to_projection_cache.keys():
        # print(f"Key: {key}, value type: {type(weight_to_projection_cache[key])}")
        P_cache = weight_to_projection_cache[key]
        Ua = P_cache["Ua"]
        Ub = P_cache["Ub"]
        M = P_cache["M"]
        print(f"Ua shape: {Ua.shape}, Ub shape: {Ub.shape}, M shape: {M.shape}")
