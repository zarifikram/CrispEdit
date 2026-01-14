import json
import argparse
import torch
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_DIR")
os.environ['HF_ENDPOINT'] = os.getenv("HF_ENDPOINT")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import wandb
from utils import prepare_prompts_from_data_type, save_model_and_tokenizer
import random

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

from easyeditor import (
    FTHyperParams,
    MENDHyperParams,
    UltraEditHyperParams,
    ROMEHyperParams,
    R_ROMEHyperParams,
    MEMITHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    AlphaEditHyperParams,
    IKEHyperParams,
    MELOHyperParams,
    LoRAHyperParams,
    BaseEditor,
)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki'])
    parser.add_argument('--editing_method', required=True, type=str, choices=['FT', 'MEND', 'ROME', 'R-ROME', 'MEMIT', 'GRACE', 'WISE', 'AlphaEdit', 'IKE', 'MELO', 'LoRA', 'UltraEdit'])
    parser.add_argument('--batch_size', required=True, type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--eval_every', required=True, type=int, default=512, help='Evaluation frequency.')
    parser.add_argument('--sequential_edit', default='True', type=str)
    parser.add_argument('--batch_edit', default='False', type=str)
    parser.add_argument('--wandb_project', type=str, default='JIGSAW', help='WandB project name.')
    args = parser.parse_args()
    return args

def get_hparams_and_editor(args):
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'UltraEdit':
        editing_hparams = UltraEditHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'MELO':
        editing_hparams = MELOHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(f"./hparams/{args.editing_method}/{args.model}")
    hparams.batch_size = args.batch_size
    hparams.model_parallel = True
    editor = BaseEditor.from_hparams(hparams)
    return hparams, editor

if __name__ == "__main__":
    args = get_arguments()
    prompts, rephrase_prompts, subject, target_new, locality_inputs, ground_truth = prepare_prompts_from_data_type(args.data_type)
    hparams, editor = get_hparams_and_editor(args)
    save_model_name = f"{args.model}_{args.editing_method}_{args.data_type}"
    print(f"Model will be saved to BASE_DIR/{save_model_name}")
    wandb.init(project=args.wandb_project, name=save_model_name, config=vars(hparams))

    if args.sequential_edit == "True" or args.sequential_edit == "true":
        sequential_edit = True
    else:
        sequential_edit = False

    if args.batch_edit == "True" or args.batch_edit == "true":
        batch_edit = True
    else:
        batch_edit = False

    if batch_edit:
        edited_model, tokenizer = editor.batch_edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            subject=subject,
            target_new=target_new,
            locality_inputs=locality_inputs,
            eval_every=args.eval_every,
        )
    else:
        metrics, edited_model, _, tokenizer = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            subject=subject,
            target_new=target_new,
            locality_inputs=locality_inputs,
            sequential_edit=sequential_edit,
            eval_every=args.eval_every,
        )

    save_model_and_tokenizer(edited_model, tokenizer, save_model_name)