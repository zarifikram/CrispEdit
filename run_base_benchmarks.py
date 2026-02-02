import os
import wandb
import random
import torch
import numpy as np
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from utils import print_time, save_clean_results
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor.util import HyperParams


SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def get_model_and_tokenizer_from_dir(edited_model_dir_local):
    PREFIX_DIR = os.getenv("HF_CACHE_DIR")
    edited_model_dir = PREFIX_DIR + edited_model_dir_local
    tokenizer = AutoTokenizer.from_pretrained(edited_model_dir)
    model = AutoModelForCausalLM.from_pretrained(edited_model_dir, device_map='auto')
    return model, tokenizer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model_dir', required=True, type=str, default=None, help='Path to edited model for evaluation.')
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki', 'safeedit_train', 'safeedit_test'])
    parser.add_argument('--eval_num', required=False, type=int, default=200, help='Number of evaluation instances to use for capability. Default uses 200.')
    parser.add_argument('--alg_name', required=True, type=str, default='ft_edit', help='Name of the editing algorithm used.')
    parser.add_argument('--model_name', required=True, type=str, default='gpt2-xl', help='Name of the base model used.')
    parser.add_argument('--wandb_project', type=str, default='CrispEdit_EVAL', help='WandB project name.')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID for resuming runs.')
    args = parser.parse_args()
    return args

def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.model_name = args.model_name
    return hparams

if __name__ == "__main__":
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)
    # device expects the device number only
    device = model.device.index

    run_name = args.edited_model_dir
    run = wandb.init(project=args.wandb_project, name=run_name, config=vars(hparams), resume=args.wandb_run_id if not args.wandb_run_id else "must", id=args.wandb_run_id)

    # before evaluation, always make sure tokenizer padding side is correct
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    lm_wrapper = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
    )

    print_time("Begin Capability Eval Time")

    tasks_with_config = {
        "ifeval":         {"shots": 0, "batch": "auto"},
        "truthfulqa_mc2": {"shots": 0, "batch": "auto"},
        "mmlu":           {"shots": 5, "batch": "auto"}, 

        # HEAVY tasks (High shots + CoT): STRICT batch limit needed
        "gsm8k_cot":      {"shots": 8,  "batch": 2}, # CoT generates long outputs, eats memory, add "batch": 1 (or whatever else) if OOM
        "arc_challenge":  {"shots": 25, "batch": 1}  # 25-shot context is massive, add "batch": 1 if OOM
    }
    results = {"results": {}}

    for task_name, config in tasks_with_config.items():
        print(f"Running {task_name} (Shots: {config['shots']}, Batch: {config['batch']})...")
        
        _results = simple_evaluate(
            model=lm_wrapper,
            tasks=[task_name],
            limit=args.eval_num,
            num_fewshot=config['shots'],
            batch_size=config['batch'],
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )
        
        if "results" in _results:
            results["results"].update(_results["results"])
            wandb.log(_results["results"])
        else:
            raise ValueError(f"No results found for task {task_name}")

    save_clean_results(results, f"./logs/{run_name}")
    artifact = wandb.Artifact('raw_results', type='dataset')
    artifact.add_file(f'./logs/{run_name}/capability.json')
    run.log_artifact(artifact)
    print_time("End Capability Eval Time")
