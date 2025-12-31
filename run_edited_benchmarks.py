import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
# os.environ['HF_ENDPOINT'] = os.getenv("HF_ENDPOINT")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from utils import print_time, prepare_requests_from_data_type, save_clean_results
from easyeditor.editors.utils import summary_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from easyeditor.evaluate.evaluate import compute_edit_quality
import random
import torch
from tqdm import tqdm
import wandb
from easyeditor.util import HyperParams
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


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
    parser.add_argument('--data_type', required=True, type=str, default='zsre', choices=['zsre', 'counterfact', 'wiki'])
    parser.add_argument('--eval_num', required=False, type=int, default=3000, help='Number of evaluation instances to use. Default uses all.')
    parser.add_argument('--max_length', required=False, type=int, default=40, help='Maximum length of the generated sequences.')
    parser.add_argument('--context_type', required=True, type=str, default='qa_inst', choices=['qa_inst', 'chat_temp', 'no_context'], help='Type of context to use for evaluation.')
    parser.add_argument('--alg_name', required=True, type=str, default='ft_edit', help='Name of the editing algorithm used.')
    parser.add_argument('--model_name', required=True, type=str, default='gpt2-xl', help='Name of the base model used.')
    parser.add_argument('--evaluation_criteria', required=True, type=str, default='exact_match', choices=['exact_match', 'llm_judge'], help='Evaluation criteria to use.')  
    parser.add_argument('--wandb_project', type=str, default='JIGSAW_EVAL', help='WandB project name.')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID for resuming runs.')
    args = parser.parse_args()
    return args

def build_hparams_from_args(args):
    hparams = HyperParams()
    hparams.alg_name = args.alg_name
    hparams.context_type = args.context_type
    hparams.max_length = args.max_length
    hparams.api_key = API_KEY
    hparams.evaluation_type = "WILD"
    hparams.model_name = args.model_name
    hparams.evaluation_criteria = args.evaluation_criteria
    return hparams

if __name__ == "__main__":
    args = get_arguments()
    hparams = build_hparams_from_args(args)
    requests = prepare_requests_from_data_type(args.data_type)
    model, tokenizer = get_model_and_tokenizer_from_dir(args.edited_model_dir)
    # device expects the device number only
    device = model.device.index

    run_name = f"{hparams.alg_name}_{args.data_type}_{hparams.model_name}"
    # if there is a run ID, use it to resume
    run = wandb.init(project=args.wandb_project, name=run_name, config=vars(hparams), resume=args.wandb_run_id if args.wandb_run_id else "must", id=args.wandb_run_id)

    # before evaluation, always make sure tokenizer padding side is correct
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    print_time("Begin Post Edit Eval Time")
    requests = random.sample(requests, len(requests))
    if args.eval_num is not None:
        requests = requests[:args.eval_num]

    all_metrics = []
    for i, request in enumerate(tqdm(requests)):
        metrics = {
            'case_id': i,
            "requested_rewrite": request,
            "pre": {},
            "post": compute_edit_quality(model, hparams.model_name, hparams, tokenizer, request, device),
        }
        all_metrics.append(metrics)

        print(f"{i} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[i]}")

    summary_metrics(all_metrics, f"./logs/{run_name}")

    print_time("End Post Edit Eval Time") 
    artifact = wandb.Artifact('mean_metrics', type='dataset')
    artifact.add_file(f'./logs/{run_name}/mean_metrics.json')
    run.log_artifact(artifact)