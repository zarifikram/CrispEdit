from datetime import datetime
import json
from easyeditor.editors.utils import _prepare_requests, _prepare_requests_safeedit

from dotenv import load_dotenv
import os

load_dotenv()
BASE_DIR = os.getenv("HF_CACHE_DIR")

def save_clean_results(results, logs_dir):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    results_filename = 'capability.json'
    output_file = os.path.join(logs_dir, results_filename)

    with open(output_file, "w") as f:
        json.dump(results["results"], f, indent=4)
    
    print(f"Clean results saved to {output_file}")
    print("Preview:", results["results"])

def print_time(process_name):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f'{process_name}: {formatted_time}')

def save_model_and_tokenizer(model, tokenizer, local_directory):
    save_directory = BASE_DIR + local_directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
def prepare_requests_from_data_type(data_type):
    if "safeedit" not in data_type:
        prompts, rephrase_prompts, subject, target_new, locality_inputs, ground_truth = prepare_prompts_from_data_type(data_type)
        requests = _prepare_requests(prompts, target_new, ground_truth, None, rephrase_prompts, locality_inputs)
    else:
        prompts, target_safe, target_unsafe, gen_prompts, questions = prepare_prompts_from_data_type_safeedit(data_type)
        requests = _prepare_requests_safeedit(prompts, target_safe, target_unsafe, gen_prompts, questions)
    return requests

def prepare_prompts_from_data_type(data_type):
    data_file = {
        "zsre163k": "zsre_mend_163k",
        "zsre10k": "zsre_mend_10k",
        "zsre": "zsre_mend_3k",
        "counterfact": "counterfact-edit_3k",
        "wiki": "wiki_big_edit_3k",
        "multi_counterfact": "multi_counterfact"
    }[data_type]
    data = json.load(open(f"./data/{data_file}.json", 'r', encoding='utf-8'))

    if data_type == 'counterfact':
        prompts = [d['prompt'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase_prompt'] for d in data]
        target_new = [d['target_new'] for d in data]
        locality_prompts = [d['locality_prompt'] for d in data]
        locality_ans = [d['locality_ground_truth'] for d in data]
    elif data_type == 'zsre' or data_type == 'zsre10k' or data_type == 'zsre163k':
        prompts = [d['src'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase'] for d in data]
        target_new = [d['alt'] for d in data]
        locality_prompts = [d['loc'][13:].capitalize() + "?" for d in data]
        locality_ans = [d['loc_ans'] for d in data]
    elif data_type == 'qaedit':
        prompts = [d['prompt'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase'] for d in data]
        target_new = [d['target'] for d in data]
        locality_prompts = [d["locality"][0]["loc"] for d in data]
        locality_ans = [d["locality"][0]["loc_ans"] for d in data]
    elif data_type == 'wiki':
        prompts = [d['prompt'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase_prompt'] for d in data]
        target_new = [d['target_new'] for d in data]
        locality_prompts = [d["locality_prompt"] for d in data]
        locality_ans = [d["locality_ground_truth"] for d in data]
    elif data_type == "multi_counterfact":
        prompts = [d['requested_rewrite']['prompt'].format(d['requested_rewrite']['subject']) for d in data]
        subject = [d['requested_rewrite']['subject'] for d in data]
        rephrase_prompts = [d['paraphrase_prompts'] for d in data]
        target_new = [d['requested_rewrite']['target_new']['str'] for d in data]
        locality_prompts = [d['neighborhood_prompts'] for d in data]
        locality_ans = [d['requested_rewrite']['target_true']['str'] for d in data]
    else:
        raise NotImplementedError(f"Data type {data_type} not supported.")

    ground_truth = ['<|endoftext|>' for d in data]  
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    return prompts, rephrase_prompts, subject, target_new, locality_inputs, ground_truth

def prepare_prompts_from_data_type_safeedit(data_type):
    data_file = {
        "safeedit_train": "SafeEdit_train",
        "safeedit_test": "SafeEdit_test",
    }[data_type]
    data = json.load(open(f"./data/{data_file}.json", 'r', encoding='utf-8'))

    prompts = [d['adversarial prompt'] for d in data]
    target_safe = [d['safe generation'] for d in data]
    target_unsafe = [d['unsafe generation'] for d in data]
    gen_prompts = [d['generalization test'] for d in data]
    questions = [d['question'] for d in data]

    return prompts, target_safe, target_unsafe, gen_prompts, questions

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