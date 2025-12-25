from datetime import datetime
import json
from easyeditor.editors.utils import _prepare_requests

BASE_DIR = "/data0/zikram/huggingface/hub/" # TO-DO: Change it

def print_time(process_name):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d %H:%M:%S")
    print(f'{process_name}: {formatted_time}')

def save_model_and_tokenizer(model, tokenizer, local_directory):
    save_directory = BASE_DIR + local_directory
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
def prepare_requests_from_data_type(data_type):
    prompts, rephrase_prompts, subject, target_new, locality_inputs, ground_truth = prepare_prompts_from_data_type(data_type)
    requests = _prepare_requests(prompts, target_new, ground_truth, None, rephrase_prompts, locality_inputs)

    return requests

def prepare_prompts_from_data_type(data_type):
    data_file = {"zsre": "zsre_mend_eval_3k",
                "counterfact": "counterfact-edit_3k",
                "wiki": "wiki_big_edit_3k"}[data_type]
    data = json.load(open(f"./data/{data_file}.json", 'r', encoding='utf-8'))

    if data_type == 'counterfact':
        prompts = [d['prompt'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase_prompt'] for d in data]
        target_new = [d['target_new'] for d in data]
        locality_prompts = [d['locality_prompt'] for d in data]
        locality_ans = [d['locality_ground_truth'] for d in data]
    elif data_type == 'zsre':
        prompts = [d['src'] for d in data]
        subject = [d['subject'] for d in data]
        rephrase_prompts = [d['rephrase'] for d in data]
        target_new = [d['alt'] for d in data]
        locality_prompts = [d['loc'] for d in data]
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