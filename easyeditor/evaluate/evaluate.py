"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_prediction_acc_real,
    is_probability_higher,
    test_safety_acc,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    OOD_PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality_safety(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    if isinstance(model,LORA):
        model=model.model
    target_safe, target_unsafe = (
        record[x] for x in ["target_safe", "target_unsafe"]
    )

    prompts = record["prompt"]
    prompts_question_only, prompts_diff_question, prompts_diff_attack, prompts_diff_attack_and_question = (
        record['generalization_test'][x] for x in ["test input of only harmful question", "test input of other attack prompt input", "test input of other question input", "test input of other questions and attack prompts"]
    )
    ret = compute_safety_rewrite_quality(model, model_name, hparams, tok, prompts, target_safe, target_unsafe, device=device, key='safety_rewrite', eval_metric=eval_metric)
    return ret
    
def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD":
        locality_function = compute_locality_quality
    elif 'counterfact' not in hparams.data_type:
        locality_function = compute_locality_quality
    else:
        locality_function = compute_locality_quality_counterfact

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                locality_function(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], record['target_new'], record, device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_safety_rewrite_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_safe: str,
    target_unsafe: str,
    device,
    key: str = 'safe_rewrite',
    eval_metric: str = 'token_em',
) -> typing.Dict:
    assert hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD", "Safety evaluation only supports WILD evaluation currently (it does not make sense otherwise!)"
    safety_acc, gen_content = test_safety_acc(model, tok, hparams, prompt, target_safe, target_unsafe, device)
    ret = {
        f"{key}_safety_acc": safety_acc,
        f"{key}_gen_content": gen_content,
    }
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    # using WILD evaluation: autoregressive decoding, natural stop criteria, LLM-as-a-Judge
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD":
        acc, gen_content = test_prediction_acc_real(model, tok, hparams, prompt, target_new, device, locality=False)
        ret = {
            f"{key}_acc": acc,
            f"{key}_gen_content": gen_content
        }
    else:  # synthetic evaluation 
        if eval_metric == 'ppl':
            ppl = PPL(model, tok, prompt, target_new, device)
            ret = {
                f"{key}_ppl": ppl
            }
        elif eval_metric == 'ood_ppl':
            ans = OOD_PPL(model, tok, prompt, target_new, device)
            ret = {
                f"ood_acc": ans
            }
        elif hparams.alg_name=="GRACE":
            # ppl = PPL(model, tok, prompt, target_new, device)
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
            f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
            ret = {
                f"{key}_acc": acc,
                # f"{key}_PPL": ppl,
                f"{key}_F1":f1     
            }        
        else:
            if 't5' in model_name.lower():
                acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
            else:
                acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
            ret = {
                f"{key}_acc": acc
            }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    locality_false_ground_truth: typing.Union[str, List[str]], # unused
    record, # unused
    device,
) -> typing.Dict:

    # using WILD evaluation
    gen_content = None
    if hasattr(hparams, 'evaluation_type') and hparams.evaluation_type == "WILD":
        acc, gen_content = test_prediction_acc_real(model, tok, hparams, prompt, locality_ground_truth, device, locality=False)
    else:  # synthetic evaluation 
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=False, vanilla_generation=hparams.alg_name=='GRACE')

        if type(acc) is not list:
            acc = [acc,]
    ret = {
        f"{locality_key}_acc": acc,
        f"{locality_key}_gen_content": gen_content
        
    }
    return ret

def compute_locality_quality_counterfact(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    locality_false_ground_truth: typing.Union[str, List[str]],
    record,
    device,
) -> typing.Dict:

    assert not hasattr(hparams, 'evaluation_type') or hparams.evaluation_type != "WILD", "We are doing synthetic evaluation for CounterFact locality! Change the code if you want WILD evaluation."
    
    ret_counterfact = compute_rewrite_quality_counterfact(model, tok, record)

    ret = {
        f"{locality_key}_acc": sum([probs['target_true'] < probs['target_new'] for probs in ret_counterfact["neighborhood_prompts_probs"]]) / len(ret_counterfact["neighborhood_prompts_probs"]),
        f"{locality_key}_gen_content": None,
    }
    return ret


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    # borrowed from AlphaEdit. Use with caution.

    # First, unpack rewrite evaluation record.
    target_new, target_true = (
        record["target_new"], record["locality"]["neighborhood"]["ground_truth"]
    )
    rewrite_prompts = [record["prompt"]]
    paraphrase_prompts = [record["rephrase_prompt"]] if type(record["rephrase_prompt"]) is str else record["rephrase_prompt"]
    neighborhood_prompts = [record["locality"]["neighborhood"]["prompt"]] if type(record["locality"]["neighborhood"]["prompt"]) is str else record["locality"]["neighborhood"]["prompt"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new,
        target_true,
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    return ret

def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])

    if 'llama' in model.config._name_or_path.lower():
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [lengths -1 for lengths in prefix_lens]

    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if 'llama' in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct



def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
    else:
        portability_correct = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
    ret = {
        f"rewrite_acc": [edit_acc]
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"{x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"{record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"{x_prefix}{record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc

    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100, vanilla_generation=False)
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
