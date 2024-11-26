import json
import os
import torch
import jsonlines
import argparse
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import sleep
from tqdm import tqdm
from llama_recipes.utils.inference_utils import get_logger, dump_jsonl
from preprocess.preprocessing import process_averitec_date


def get_prompt(claim, claimer, claim_date, mode='question_generation'):
    if mode == 'question_generation':
        prompt = f"""<|begin_of_text|> You are a fact-checker and your task is to generate critical questions for verifying the following claim.\nClaim date: {claim_date}\nClaimer: {claimer}\nClaim: {claim}\nQuestions: """
    else:
        assert NotImplementedError
    return prompt


def run(model, tokenizer, mode, model_category, model_size, dataset, test_file, appendix, sampling=False):
    if sampling:
        description = f'{model_category}_{model_size}_{mode}_{dataset}_sampling_{appendix}'
    else:
        description = f'{model_category}_{model_size}_{mode}_{dataset}_greedy_{appendix}'

    logger = get_logger('./logs/', description)
    output_path = f'./data/question_generation/{description}.jsonl'
    with open(test_file) as file:
        data = json.load(file)

    for element in tqdm(data):
        claim = element['claim']
        claimer = element['speaker'] if element['speaker'] is not None else 'Unknown'
        claim_date = process_averitec_date(element['claim_date'])
        prompt = get_prompt(claim=claim, claimer=claimer, claim_date=claim_date, mode=mode)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        if sampling:
            output = model.generate(input_ids=inputs.input_ids.to('cuda'), max_new_tokens=300, do_sample=True, top_k=50,
                                    temperature=0.7, num_return_sequences=10, return_dict_in_generate=True,
                                    output_scores=True)
            generation = tokenizer.batch_decode(output.sequences, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            for element in generation:
                print(element)
            transition_scores = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True)
            generated_tokens = output.sequences[:, input_length:]
            token_probabilities_all = []
            for i in range(generated_tokens.shape[0]):
                token_probabilities = []
                for tok, score in zip(generated_tokens[i], transition_scores[i]):
                    token_probabilities.append((tokenizer.decode(tok), float(round(np.exp(score.cpu().numpy()), 2))))
                token_probabilities_all.append(token_probabilities)
            dump_jsonl(
                [{'claim': claim, 'info': element, 'generation': generation, 'tok_prob': token_probabilities_all}],
                output_path, append=True)



        else:
            outputs = model.generate(input_ids=inputs.input_ids.to('cuda'), max_new_tokens=300, do_sample=False,
                                     return_dict_in_generate=True, output_scores=True)
            generation = \
                tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                    0]
            print(generation)
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True)

            generated_tokens = outputs.sequences[:, input_length:]
            token_probabilities = []
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                token_probabilities.append((tokenizer.decode(tok), float(round(np.exp(score.cpu().numpy()), 2))))
            dump_jsonl([{'claim': claim, 'info': element, 'generation': generation, 'tok_prob': token_probabilities}],
                       output_path, append=True)

        logger.info("\n==================================\n")


if __name__ == "__main__":
    data_info = {"dataset": "dev", "test_file": "./data/dev.json"}
    LORA_WEIGHTS = '/hkfs/work/workspace_haic/scratch/vl8701-llm2/checkpoint/averitec/question_generation/sub_included/epoch_9'
    BASE_MODEL = '/hkfs/work/workspace_haic/scratch/vl8701-llm2/llama3/Meta-Llama-3-70B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = 128001
    tokenizer.eos_token_id = 128001
    # model = None
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map='balanced')
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.bfloat16, device_map='balanced')
    run(model=model, tokenizer=tokenizer, mode='question_generation', model_category='llama3-instruct-lora',
        model_size='70B', dataset=data_info['dataset'], test_file=data_info['test_file'],
        appendix=f'sub_included', sampling=False)
