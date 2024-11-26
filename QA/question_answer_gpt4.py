import os
import numpy as np
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from llama_recipes.utils.inference_utils import dump_jsonl
import openai
from time import sleep
from openai import OpenAI

def get_messages(question, context):
    system = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and truthful answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please[B give a full and complete answer for the question."
    messages = [
        {"role": "system", "content": system + "\n\n" + "Context:\n" + context + "\n\n"},
        {"role": "user", "content": "Question:\n" + instruction + " " + question + "\n\nAssistant:"}
    ]

    return messages


def generate_response(model, tokenizer, prompt, terminators, greedy):
    tokenized_prompt = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(
        model.device)
    response_tok_probs = []
    if greedy:
        outputs = model.generate(input_ids=tokenized_prompt.input_ids,
                                 attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, do_sample=False,
                                 eos_token_id=terminators, return_dict_in_generate=True, output_scores=True)
    else:
        outputs = model.generate(input_ids=tokenized_prompt.input_ids,
                                 attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, do_sample=True,
                                 top_k=50, temperature=0.7, num_return_sequences=10,
                                 eos_token_id=terminators, return_dict_in_generate=True, output_scores=True)
    generation = \
        tokenizer.batch_decode(outputs.sequences,
                               skip_special_tokens=True, clean_up_tokenization_spaces=False)

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True)
    generated_tokens = outputs.sequences[:, tokenized_prompt.input_ids.shape[-1]:]
    for i in range(len(generation)):
        response = generation[i].replace(prompt, "")
        tok_probs = []
        for tok, score in zip(generated_tokens[i], transition_scores[i]):
            tok_probs.append((tokenizer.decode(tok), float(round(np.exp(score.cpu().numpy()), 2))))
        response_tok_probs.append((response, tok_probs))

    return response_tok_probs


def run(question_retrieval_file, top_k, combined, model='gpt-4o-mini', temperature=0, max_tokens=128):
    client = OpenAI()
    combined_text = 'combined' if combined else 'separate'
    question_retrieval = list(jsonlines.open(question_retrieval_file))
    mode = 'train' if 'train' in question_retrieval_file else 'dev'
    output_file = f'./data/dev/qa_generated_questions_{combined_text}_{mode}_top{top_k}_{model}_round2.jsonl'

    error_idx = []
    for idx, element in enumerate(question_retrieval[202:203]):
        claim = element['claim']
        questions = element['generated_questions']
        top_sentence_chunks = element['sentence_chunks']
        question_chunks_pairs = list(zip(questions, top_sentence_chunks))
        responses = []
        for pair in question_chunks_pairs:
            question, sentence_chunks = pair

            sentence_chunks = sentence_chunks[:top_k]
            if combined:
                sentence_chunks = sentence_chunks[::-1]
                chunk = '\n\n'.join(sentence_chunks)
                messages = get_messages(question, chunk)
                #print(messages)
                try:
                    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                              max_tokens=max_tokens)

                    generated_text = response.choices[0].message.content
                    print(generated_text)
                except Exception as e:
                    print(e)
                    error_idx.append(idx)
                    generated_text = ''
                responses.append(generated_text)
        element['greedy_responses'] = responses
        #dump_jsonl([element], output_file, append=True)
        sleep(5)

    print('#####################')


if __name__ == '__main__':
    model = None
    run('./data/dev/sorted_claim_generated_question_512_dev_round2_all.jsonl',
        top_k=3, combined=True)
