import numpy as np
import jsonlines
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from llama_recipes.utils.inference_utils import dump_jsonl


def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(
        ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in
         messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input


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


def run(model, tokenizer, question_retrieval_file, mode, top_k, combined):
    combined_text = 'combined' if combined else 'separate'
    question_retrieval = list(jsonlines.open(question_retrieval_file))

    output_file = f'./data/dev/qa_generated_questions_{combined_text}_{mode}_top{top_k}_greedy_sampling_all_rounds.jsonl'

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for element in tqdm(question_retrieval):
        claim = element['claim']
        questions = element['generated_questions']
        sentence_chunks = element['sentence_chunks']
        question_chunks_pairs = list(zip(questions, sentence_chunks))
        responses = {'greedy': [], 'sampling': []}
        for pair in question_chunks_pairs:
            question, sentence_chunks = pair
            messages = [
                {"role": "user", "content": question}
            ]
            sentence_chunks = sentence_chunks[:top_k]
            if combined:
                sentence_chunks = sentence_chunks[::-1]
                chunk = '\n\n'.join(sentence_chunks)
                formatted_input = get_formatted_input(messages, chunk)

                greedy_generation_tok_prob = generate_response(model, tokenizer, formatted_input, terminators,
                                                               greedy=True)

                sampling_generation_tok_prob = generate_response(model, tokenizer, formatted_input, terminators,
                                                                 greedy=False)
                responses['greedy'].append(greedy_generation_tok_prob)
                responses['sampling'].append(sampling_generation_tok_prob)

            else:
                question_response = []
                for chunk in sentence_chunks:
                    formatted_input = get_formatted_input(messages, chunk)
                    greedy_generation_tok_prob = generate_response(model, tokenizer, formatted_input, terminators,
                                                                   greedy=True)
                    responses['greedy'].append(greedy_generation_tok_prob)
                    responses['sampling'].append([])
                    # print(response)
                    # question_response.append((response, token_prob))
                # responses.append(question_response)

        element['greedy_responses'] = responses['greedy']
        element['sampling_responses'] = responses['sampling']
        element['sentence_chunks'] = [item[:top_k] for item in element['sentence_chunks']]
        element['scores'] = [item[:top_k] for item in element['scores']]
        element['urls'] = [item[:top_k] for item in element['urls']]

        dump_jsonl([element], output_file, append=True)

        print('#####################')


if __name__ == '__main__':
    # /hkfs/work/workspace_haic/scratch/vl8701-llm2
    model_id = '/hkfs/work/workspace_haic/scratch/vl8701-llm/Llama3-ChatQA-2-70B'
    # LORA_WEIGHTS = '/hkfs/work/workspace_haic/scratch/vl8701-llm2/checkpoint/averitec/question_answering/few_nei/epoch_2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 128001
    tokenizer.eos_token_id = 128001
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    # model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.bfloat16, device_map='balanced')
    # model = None
    run(model, tokenizer, './data/dev/sorted_claim_generated_question_512_dev_all_rounds.jsonl', mode='dev',
        top_k=3, combined=True)
