import itertools
import json
import jsonlines
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from llama_recipes.utils.inference_utils import dump_jsonl
from tqdm import tqdm
from fairseq.data.data_utils import collate_tokens
from collections import Counter


def combine_file(input_files, reference_file, output_file):
    with open(reference_file) as file:
        reference_data = json.load(file)
    generation_data = []
    for input_file in input_files:
        generation_data += list(jsonlines.open(input_file))
    for idx, element in enumerate(generation_data):
        if element['claim'] == reference_data[idx]['claim']:
            dump_jsonl([element], output_file, append=True)


def get_generated_question(generation_file, sampling=False):
    data = list(jsonlines.open(generation_file))
    for idx, element in enumerate(data):
        claim = element['claim']
        original_questions = [item['question'] for item in element['info']['questions']]
        generation = element['generation'].split('Questions:')[1].strip()

        generated_questions = generation.split('?')
        generated_questions = [question for question in generated_questions if question != '']

        generated_questions = list(set([question.strip() + '?' for question in generated_questions]))
        print(f'{idx}: {claim}')
        if len(generated_questions) > 0:
            print(f'---generated question---')
            for question in generated_questions:
                print(question)
            print('\n')
            print(f'---original question---')
            for question in original_questions:
                print(question)
        else:
            print(f'------------------')
        print('##############')


def get_text_from_url(tokenizer, claim_id, top_urls, mode):
    search_data = list(jsonlines.open(
        f"/hkfs/work/workspace_haic/scratch/vl8701-llm2/checkpoint/averitec/data_store/knowledge_store/test/{claim_id}.json"))
    total_urls = [item['url'] for item in search_data]
    top_url_idxes = [total_urls.index(url) for url in top_urls]
    for idx in top_url_idxes:
        pass


def check_answer_similarity(encoding_model, greedy_answer, sampled_answers, threshold):
    selected = [greedy_answer]
    for element in sampled_answers:
        pairs = [(item, element) for item in selected]
        similarities = []
        for pair in pairs:
            embeddings = encoding_model.encode([pair[0], pair[1]])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            similarities.append(similarity)
        if all(similarity < threshold for similarity in similarities):
            selected.append(element)
    return selected


def select_answer(input_file, output_file):
    encoding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    generation_data = list(jsonlines.open(input_file))
    idx = 0
    error_idx = []
    for element in tqdm(generation_data):
        try:
            greedy_answers = [item[0][0] for item in element['greedy_responses']]
            sampled_answers = []
            for item in element['sampling_responses']:
                sampled_answers.append([part[0] for part in item])
            assert len(greedy_answers) == len(sampled_answers)

            selected_responses = []

            for i in range(len(greedy_answers)):
                selected = check_answer_similarity(encoding_model, greedy_answers[i], sampled_answers[i], threshold=0.6)
                selected_responses.append(selected)
            element['selected_responses'] = selected_responses
            dump_jsonl([element], output_file, append=True)
        except:
            error_idx.append(idx)
        idx += 1
    print(error_idx)


def rank_question_similarity(encoding_model, greedy_questions, sampling_questions):
    greedy_question = " ".join(greedy_questions)
    similarities = []
    for item in sampling_questions:
        sampled_question = " ".join(item)
        embeddings = encoding_model.encode([greedy_question, sampled_question])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
    similarities_rank = sorted(range(len(similarities)), key=lambda k: similarities[k])

    return similarities_rank


def rank_question_similarities_round2(encoding_model, existing_questions, rest_questions):
    similarities = []
    existing_questions = [" ".join(item) for item in existing_questions]
    rest_questions = [" ".join(item) for item in rest_questions]
    for q_rest in rest_questions:
        similarity = 0
        for q_exist in existing_questions:
            embeddings = encoding_model.encode([q_exist, q_rest])
            similarity += cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity / len(existing_questions))
    similarities_rank = sorted(range(len(similarities)), key=lambda k: similarities[k])
    return similarities_rank


def combine_greedy_sampling(greedy_file, sampling_file, output_file):
    greedy = list(jsonlines.open(greedy_file))
    sampling = list(jsonlines.open(sampling_file))
    for idx, element in enumerate(greedy):
        claim = element['claim']
        info = element['info']
        greedy_question = element['generated_questions']
        assert claim == sampling[idx]['claim']
        sampling_question = sampling[idx]['generated_questions']
        dump_jsonl([{'claim': claim, 'info': info, 'greedy_questions': greedy_question,
                     'sampling_questions': sampling_question}], output_file, append=True)


def evaluate_entailment(input_file):
    qa_entailment = list(jsonlines.open(input_file))
    c = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    d = {}
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device='cuda')
    for element in qa_entailment:
        generated_questions = element['generated_questions']
        supported = 0
        for i in range(len(generated_questions)):
            selected_responses = element['selected_responses'][i]
            statement_entailment = element['selected_statement_entailment'][i]
            entailment_labels = [item[0] for item in statement_entailment]
            num_s = sum([1 if label == 'Supports' else 0 for label in entailment_labels])
            c[num_s] += 1
            entailment_probs = [item[1] for item in statement_entailment]
            supported_responses = [selected_responses[i] for i in range(len(selected_responses)) if
                                   entailment_labels[i] == 'Supports']
            supported_probs = [entailment_probs[i][0][1] for i in range(len(selected_responses)) if
                               entailment_labels[i] == 'Supports']
            if len(supported_responses) > 0:
                question_response_pairs = [(generated_questions[i], response) for response in supported_responses]
                scores = model.predict(question_response_pairs, show_progress_bar=False).tolist()
                scores_ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                selected_response = supported_responses[scores_ranking[0]]

                supported += 1
        key = f'{len(generated_questions)}/{supported}'
        if key in d.keys():
            d[key] += 1
        else:
            d[key] = 1

    print(c)
    print({k: v for k, v in sorted(d.items(), key=lambda item: int(item[0].split('/')[0]))})
    total = 0
    for _, value in d.items():
        total += value
    print(total)


def truncation(prem_hypo_pair, nli_model):  # truncation strategy only first
    premise, hypothesis = prem_hypo_pair
    num_hypothesis_token = len(nli_model.encode(hypothesis).tolist())
    premise = nli_model.decode(nli_model.encode(premise)[:(1024 - num_hypothesis_token - 10)])

    return nli_model.encode(premise, hypothesis)


def check_statement_chunks_entailment(prem_hypo_pairs, nli_model, batch_size=8):
    softmax = torch.nn.Softmax(dim=1)
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in prem_hypo_pairs]
    max_length_exceed_indexes = [i for i in range(len(pair_tokens)) if len(pair_tokens[i]) > 1024]
    for index in max_length_exceed_indexes:
        truncated_tokens = truncation((prem_hypo_pairs[index][0], prem_hypo_pairs[index][1]), nli_model)
        pair_tokens[index] = truncated_tokens[:1024]
    batches = [pair_tokens[x:x + batch_size] for x in range(0, len(pair_tokens), batch_size)]
    labels = torch.tensor([]).to('cuda')
    probs = torch.tensor([]).to('cuda')

    for batch in batches:
        batch = collate_tokens(batch, pad_idx=1, pad_to_length=1024)
        with torch.no_grad():
            batch_logits = nli_model.predict('mnli', batch)
        batch_probs = softmax(batch_logits)
        batch_label = batch_probs.argmax(dim=1)
        labels = torch.cat((labels, batch_label), dim=0)
        probs = torch.cat((probs, batch_probs), dim=0)
    labels = np.reshape([int(labels[j].item()) for j in range(labels.shape[0])], (-1, 3)).tolist()
    entailment_probs = np.reshape([probs[j][2].item() for j in range(probs.shape[0])], (-1, 3)).tolist()
    return labels, entailment_probs


def select_best_response(statements, entailment_labels, entailment_probs):
    entailed_existing = False
    selected_statement_index = None
    sentence_chunk_index = None
    max_entailment_prob = float('-inf')
    for i in range(len(statements)):
        entailment_label = entailment_labels[i]
        entailment_prob = entailment_probs[i]
        if 2 in entailment_label:
            entailed_existing = True
            if max(entailment_prob) > max_entailment_prob:
                max_entailment_prob = max(entailment_prob)
                sentence_chunk_index = np.argmax(entailment_prob)
                selected_statement_index = i
    return entailed_existing, selected_statement_index, sentence_chunk_index


def response_entailment_check(input_file, output_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')

    qa_statements = list(jsonlines.open(input_file))
    for element in tqdm(qa_statements):
        generated_questions = element['generated_questions']
        sentence_chunks = element['sentence_chunks']
        selected_statements = element['selected_statements']
        selected_responses = element['selected_responses']
        checked_responses = []
        retained_responses = []
        for i in range(len(generated_questions)):
            chunks = sentence_chunks[i]
            assert len(chunks) == 3
            statements = selected_statements[i]
            responses = selected_responses[i]
            prem_hypo_pairs = []

            for statement in statements:
                for chunk in chunks:
                    prem_hypo_pairs.append((chunk, statement))

            entailment_labels, entailment_probs = check_statement_chunks_entailment(prem_hypo_pairs, nli_model,
                                                                                    batch_size=8)
            entailed_existing, selected_statement_index, sentence_chunk_index = select_best_response(statements,
                                                                                                     entailment_labels,
                                                                                                     entailment_probs)
            if entailed_existing:
                checked_responses.append([responses[selected_statement_index], int(sentence_chunk_index)])
                retained_responses.append(int(0))
            else:
                nei, response_idx = check_nei_response(responses)
                if nei:
                    nei_response = responses[response_idx]
                    checked_responses.append([nei_response, None])
                    retained_responses.append(int(1))
                else:
                    checked_responses.append([None, None])
                    retained_responses.append(int(0))
        assert len(checked_responses) == len(generated_questions)
        element['checked_responses'] = checked_responses
        element['retained_responses'] = retained_responses

        dump_jsonl([element], output_file, append=True)


def check_nei_response(responses):
    nei_response = False
    nei_response_index = None
    for response in responses:
        relo = response.lower()
        if ('sorry' in relo) or ('no information' in relo) or ('not provide' in relo) or (
                'no evidence' in relo) or ('not enough information' in relo):
            nei_response = True
            nei_response_index = responses.index(response)
            break
    return nei_response, nei_response_index


def check_qa_completeness(statement_check_file, greedy_sampling_questions_file, check_round):
    encoding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    statements_check = list(jsonlines.open(statement_check_file))
    greedy_sampling_questions = list(jsonlines.open(greedy_sampling_questions_file))
    d = {}
    further_check = {}
    for element in statements_check:

        claim_id = element['info']['claim_id']
        generated_questions = element['generated_questions']

        greedy_questions = greedy_sampling_questions[claim_id]['greedy_questions']

        sampling_questions = greedy_sampling_questions[claim_id]['sampling_questions']

        checked_response = element['checked_responses']
        retained_responses = element['retained_responses']
        response_complete = 0
        for item in checked_response:
            if item[0] is not None:
                response_complete += 1
        if response_complete > 0:
            if (response_complete / len(generated_questions)) > 0.8 and sum(
                    retained_responses) / response_complete < 0.3:
                pass
            else:
                if check_round == 1:
                    similarities_rank = rank_question_similarity(encoding_model, greedy_questions, sampling_questions)
                    further_check[claim_id] = similarities_rank[0]
                elif check_round == 2:
                    rest_questions = [item for item in sampling_questions if item != generated_questions]
                    similarities_rank = rank_question_similarities_round2(encoding_model,
                                                                          [greedy_questions, generated_questions],
                                                                          rest_questions)
                    further_check[claim_id] = sampling_questions.index(rest_questions[similarities_rank[0]])
                else:
                    further_check[claim_id] = int(0)

        else:
            if check_round == 1:
                similarities_rank = rank_question_similarity(encoding_model, greedy_questions, sampling_questions)
                further_check[claim_id] = similarities_rank[0]
            elif check_round == 2:
                rest_questions = [item for item in sampling_questions if item != generated_questions]
                similarities_rank = rank_question_similarities_round2(encoding_model,
                                                                      [greedy_questions, generated_questions],
                                                                      rest_questions)
                further_check[claim_id] = sampling_questions.index(rest_questions[similarities_rank[0]])
            else:
                further_check[claim_id] = int(0)
        key = f'{len(generated_questions)}/{response_complete}'
        if key in d.keys():
            d[key] += 1
        else:
            d[key] = 1
    with open('./data/test/further_check_round3.json', 'w') as file:
        json.dump(further_check, file)

    print(f'num further check: {len(further_check)}')
    print(further_check)

    # print({k: v for k, v in sorted(d.items(), key=lambda item: int(item[0].split('/')[0]))})


def combine_statements(input_files, question_file, index_file, output_file):
    statement_data = []
    for input_file in input_files:
        statement_data += list(jsonlines.open(input_file))
    questions = list(jsonlines.open(question_file))
    with open(index_file) as file:
        index_dict = json.load(file)
    for element in statement_data:
        claim_id = element['info']['claim_id']
        generated_questions = element['generated_questions']
        question_index = index_dict[str(claim_id)]
        selected_questions = questions[claim_id]['sampling_questions'][question_index]
        if selected_questions == generated_questions:
            dump_jsonl([element], output_file, append=True)


def check_difference(round1_file1, round1_file2):
    with open(round1_file1) as file:
        check1 = json.load(file)
    with open(round1_file2) as file:
        check2 = json.load(file)
    check1_keys = list(check1.keys())

    check2_keys = list(check2.keys())
    assert check1_keys == check2_keys
    assert check1 == check2


def collect_qa_url_pairs(element):
    processed_pairs = []
    generated_questions = element['generated_questions']
    urls = element['urls']

    checked_responses = element['checked_responses']
    checked_answers = [
        item[0] if item[0] is not None else 'No answer could be found.' for
        item in checked_responses]
    checked_urls_idxes = [item[1] for item in checked_responses]

    assert len(generated_questions) == len(checked_answers) == len(checked_urls_idxes)

    for i in range(len(generated_questions)):
        url = urls[i][checked_urls_idxes[i]] if checked_urls_idxes[i] is not None else ''
        if 'sorry' in checked_answers[i].lower():
            answer = 'No answer could be found.'
        else:
            answer = checked_answers[i]
        processed_pairs.append(
            {'question': generated_questions[i], 'answer': answer, 'url': url,
             'scraped_text': ''})
    return processed_pairs


def collect_qa_url_pairs_v2(element):
    processed_pairs = []
    generated_questions = element['generated_questions']
    urls = element['urls']

    checked_responses = element['checked_responses']
    checked_answers = [item[0] for item in checked_responses]
    checked_urls_idxes = [item[1] for item in checked_responses]

    assert len(generated_questions) == len(checked_answers) == len(checked_urls_idxes)

    for i in range(len(generated_questions)):
        answer = checked_answers[i]
        if answer is not None:
            url = urls[i][checked_urls_idxes[i]] if checked_urls_idxes[i] is not None else ''
            if 'sorry' in checked_answers[i].lower():
                answer = 'No answer could be found.'

            processed_pairs.append(
                {'question': generated_questions[i], 'answer': answer, 'url': url,
                 'scraped_text': ''})
    return processed_pairs


def combine_qa_results(greedy_checked_file, round1_check_indexes_file, round1_checked_file, round2_check_indexes_file,
                       round2_checked_file, round3_checked_indexes_file, reference_file, output_file):
    encoding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')

    greedy_checked = list(jsonlines.open(greedy_checked_file))
    round1_checked = list(jsonlines.open(round1_checked_file))
    round2_checked = list(jsonlines.open(round2_checked_file))

    with open(round1_check_indexes_file) as file:
        round1_idxes = json.load(file)
    with open(round2_check_indexes_file) as file:
        round2_idxes = json.load(file)
    with open(round3_checked_indexes_file) as file:
        round3_idxes = json.load(file)
    round1_idxes = [int(key) for key in round1_idxes.keys()]
    round2_idxes = [int(key) for key in round2_idxes.keys()]
    round3_idxes = [int(key) for key in round3_idxes.keys()]
    processed = []
    num_increased = 0
    for greedy_element in greedy_checked:
        claim_id = greedy_element['info']['claim_id']
        if claim_id not in round1_idxes:
            greedy_evidence = collect_qa_url_pairs(greedy_element)
            if len(greedy_evidence) >= 10:
                greedy_evidence = filter_qa_pairs(encoding_model, greedy_evidence, threshold=0.5)
            if 5 < len(greedy_evidence) < 10:
                greedy_evidence = filter_qa_pairs(encoding_model, greedy_evidence, threshold=0.8)
            processed.append({'claim_id': claim_id, 'claim': greedy_element['claim'], 'evidence': greedy_evidence})

    for round1_element in round1_checked:
        claim_id = round1_element['info']['claim_id']
        if claim_id not in round2_idxes:
            greedy_element = [element for element in greedy_checked if element['info']['claim_id'] == claim_id][0]
            greedy_evidence = collect_qa_url_pairs_v2(greedy_element)
            num_increased += len(greedy_evidence)
            round1_evidence = collect_qa_url_pairs(round1_element)
            total_evidence = greedy_evidence + round1_evidence
            if len(total_evidence) >= 10:
                filtered_evidence = filter_qa_pairs(encoding_model, total_evidence, threshold=0.5)
            elif 5 < len(total_evidence) < 10:
                filtered_evidence = filter_qa_pairs(encoding_model, total_evidence, threshold=0.8)
            else:
                filtered_evidence = total_evidence
            processed.append(
                {'claim_id': claim_id, 'claim': round1_element['claim'], 'evidence': filtered_evidence})

    for round2_element in round2_checked:
        claim_id = round2_element['info']['claim_id']
        if claim_id not in round3_idxes:
            greedy_element = [element for element in greedy_checked if element['info']['claim_id'] == claim_id][0]
            greedy_evidence = collect_qa_url_pairs_v2(greedy_element)
            round1_element = [element for element in round1_checked if element['info']['claim_id'] == claim_id][0]
            round1_evidence = collect_qa_url_pairs_v2(round1_element)
            num_increased += len(greedy_evidence)
            num_increased += len(round1_evidence)
            round2_evidence = collect_qa_url_pairs(round2_element)
            total_evidence = greedy_evidence + round1_evidence + round2_evidence
            if len(total_evidence) >= 10:
                filtered_evidence = filter_qa_pairs(encoding_model, total_evidence, threshold=0.5)
            elif 5 < len(total_evidence) < 10:
                filtered_evidence = filter_qa_pairs(encoding_model, total_evidence, threshold=0.8)
            else:
                filtered_evidence = total_evidence
            processed.append({'claim_id': claim_id, 'claim': round2_element['claim'],
                              'evidence': filtered_evidence})
    processed_recycled = recycle_responses_from_previous_rounds(round3_idxes, greedy_checked, round1_checked,
                                                                round2_checked)
    processed += processed_recycled
    processed.sort(key=lambda x: x['claim_id'])
    num_total = 0
    for element in processed:
        num_total += len(element['evidence'])

    print(num_total)
    with open(reference_file) as file:
        reference_data = json.load(file)
    for idx, element in enumerate(processed):
        assert element['claim'] == reference_data[idx]['claim']

    with open(output_file, "w", encoding="utf-8") as output_file:
        json.dump(processed, output_file, ensure_ascii=False, indent=4)


def recycle_qa_url_pairs(element):
    pairs = []
    generated_questions = element['generated_questions']
    urls = element['urls']
    checked_responses = element['checked_responses']
    checked_answers = [item[0] for item in checked_responses]
    checked_urls_idxes = [item[1] for item in checked_responses]
    for i, answer in enumerate(checked_answers):
        if answer is not None:
            if 'sorry' in answer.lower():
                answer = 'No answer could be found.'
            question = generated_questions[i]
            url = urls[i][checked_urls_idxes[i]] if checked_urls_idxes[i] is not None else ''
            pairs.append({"question": question, 'answer': answer, 'url': url, 'scraped_text': ''})
    return pairs


def recycle_responses_from_previous_rounds(idxes, greedy_checked, round1_checked, round2_checked):
    processed = []
    encoding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    round1_idxes = [element['info']['claim_id'] for element in round1_checked]
    round2_idxes = [element['info']['claim_id'] for element in round2_checked]
    for idx in idxes:
        qa_url_pairs = []
        # recycle qa pairs from greedy round
        greedy_element = greedy_checked[idx]
        greedy_qa_url_pairs = recycle_qa_url_pairs(greedy_element)
        qa_url_pairs += greedy_qa_url_pairs

        # recycle qa_pairs from round 1 checked
        round1_element = round1_checked[round1_idxes.index(idx)]
        round1_qa_url_pairs = recycle_qa_url_pairs(round1_element)
        qa_url_pairs += round1_qa_url_pairs

        # recycle qa_pairs from round 2 checked
        round2_element = round2_checked[round2_idxes.index(idx)]
        round2_qa_url_pairs = recycle_qa_url_pairs(round2_element)
        qa_url_pairs += round2_qa_url_pairs

        # process special cases
        if len(qa_url_pairs) >= 10:
            qa_url_pairs = filter_qa_pairs(encoding_model, qa_url_pairs, threshold=0.5)
        elif 5 < len(qa_url_pairs) < 10:
            qa_url_pairs = filter_qa_pairs(encoding_model, qa_url_pairs, threshold=0.8)
        if len(qa_url_pairs) == 0:
            print(idx)
            generated_questions = greedy_element['generated_questions']
            answers = ['No answer could be found.'] * len(generated_questions)
            for i in range(len(generated_questions)):
                if 'sorry' in answers[i].lower():
                    answer = 'No answer could be found.'
                else:
                    answer = answers[i]
                qa_url_pairs.append(
                    {'question': generated_questions[i], 'answer': answer, 'url': '', 'scraped_text': ''})
        processed.append({'claim_id': idx, 'claim': greedy_element['claim'], 'evidence': qa_url_pairs})
    return processed


def filter_qa_pairs(encoding_model, qa_pairs, threshold=0.8):
    selected_pairs = [qa_pairs[0]]
    for i in range(1, len(qa_pairs)):
        existing_questions = [pair['question'] for pair in selected_pairs]
        question = qa_pairs[i]['question']
        if question not in existing_questions:
            answer = qa_pairs[i]['answer']
            qa = f'{question} {answer}'
            existing_qas = [f'{pair["question"]} {pair["answer"]}' for pair in selected_pairs]

            compare_pairs = [(qa, existing_qa) for existing_qa in existing_qas]
            similarities = []
            for pair in compare_pairs:
                embeddings = encoding_model.encode([pair[0], pair[1]], show_progress_bar=False)
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                similarities.append(similarity)
            if all(similarity < threshold for similarity in similarities):
                selected_pairs.append(qa_pairs[i])
        else:
            print(question)
    return selected_pairs


def list_idx():
    with open('./data/test/further_check_round1.json') as file:
        idx1 = json.load(file)
    with open('./data/test/further_check_round2.json') as file:
        idx2 = json.load(file)
    with open('./data/test/further_check_round3.json') as file:
        idx3 = json.load(file)
    idx1 = [int(key) for key in idx1.keys()]
    print('############Check 1#################')
    print(idx1)
    idx2 = [int(key) for key in idx2.keys()]
    print('############Check 2#################')
    print(idx2)
    idx3 = [int(key) for key in idx3.keys()]
    print('############Check 3#################')
    print(idx3)


def add_checked_statements(input_file, output_file):
    check_data = list(jsonlines.open(input_file))
    for element in check_data:
        checked_statements = []
        checked_responses = element['checked_responses']
        selected_responses = element['selected_responses']
        selected_statements = element['selected_statements']
        for i in range(len(checked_responses)):
            checked_response = checked_responses[i][0]
            if checked_response is not None:
                response_idx = selected_responses[i].index(checked_response)
                checked_statements.append(selected_statements[i][response_idx])
            else:
                checked_statements.append(None)
        element['checked_statements'] = checked_statements
        assert len(checked_statements) == len(element['generated_questions'])
        dump_jsonl([element], output_file, append=True)


def get_round1_round2_questions_for_all(generated_question_file):
    encoding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    generated_questions = list(jsonlines.open(generated_question_file))
    round1_further_check = {}
    round2_further_check = {}
    for element in generated_questions:
        claim_id = element['info']['claim_id']
        greedy_questions = element['greedy_questions']
        sampling_questions = element['sampling_questions']
        # round 1 questions
        similarities_rank = rank_question_similarity(encoding_model, greedy_questions, sampling_questions)
        round1_further_check[claim_id] = similarities_rank[0]
        round1_questions = sampling_questions[similarities_rank[0]]
        # round 2 questions
        rest_questions = [item for item in sampling_questions if item != round1_questions]
        similarities_rank = rank_question_similarities_round2(encoding_model,
                                                              [greedy_questions, round1_questions],
                                                              rest_questions)
        round2_further_check[claim_id] = sampling_questions.index(rest_questions[similarities_rank[0]])
    with open('./data/dev/further_check_dev_round1_all_claims.json', 'w') as file:
        json.dump(round1_further_check, file)
    with open('./data/dev/further_check_dev_round2_all_claims.json', 'w') as file:
        json.dump(round2_further_check, file)


def add_claim_id_to_dev(input_file, output_file):
    if 'jsonl' in input_file:
        data = list(jsonlines.open(input_file))
        for idx, element in enumerate(data):
            element['info']['claim_id'] = idx
            dump_jsonl([element], output_file, append=True)
    else:
        processed_data = []
        with open(input_file) as file:
            data = json.load(file)
        for idx, element in enumerate(data):
            element['claim_id'] = idx
            processed_data.append(element)
        with open(output_file, 'w') as file:
            json.dump(processed_data, file)


def sort_generations(input_file, output_file):
    q1 = list(jsonlines.open('./data/dev/sorted_claim_generated_question_512_dev_round1_all.jsonl'))
    q2 = list(jsonlines.open('./data/dev/sorted_claim_generated_question_512_dev_round2_all.jsonl'))
    generation = list(jsonlines.open(input_file))
    q1_ids = [element['info']['claim_id'] for element in q1]
    q2_ids = [element['info']['claim_id'] for element in q2]
    q_ids = [element['info']['claim_id'] for element in generation[500:]]
    q_gen = [element['generated_questions'] for element in generation[500:]]

    q1_idxes = []
    for id in q1_ids:
        q1_idxes.append(q_ids.index(id))

    q2_idxes = [i for i in range(1000) if i not in q1_idxes]

    q1_gen = [element['generated_questions'] for element in q1]
    q2_gen = [element['generated_questions'] for element in q2]

    processed = generation[:500]
    for idx in q1_idxes:
        processed.append(generation[500:][idx])
    for idx in q2_idxes:
        processed.append(generation[500:][idx])
    for idx, element in enumerate(processed[500:1000]):
        assert element['generated_questions'] == q1_gen[idx]
    for idx, element in enumerate(processed[1000:]):
        assert element['generated_questions'] == q2_gen[idx]
    for element in processed:
        dump_jsonl([element], output_file, append=True)


def separate_generation(input_file):
    generation = list(jsonlines.open(input_file))
    for element in generation[:500]:
        dump_jsonl([element],
                   './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_greedy_statements_check.jsonl',
                   append=True)
    for element in generation[500:1000]:
        dump_jsonl([element],
                   './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_round1_statements_check.jsonl',
                   append=True)
    for element in generation[1000:1500]:
        dump_jsonl([element],
                   './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_round2_statements_check.jsonl',
                   append=True)


def create_qa_evidence_gpt4():
    greedy = list(jsonlines.open('./data/dev/qa_generated_questions_combined_dev_top3_gpt-4-turbo.jsonl'))
    round1 = list(jsonlines.open('./data/dev/qa_generated_questions_combined_dev_top3_gpt-4-turbo_round1.jsonl'))
    with open('./data/dev.json') as file:
        dev = json.load(file)
    processed = []
    for idx in range(500):
        greedy_element = greedy[idx]
        round1_element = round1[idx]
        processed_single = {}
        claim = greedy_element['claim']
        assert claim == round1_element['claim'] == dev[idx]['claim']
        greedy_questions = greedy_element['generated_questions']
        greedy_answers = greedy_element['greedy_responses']
        round1_questions = round1_element['generated_questions']
        round1_answers = round1_element['greedy_responses']
        assert len(greedy_questions) == len(greedy_answers)
        assert len(round1_questions) == len(round1_answers)
        questions = greedy_questions + round1_questions
        answers = greedy_answers + round1_answers
        processed_single['claim'] = claim
        processed_single['claim_id'] = idx
        processed_single['evidence'] = []
        for pair in zip(questions, answers):
            processed_single['evidence'].append({'question': pair[0], 'answer': pair[1]})
        processed.append(processed_single)
    with open('./data/dev/qa_dev_gpt4_turbo_two_rounds.json', 'w') as file:
        json.dump(processed, file)


if __name__ == '__main__':
    create_qa_evidence_gpt4()
    # separate_generation('./data/dev/qa_generated_questions_combined_dev_top3_statements_check_all_rounds.jsonl')
    # sort_generations('./data/dev/qa_generated_questions_combined_dev_top3_answer_selection.jsonl',
    #                 './data/dev/qa_generated_questions_combined_dev_top3_answer_selection-1.jsonl')
    # add_claim_id_to_dev('./data/dev/llama3_question_generation_dev_greedy_sampling_sub_included_wo_dev_processed.jsonl',
    #                    './data/dev/llama3_question_generation_dev_greedy_sampling_sub_included_wo_dev_processed-1.jsonl')
    # get_round1_round2_questions_for_all(
    #    './data/dev/llama3_question_generation_dev_greedy_sampling_sub_included_wo_dev_processed.jsonl')
    # add_checked_statements('./data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl',
    #                       './data/test/qa_generated_questions_combined_test_top3_statements_check_round2_v2.jsonl')
    '''combine_qa_results('./data/test/qa_generated_questions_combined_test_top3_statements_check.jsonl',
                       './data/test/further_check_round1.json',
                       './data/test/qa_generated_questions_combined_test_top3_statements_check_round1.jsonl',
                       './data/test/further_check_round2.json',
                       './data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl',
                       './data/test/further_check_round3.json', './data/data_test.json',
                       './data/test/combined_qa_results_augmented.json')'''
    # check_qa_completeness('./data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl',
    #                      './data/test/llama3_question_generation_test_greedy_sampling_sub_included_w_dev_processed.jsonl'
    #                      , check_round=3)
    # response_entailment_check('./data/dev/qa_generated_questions_combined_dev_top3_statements_all_rounds.jsonl',
    #                          './data/dev/qa_generated_questions_combined_dev_top3_statements_check_all_rounds.jsonl')
    # check_difference('./data/test/further_check_round2.json', './data/test/further_check_round2-1.json')
    # combine_statements(['./data/test/qa_generated_questions_combined_test_top3_statements_round2_222.jsonl',
    #                    './data/test/qa_generated_questions_combined_test_top3_statements_round2_445.jsonl'],
    #                   './data/test/llama3_question_generation_test_greedy_sampling_sub_included_w_dev_processed.jsonl',
    #                   './data/test/further_check_round2.json',
    #                   './data/test/qa_generated_questions_combined_test_top3_statements_round2.jsonl')
    # select_question('./data/test/llama3_question_generation_test_greedy_sampling_sub_included_w_dev_processed.jsonl',
    #                None)
    # check_qa_completeness('./data/test/qa_generated_questions_combined_test_top3_statements_check_round1.jsonl',
    #                      './data/test/llama3_question_generation_test_greedy_sampling_sub_included_w_dev_processed.jsonl'
    #                      , check_round=2)
    # evaluate_entailment('./data/test/qa_generated_questions_combined_test_top3_statements_entailment_mixtral.jsonl')
    # check_nei_response('./data/test/qa_generated_questions_combined_test_top3_statements.jsonl')
    '''response_entailment_check('./data/test/qa_generated_questions_combined_test_top3_statements_round2_rest.jsonl',
                              './data/test/qa_generated_questions_combined_test_top3_statements_check_round2_rest.jsonl')'''
    '''combine_file(['./data/test/qa_generated_questions_combined_test_top3_statements_entailment_mixtral_1100.jsonl',
                  './data/test/qa_generated_questions_combined_test_top3_statements_entailment_mixtral_2215.jsonl'],
                 './data/data_test.json',
                 './data/test/qa_generated_questions_combined_test_top3_statements_entailment_mixtral.jsonl')'''
    '''combine_greedy_sampling(
        './data/dev/llama3_question_generation_dev_greedy_sub_included_wo_dev_processed.jsonl',
        './data/dev/llama3_question_generation_dev_sampling_sub_included_wo_dev_processed.jsonl',
        './data/dev/llama3_question_generation_dev_greedy_sampling_sub_included_wo_dev_processed.jsonl'
    )'''
    # select_answer('./data/dev/qa_generated_questions_combined_dev_top3_greedy_sampling_all_rounds.jsonl',
    #              output_file='./data/dev/qa_generated_questions_combined_dev_top3_answer_selection_all_rounds.jsonl')
