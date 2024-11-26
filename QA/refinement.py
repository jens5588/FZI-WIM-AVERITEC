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

    for i, answer in enumerate(checked_answers):
        if answer is not None:
            if 'sorry' in checked_answers[i].lower():
                answer = 'No answer could be found.'
            url = urls[i][checked_urls_idxes[i]] if checked_urls_idxes[i] is not None else ''
            processed_pairs.append(
                {'question': generated_questions[i], 'answer': answer, 'url': url,
                 'scraped_text': ''})
    return processed_pairs


def filter_qa_pairs(embedding_model, cross_encoder, claim, qa_pairs, threshold=0.9):
    qa_combined = [f'{pair["question"]} {pair["answer"]}' for pair in qa_pairs]
    claim_qa_text_pairs = [(claim, qa) for qa in qa_combined]
    scores = cross_encoder.predict(claim_qa_text_pairs, batch_size=64, show_progress_bar=False).tolist()
    scores_ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sorted_qa_pairs = [qa_pairs[i] for i in scores_ranking]

    selected_pairs = [sorted_qa_pairs[0]]

    for i in range(1, len(sorted_qa_pairs)):
        existing_questions = [pair['question'] for pair in selected_pairs]
        question = sorted_qa_pairs[i]['question']
        if question not in existing_questions:
            answer = sorted_qa_pairs[i]['answer']
            qa = f'{question} {answer}'
            existing_qas = [f'{pair["question"]} {pair["answer"]}' for pair in selected_pairs]

            compare_pairs = [(qa, existing_qa) for existing_qa in existing_qas]
            similarities = []
            for pair in compare_pairs:
                embeddings = embedding_model.encode([pair[0], pair[1]], show_progress_bar=False)
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                similarities.append(similarity)
            if all(similarity < threshold for similarity in similarities):
                selected_pairs.append(sorted_qa_pairs[i])
        else:
            print(question)
    return selected_pairs[::-1]


def combine_qa_results(greedy_checked_file, round1_check_indexes_file, round1_checked_file, round2_check_indexes_file,
                       round2_checked_file, round3_checked_indexes_file, reference_file, output_file):
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device='cuda')

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
        claim = greedy_element['claim']
        if claim_id not in round1_idxes:
            greedy_evidence = collect_qa_url_pairs(greedy_element)

            if len(greedy_evidence) >= 10:
                greedy_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, greedy_evidence, threshold=0.5)
            if 5 < len(greedy_evidence) < 10:
                greedy_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, greedy_evidence, threshold=0.8)
            processed.append({'claim_id': claim_id, 'claim': greedy_element['claim'], 'evidence': greedy_evidence})

    for round1_element in round1_checked:
        claim_id = round1_element['info']['claim_id']
        claim = round1_element['claim']
        if claim_id not in round2_idxes:
            greedy_element = [element for element in greedy_checked if element['info']['claim_id'] == claim_id][0]
            greedy_evidence = collect_qa_url_pairs_v2(greedy_element)
            num_increased += len(greedy_evidence)
            round1_evidence = collect_qa_url_pairs(round1_element)
            total_evidence = greedy_evidence + round1_evidence
            if len(total_evidence) >= 10:
                filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, total_evidence,
                                                    threshold=0.5)
            elif 5 < len(total_evidence) < 10:
                filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, total_evidence,
                                                    threshold=0.8)
            else:
                filtered_evidence = total_evidence
            processed.append(
                {'claim_id': claim_id, 'claim': round1_element['claim'], 'evidence': filtered_evidence})

    for round2_element in round2_checked:
        claim_id = round2_element['info']['claim_id']
        claim = round2_element['claim']
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
                filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, total_evidence,
                                                    threshold=0.5)
            elif 5 < len(total_evidence) < 10:
                filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, claim, total_evidence,
                                                    threshold=0.8)
            else:
                filtered_evidence = total_evidence
            processed.append({'claim_id': claim_id, 'claim': round2_element['claim'],
                              'evidence': filtered_evidence})
    processed_recycled = recycle_responses_from_previous_rounds(round3_idxes, greedy_checked, round1_checked,
                                                                round2_checked, embedding_model, cross_encoder)
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


def recycle_responses_from_previous_rounds(idxes, greedy_checked, round1_checked, round2_checked, embedding_model,
                                           cross_encoder):
    processed = []
    round1_idxes = [element['info']['claim_id'] for element in round1_checked]
    round2_idxes = [element['info']['claim_id'] for element in round2_checked]
    for idx in idxes:
        qa_url_pairs = []
        # recycle qa pairs from greedy round
        greedy_element = greedy_checked[idx]
        claim = greedy_element['claim']
        greedy_qa_url_pairs = collect_qa_url_pairs_v2(greedy_element)
        qa_url_pairs += greedy_qa_url_pairs

        # recycle qa_pairs from round 1 checked
        round1_element = round1_checked[round1_idxes.index(idx)]
        round1_qa_url_pairs = collect_qa_url_pairs_v2(round1_element)
        qa_url_pairs += round1_qa_url_pairs

        # recycle qa_pairs from round 2 checked
        round2_element = round2_checked[round2_idxes.index(idx)]
        round2_qa_url_pairs = collect_qa_url_pairs_v2(round2_element)
        qa_url_pairs += round2_qa_url_pairs

        # process special cases
        if len(qa_url_pairs) >= 10:
            qa_url_pairs = filter_qa_pairs(embedding_model, cross_encoder, claim, qa_url_pairs, threshold=0.5)
        elif 5 < len(qa_url_pairs) < 10:
            qa_url_pairs = filter_qa_pairs(embedding_model, cross_encoder, claim, qa_url_pairs, threshold=0.8)
        if len(qa_url_pairs) == 0:
            print(idx)
            generated_questions = greedy_element['generated_questions']
            answers = ['No answer could be found.'] * len(generated_questions)
            for i, answer in enumerate(answers):
                qa_url_pairs.append(
                    {'question': generated_questions[i], 'answer': answer, 'url': '', 'scraped_text': ''})
        processed.append({'claim_id': idx, 'claim': greedy_element['claim'], 'evidence': qa_url_pairs})
    return processed


def combine_conditional_check_rest():
    round1_conditional_check = list(
        jsonlines.open('./data/test/qa_generated_questions_combined_test_top3_statements_check_round1.jsonl'))
    print(len(round1_conditional_check))
    round1_rest = list(
        jsonlines.open('./data/test/qa_generated_questions_combined_test_top3_statements_check_round1_rest.jsonl'))
    print(len(round1_rest))
    round1 = round1_conditional_check + round1_rest
    round1.sort(key=lambda x: x['info']['claim_id'])
    round1_claim_ids = [element['info']['claim_id'] for element in round1]
    assert round1_claim_ids == list(range(2215))
    round2_conditional_check = list(
        jsonlines.open('./data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl'))
    print(len(round2_conditional_check))
    round2_rest = list(
        jsonlines.open('./data/test/qa_generated_questions_combined_test_top3_statements_check_round2_rest.jsonl'))
    print(len(round2_rest))
    round2 = round2_conditional_check + round2_rest
    round2.sort(key=lambda x: x['info']['claim_id'])
    round2_claim_ids = [element['info']['claim_id'] for element in round2]
    assert round2_claim_ids == list(range(2215))
    for element in round1:
        dump_jsonl([element],
                   './data/test/qa_generated_questions_combined_test_top3_statements_check_round1_all_included.jsonl',
                   append=True)
    for element in round2:
        dump_jsonl([element],
                   './data/test/qa_generated_questions_combined_test_top3_statements_check_round2_all_included.jsonl',
                   append=True)


def combine_qa_results_all_included(greedy_checked_file, round1_checked_file, round2_checked_file, reference_file,
                                    output_file):
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device='cuda')

    greedy_checked = list(jsonlines.open(greedy_checked_file))
    print(len(greedy_checked))
    round1_checked = list(jsonlines.open(round1_checked_file))
    round2_checked = list(jsonlines.open(round2_checked_file))
    assert len(greedy_checked) == len(round1_checked) == len(round2_checked)
    processed = []
    num_no_answered = 0
    for idx in tqdm(range(len(greedy_checked))):
        qa_url_pairs = []
        greedy_element = [element for element in greedy_checked if element['info']['claim_id'] == idx][0]
        greedy_evidence = collect_qa_url_pairs_v2(greedy_element)
        round1_element = [element for element in round1_checked if element['info']['claim_id'] == idx][0]
        round1_evidence = collect_qa_url_pairs_v2(round1_element)
        round2_element = [element for element in round2_checked if element['info']['claim_id'] == idx][0]
        round2_evidence = collect_qa_url_pairs(round2_element)
        total_evidence = greedy_evidence + round1_evidence + round2_evidence
        assert greedy_element['claim'] == round1_element['claim'] == round2_element['claim']
        if len(total_evidence) >= 10:
            filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, greedy_element['claim'], total_evidence,
                                                threshold=0.5)
        elif 5 < len(total_evidence) < 10:
            filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, greedy_element['claim'], total_evidence,
                                                threshold=0.8)
        elif 1 <= len(total_evidence) <= 5:
            filtered_evidence = total_evidence
        else:
            num_no_answered += 1
            generated_questions = greedy_element['generated_questions']
            answers = ['No answer could be found.'] * len(generated_questions)
            for i, answer in enumerate(answers):
                qa_url_pairs.append(
                    {'question': generated_questions[i], 'answer': answer, 'url': '', 'scraped_text': ''})
            filtered_evidence = qa_url_pairs
        processed.append({'claim_id': idx, 'claim': greedy_element['claim'],
                          'evidence': filtered_evidence})
    processed.sort(key=lambda x: x['claim_id'])
    print(f'num no answer {num_no_answered}')

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


def combine_qa_results_greedy_only(greedy_file, reference_file, output_file):
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device='cuda')
    greedy_checked = list(jsonlines.open(greedy_file))
    num_no_answered = 0
    processed = []
    for idx in tqdm(range(len(greedy_checked))):
        qa_url_pairs = []
        greedy_element = [element for element in greedy_checked if element['info']['claim_id'] == idx][0]
        greedy_evidence = collect_qa_url_pairs_v2(greedy_element)
        if len(greedy_evidence) >= 10:
            filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, greedy_element['claim'],
                                                greedy_evidence,
                                                threshold=0.5)
        elif 5 < len(greedy_evidence) < 10:
            filtered_evidence = filter_qa_pairs(embedding_model, cross_encoder, greedy_element['claim'],
                                                greedy_evidence,
                                                threshold=0.8)
        elif 1 <= len(greedy_evidence) <= 5:
            filtered_evidence = greedy_evidence
        else:
            num_no_answered += 1
            generated_questions = greedy_element['generated_questions']
            answers = ['No answer could be found.'] * len(generated_questions)
            for i, answer in enumerate(answers):
                qa_url_pairs.append(
                    {'question': generated_questions[i], 'answer': answer, 'url': '', 'scraped_text': ''})
            filtered_evidence = qa_url_pairs
        processed.append({'claim_id': idx, 'claim': greedy_element['claim'],
                          'evidence': filtered_evidence})
    processed.sort(key=lambda x: x['claim_id'])
    print(f'num no answer {num_no_answered}')

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


if __name__ == '__main__':
    combine_qa_results_greedy_only('./data/dev/qa_generated_questions_combined_chatqa2_dev_top3_greedy_statements_check.jsonl',
                                   './data/dev.json',
                                   './data/dev/combined_qa_results_chatqa2_dev_greedy.json')
    '''combine_qa_results_all_included(
        './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_greedy_statements_check.jsonl',
        './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_round1_statements_check.jsonl',
        './data/dev/qa_generated_questions_combined_chatqa2_dev_top3_round2_statements_check.jsonl',
        './data/dev.json',
        './data/dev/combined_chatqa2_dev_qa_results_all_included.json')'''
    # combine_conditional_check_rest()
    '''combine_qa_results('./data/test/qa_generated_questions_combined_test_top3_statements_check.jsonl',
                       './data/test/further_check_round1.json',
                       './data/test/qa_generated_questions_combined_test_top3_statements_check_round1.jsonl',
                       './data/test/further_check_round2.json',
                       './data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl',
                       './data/test/further_check_round3.json', './data/data_test.json',
                       './data/test/combined_qa_results_cross.json')'''
