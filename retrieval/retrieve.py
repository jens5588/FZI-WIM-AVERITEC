import itertools
import json
import jsonlines
import torch
import torch.nn.functional as F
import numpy as np
import nltk
import os
import pandas as pd
from fairseq.data.data_utils import collate_tokens
from collections import Counter
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from collections import Counter
from tqdm import tqdm
from time import time
from rank_bm25 import BM25Okapi
from sklearn.metrics import confusion_matrix, classification_report

from joblib import Parallel, delayed, parallel_backend
from llama_recipes.utils.inference_utils import dump_jsonl


# https://huggingface.co/Salesforce/SFR-Embedding-Mistral
def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_embeddings(texts, model, tokenizer, max_length, process_count, normalize=False):
    batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True,
                           return_tensors='pt').to(f'cuda:{process_count}')
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def get_url_text(mode, idx):
    if mode == 'train':
        search_data = list(
            jsonlines.open(
                f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json"))
    elif mode == 'dev':
        search_data = list(
            jsonlines.open(
                f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json"))
    else:
        raise NotImplementedError
    url_text = []
    for item in search_data:
        if len(item["url2text"]) > 0:
            url_text.append((item["url"], " ".join(item["url2text"])))

    return url_text


def get_remaining_retrieval_indexes(data, existing_retrieval_path, total_retrieval_indexes):
    print(f'total {len(total_retrieval_indexes)}')
    existing_retrieval = list(jsonlines.open(existing_retrieval_path))
    existing_claims = [key for element in existing_retrieval for key, _ in element.items()]
    claims_list = [element['claim'] for element in data]
    existing_indexes = list(set([claims_list.index(claim) for claim in existing_claims]))

    print(f'existing indexes: {len(existing_indexes)}')
    remain_indexes = [idx for idx in total_retrieval_indexes if idx not in existing_indexes]
    print(len(remain_indexes))
    return remain_indexes


def simple_claim_retrieval(input_file, embedding_model, max_length, batch_size, process_count, num_processes,
                           question_included=False, normalize=False):
    # gte-Qwen2-7B-instruct ; SFR-Embedding-Mistral
    tokenizer = AutoTokenizer.from_pretrained(f"/hkfs/work/workspace_haic/scratch/vl8701-llm/{embedding_model}",
                                              use_fast=True)
    model = AutoModel.from_pretrained(f"/hkfs/work/workspace_haic/scratch/vl8701-llm/{embedding_model}",
                                      torch_dtype=torch.bfloat16).to(f'cuda:{process_count}')
    with open(input_file) as file:
        data = json.load(file)

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    normalized_text = 'w_normalization' if normalize else 'wo_normalization'
    query_text = 'claim_question' if question_included else 'claim_only'
    if 'train' in input_file:
        mode = 'train'
    elif 'dev' in input_file:
        mode = 'dev'
    else:
        raise NotImplementedError
    sorted_claim_url_path = f'./data/sorted_{query_text}_url_pairs_{embedding_model}_{max_length}_{normalized_text}_{mode}_process_{process_count}.jsonl'
    with open('./data/train_wo_text_idxes.json') as file:
        wo_text_idxes = json.load(file)
    total_retrieval_indexes = [idx for idx in list(range(len(data))) if idx not in wo_text_idxes]
    existing_retrieval_path = f'./data/sorted_{query_text}_url_pairs_{embedding_model}_{max_length}_{normalized_text}_{mode}.jsonl'

    if os.path.exists(existing_retrieval_path):
        remain_indexes = get_remaining_retrieval_indexes(data, existing_retrieval_path, total_retrieval_indexes)
    else:
        remain_indexes = total_retrieval_indexes
    num_per_process = len(remain_indexes) // num_processes + 1
    process_idxes = remain_indexes[
                    int(process_count * num_per_process):int((process_count + 1) * num_per_process)]

    for idx in tqdm(process_idxes):
        element = data[idx]
        claim = element['claim']
        if question_included:
            questions = [item['question'] for item in element['questions']]
            questions_text = " ".join(questions)

        if mode == 'train':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json"))
        elif mode == 'dev':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json"))
        else:
            assert NotImplementedError
        url_text = []
        for item in search_data:
            if len(item["url2text"]) > 0:
                url_text.append({item["url"]: " ".join(item["url2text"])})
        documents = [value for item in url_text for key, value in item.items()]
        documents_length = [len(document) for document in documents]
        length_sorted_document_indexes = sorted(range(len(documents_length)), key=lambda k: documents_length[k],
                                                reverse=True)

        length_sorted_documents = [documents[j] for j in length_sorted_document_indexes]
        length_sorted_url_text = [url_text[j] for j in length_sorted_document_indexes]
        if question_included:
            claim += questions_text
            queries = [get_detailed_instruct(task, claim)]
        else:
            queries = [get_detailed_instruct(task, claim)]
        input_texts = queries + length_sorted_documents

        if len(input_texts) % batch_size != 0:
            batches = len(input_texts) // batch_size + 1
        else:
            batches = len(input_texts) // batch_size
        inputs_embeddings = None
        for i in tqdm(range(batches)):
            if i != batches - 1:
                batch_candidate_embeddings = get_embeddings(input_texts[i * batch_size: (i + 1) * batch_size], model,
                                                            tokenizer, max_length, process_count=process_count,
                                                            normalize=normalize)
            else:
                batch_candidate_embeddings = get_embeddings(input_texts[i * batch_size:], model, tokenizer, max_length,
                                                            process_count=process_count, normalize=normalize)
            if inputs_embeddings is None:
                inputs_embeddings = batch_candidate_embeddings
            else:
                inputs_embeddings = torch.cat((inputs_embeddings, batch_candidate_embeddings), dim=0)

        # save embeddings
        embedding_path = \
            f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/embeddings/{query_text}_{embedding_model}_{max_length}_{normalized_text}_{mode}_{idx}.pt'
        torch.save(inputs_embeddings, embedding_path)

        scores = torch.flatten((inputs_embeddings[:1] @ inputs_embeddings[1:].T)).tolist()
        assert len(scores) == len(length_sorted_documents)

        relevance_sorted_indexes = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        relevance_sorted_url_text = [length_sorted_url_text[sorted_index] for sorted_index in relevance_sorted_indexes]
        relevance_sorted_urls = [key for element in relevance_sorted_url_text for key, _ in element.items()]
        dump_jsonl([{f'{claim}': relevance_sorted_urls}], sorted_claim_url_path, append=True)


def coarse_retrieval_bm25(input_file, process_count, num_processes):
    output_file = f'./data/coarse_retrieval_bm25_train_{process_count}.jsonl'
    with open(input_file) as file:
        data = json.load(file)
    with open('./data/train_wo_text_idxes.json') as file:
        train_wo_text_idxes = json.load(file)
    if 'train' in input_file:
        mode = 'train'
    elif 'dev' in input_file:
        mode = 'dev'
    else:
        raise NotImplementedError
    idxes_retrieval = [idx for idx in list(range(len(data))) if idx not in train_wo_text_idxes]
    if len(idxes_retrieval) % num_processes != 0:
        num_per_process = len(idxes_retrieval) // num_processes + 1
    else:
        num_per_process = len(idxes_retrieval) / num_processes
    process_idxes = idxes_retrieval[
                    int(process_count * num_per_process):int((process_count + 1) * num_per_process)]
    for idx in tqdm(process_idxes):
        element = data[idx]
        claim = element['claim']
        url_text = get_url_text(mode, idx)
        urls = [item[0] for item in url_text]
        texts = [item[1] for item in url_text]
        tokenized_docs = [nltk.word_tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(claim))
        sorted_idx = np.argsort(scores)[::-1].tolist()
        sorted_urls = [urls[i] for i in sorted_idx]
        dump_jsonl([{f'{claim}': sorted_urls}], output_file, append=True)


def evaluate_retrieve(input_file, retrieval_claim_file, retrieval_claim_question_file):
    with open(input_file) as file:
        data = json.load(file)
    with open('./data/train_wo_text_idxes.json') as file:
        train_wo_text_idxes = json.load(file)
    data = [data[idx] for idx in range(len(data)) if idx not in train_wo_text_idxes]
    claims = [element['claim'] for element in data]
    retrieval_claim = list(jsonlines.open(retrieval_claim_file))
    retrieval_claim_queries = [key for element in retrieval_claim for key, value in element.items()]

    retrieval_claim_question = list(jsonlines.open(retrieval_claim_question_file))
    retrieval_claim_question_queries = [key for element in retrieval_claim_question for key, value in element.items()]
    retrieval_claim_question_claims = []
    for claim_question_query in retrieval_claim_question_queries:
        for claim_query in retrieval_claim_queries:
            if claim_query in claim_question_query:
                retrieval_claim_question_claims.append(claim_query)
                break
    assert len(retrieval_claim_question_queries) == len(retrieval_claim_question_claims)
    '''for element in zip(retrieval_claim_question_claims,retrieval_claim_question_queries):
        print(element[0])
        print(element[1])
        print('#################')'''
    proportions_claim = []
    proportions_claim_question = []
    num_error = 0
    for idx, element in enumerate(retrieval_claim_question):
        claim_question_query, claim_question_urls = list(element.items())[0]
        claim = retrieval_claim_question_claims[idx]
        # print(claim)
        retrieval_claim_idx = retrieval_claim_queries.index(claim)
        _, claim_urls = list(retrieval_claim[retrieval_claim_idx].items())[0]
        proportion_claim = []
        proportion_claim_question = []
        claim_idx = claims.index(claim)

        source_urls = []
        questions = data[claim_idx]['questions']
        for question in questions:
            for answer in question['answers']:
                source_urls.append(answer['source_url'])
        source_urls = list(set(source_urls))
        num_documents = len(claim_question_urls)

        for url in source_urls:
            try:
                retrieved_url_ranking_claim = claim_question_urls.index(url) + 1
                retrieved_url_ranking_claim_question = claim_urls.index(url) + 1
                proportion_claim_question.append(retrieved_url_ranking_claim_question / num_documents)
                proportion_claim.append(retrieved_url_ranking_claim / num_documents)
            except:
                pass
        if len(proportion_claim) > 0 and len(proportion_claim_question) > 0:
            avg_proportion_claim = sum(proportion_claim) / len(proportion_claim)
            avg_proportion_claim_question = sum(proportion_claim_question) / len(proportion_claim_question)
            proportions_claim.append(avg_proportion_claim)
            proportions_claim_question.append(avg_proportion_claim_question)
        else:
            num_error += 1
            print(f'claim: {claim}')
            print(f'claim question query: {claim_question_query}')
            print('---------------')

    print(f'Total: {len(retrieval_claim_question)} - Error:{num_error}')
    print(pd.Series(proportions_claim).describe(percentiles=[0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.995, 0.999]))
    print(pd.Series(proportions_claim_question).describe(percentiles=[0.1, 0.2, 0.5, 0.75, 0.9, 0.95, 0.995, 0.999]))


def check_url_text(input_file):
    with open(input_file) as file:
        data = json.load(file)
    mode = 'train' if 'train' in input_file else 'dev'
    num_correction = 0
    idx = 0
    idx_wo_text = []
    for element in tqdm(data):
        if mode == 'train':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json"))
        elif mode == 'dev':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json"))
        url_text = {}
        source_urls = []
        cached_source_urls = []
        for item in search_data:
            url2text = " ".join(item['url2text'])
            if len(url2text) > 0:
                url_text[item['url']] = " ".join(item['url2text'])

        questions = element['questions']
        for question in questions:
            for answer in question['answers']:
                source_urls.append(answer['source_url'])
                if 'cached_source_url' in list(answer.keys()):
                    cached_source_urls.append(answer['cached_source_url'])
        source_urls = list(set(source_urls))
        urls = [url for url in source_urls if url in list(url_text.keys())]
        cached_urls = [url for url in cached_source_urls if url in list(url_text.keys())]

        if len(urls) == 0:
            idx_wo_text.append(idx)
            if len(cached_urls) > 0:
                num_correction += 1

        idx += 1
    print(num_correction)
    with open('./data/train_wo_text_idxes.json', 'w') as file:
        json.dump(idx_wo_text, file)


def combine_generation(input_files, output_file):
    generated_data = []
    for file in input_files:
        generation_part = list(jsonlines.open(file))
        generated_data.extend(generation_part)
    for element in generated_data:
        dump_jsonl([element], output_file, append=True)


def build_sentence_chunking(sentences, tokenizer, max_length):
    sentences_tokens = tokenizer(sentences, truncation=True, max_length=max_length)
    num_tokens = [len(element) - 2 for element in sentences_tokens.input_ids]
    sentence_chunks = []
    chunks_nums = []

    while len(num_tokens) > 0:
        if sum(num_tokens) <= max_length:
            sentence_chunks.append(sentences)
            chunks_nums.append(num_tokens)
            break
        else:
            for i in range(2, len(num_tokens) + 1):
                if sum(num_tokens[:i]) >= max_length:
                    sentence_chunks.append(sentences[:i - 1])
                    chunks_nums.append(num_tokens[:i - 1])
                    num_tokens = num_tokens[i - 1:]
                    sentences = sentences[i - 1:]
                    break

    sums = [1 if sum(element) <= max_length else 0 for element in chunks_nums]
    assert sum(sums) == len(chunks_nums)
    assert list(itertools.chain(*chunks_nums)) == [len(element) - 2 for element in sentences_tokens.input_ids]

    return sentence_chunks


def cross_encoder_retrieval(input_file, max_sequence_length, process_count, mode, sampling=False,
                            sampling_index_file=None, top_k=100, question_included=True):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device=f'cuda:{process_count}')

    data = list(jsonlines.open(input_file))
    query_text = 'claim_generated_question' if question_included else 'claim_only'

    sorted_claim_url_path = f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/retrieval_chunks/{mode}/sorted_{query_text}_{max_sequence_length}_{mode}_round2_all_process_{process_count}.jsonl'

    # with open('./data/test/further_check_round2.json') as file:
    #    existing_data = json.load(file)
    # existing_indexes = list(existing_data.keys())
    # existing_indexes = sorted([int(idx) for idx in existing_indexes])
    existing_indexes = []
    total_indexes = []
    if sampling and sampling_index_file is not None:
        with open(sampling_index_file) as file:
            sampling_info = json.load(file)
        total_indexes = list(sampling_info.keys())
        total_indexes = sorted([int(idx) for idx in total_indexes])
        total_indexes = [idx for idx in total_indexes if idx not in existing_indexes]
    elif not sampling:
        total_indexes = list(range(len(data)))

    num_per_process = len(total_indexes) // num_processes + 1
    process_idxes = total_indexes[int(process_count * num_per_process):int((process_count + 1) * num_per_process)]
    error_idx = []

    for idx in tqdm(process_idxes):
        print(idx)
        element = data[idx]
        claim = element['claim']
        speaker = element['info']['speaker']
        info = element['info']
        if not sampling:
            generated_questions = element['greedy_questions']
        else:
            generated_questions = element['sampling_questions'][sampling_info[str(idx)]]

        if speaker is not None:
            speaker = info['speaker'] if 'http' not in info['speaker'] else None
        if question_included:
            if speaker is not None:
                queries = [f'{speaker} stated {claim}  {question}' for question in generated_questions]
            else:
                queries = [f'{claim}  {question}' for question in generated_questions]
        else:
            queries = [f'{claim}']

        try:
            if mode == 'train':
                search_data = list(
                    jsonlines.open(
                        f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json"))
            elif mode == 'dev':
                search_data = list(
                    jsonlines.open(
                        f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json"))
            elif mode == 'test':
                search_data = list(
                    jsonlines.open(
                        f"/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/test/{idx}.json"))
            all_scores = []
            all_urls = []
            all_sentence_chunks = []

            for query in queries:
                query_tokens = model.tokenizer(query, return_tensors='pt').input_ids.shape[1]
                max_length = max_sequence_length - query_tokens - 3  # 3 for special tokens cls, sep, sep
                query_chunk_pairs = []
                urls = []
                for item in search_data:
                    if len(item["url2text"]) > 0:
                        url2text = item["url2text"]
                        url = item['url']
                        sentence_chunks = build_sentence_chunking(url2text, model.tokenizer, max_length=max_length)
                        query_chunk_pairs += [(query, " ".join(chunk)) for chunk in sentence_chunks]
                        urls += [url] * len(sentence_chunks)
                scores = model.predict(query_chunk_pairs, batch_size=256, show_progress_bar=False).tolist()
                scores_ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                sorted_scores = [scores[i] for i in scores_ranking]
                sorted_sentence_chunks = [query_chunk_pairs[i][1] for i in scores_ranking]
                sorted_urls = [urls[i] for i in scores_ranking]
                assert len(query_chunk_pairs) == len(sorted_urls) == len(sorted_sentence_chunks) == len(sorted_scores)

                all_sentence_chunks.append(sorted_sentence_chunks[:top_k])
                all_scores.append(sorted_scores[:top_k])
                all_urls.append(sorted_urls[:top_k])

            dump_jsonl(
                [{'claim': claim, 'info': info, 'generated_questions': generated_questions, 'queries': queries,
                  'sentence_chunks': all_sentence_chunks,
                  'scores': all_scores, 'urls': all_urls}], sorted_claim_url_path, append=True)
        except Exception as e:
            print(e)
            error_idx.append(idx)
    with open(f'./data/dev/process_{mode}_round2_all_{process_count}_error_idx.json', 'w') as file:
        json.dump(error_idx, file)


def combine_cross_encoder_retrieval(input_files, reference_file, output_file):
    reference = list(jsonlines.open(reference_file))
    # existing_retrieval = list(jsonlines.open('./data/test/sorted_claim_generated_question_512_tes_part.jsonl'))
    # retrieval_data = existing_retrieval
    retrieval_data = []
    for file in input_files:
        retrieval = list(jsonlines.open(file))
        retrieval_data += retrieval

    retrieval_data.sort(key=lambda x: x['info']['claim_id'])
    finished_idxes = []
    not_finished_idxes = []
    for idx, element in enumerate(reference):
        retrieved = [item for item in retrieval_data if item['info']['claim_id'] == idx]
        if len(retrieved) == 1:
            assert len(retrieved[0]['generated_questions']) == len(retrieved[0]['sentence_chunks'])
            assert retrieved[0]['generated_questions'] == element['generated_questions']
            # dump_jsonl([retrieved[0]], output_file, append=True)
            finished_idxes.append(idx)
        else:
            not_finished_idxes.append(idx)
    print(len(finished_idxes))
    print(len(not_finished_idxes))
    print(not_finished_idxes)
    assert len(finished_idxes) + len(not_finished_idxes) == len(reference)

    '''with open('./data/test/rest_idxes.json', 'w') as file:
        json.dump(not_finished_idxes, file)'''


def check_entailment_chunks(input_file, batch_size, reference_file):
    label_mapping = {'Refuted': 0, 'Not Enough Evidence': 1, 'Supported': 2, 'Conflicting Evidence/Cherrypicking': 3}
    with open(reference_file) as file:
        data = json.load(file)
    reference_claims = [element['claim'] for element in data]
    gold_references_labels = [element['label'] for element in data]
    softmax = torch.nn.Softmax(dim=1)
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')
    retrieval = list(jsonlines.open(input_file))
    gold_labels = []
    predictions = []
    error_idx = []
    idx = 0
    for element in tqdm(retrieval):
        claim = element['claim']
        claim_ref_idx = reference_claims.index(claim)

        chunks = element['sentence_chunks'][:10]
        premise_hypo_pairs = [nli_model.encode(sentence, claim) for sentence in chunks]
        premise_hypo_pairs = [pair[:1024] for pair in premise_hypo_pairs]
        batches = [premise_hypo_pairs[i:i + batch_size] for i in range(0, len(premise_hypo_pairs), batch_size)]
        predicted_labels = torch.tensor([]).to('cuda')
        try:
            for chunk in batches:
                batch = collate_tokens(chunk, pad_idx=1)
                with torch.no_grad():
                    batch_logits = nli_model.predict('mnli', batch)
                batch_probs = softmax(batch_logits)
                batch_label = batch_probs.argmax(dim=1)
                predicted_labels = torch.cat((predicted_labels, batch_label), dim=0)
            predicted_labels = np.array(predicted_labels.tolist()).flatten().tolist()
            predicted_labels = [int(predicted_label) for predicted_label in predicted_labels]
            # print(predicted_labels)
            d = dict(Counter(predicted_labels))
            final_label = max(d, key=d.get)
            predictions.append(final_label)
            gold_label = label_mapping[gold_references_labels[claim_ref_idx]]
            gold_labels.append(gold_label)
            if gold_label == 3:
                print(f'{idx}-{d}')
        except:
            error_idx.append(idx)
            pass
        idx += 1

    print(classification_report(gold_labels, predictions))
    print(confusion_matrix(gold_labels, predictions))
    print(error_idx)


def index_retrieval(input_files, reference, output_file):
    with open(reference) as file:
        reference_data = json.load(file)
    with open('./data/rest_idxes.json') as file:
        rest_index = json.load(file)
    # rest_index = [i for i in range(len(reference_data)) if i not in wo_index]
    num_per_process = len(rest_index) // 8 + 1
    index_split = [rest_index[int(i * num_per_process):int((i + 1) * num_per_process)] for i in range(8)]
    index_split[6] = [i for i in index_split[6] if i not in [2954, 2955, 2956, 2961, 2962]]
    print(index_split)
    for i, input_file in enumerate(input_files):
        process_count = int(input_file.split('process_')[1][0])
        assert process_count == i
        idxes = index_split[i]
        data = list(jsonlines.open(input_file))
        # assert len(idxes)==len(data)
        for j, element in enumerate(data):
            idx = idxes[j]
            claim = element['claim']
            try:
                assert claim == reference_data[idx]['claim']
                print(idx)
            except:
                print(f'error {idx}')
            element['sentence_chunks'] = element['sentence_chunks'][:100]
            element['scores'] = element['scores'][:100]
            element['urls'] = element['urls'][:100]
            element['idx'] = idx
            dump_jsonl([element], output_file, append=True)
        print(i)
        print('#####')


def sort_elements(input_file, output_file):
    retrieval = list(jsonlines.open(input_file))
    retrieval.sort(key=lambda x: x['idx'])
    for element in retrieval:
        dump_jsonl([element], output_file, append=True)


def evaluate_cross_encoder_retrieval(reference_file, retrieval_file, mode):
    with open(reference_file) as file:
        reference = json.load(file)
    retrieval = list(jsonlines.open(retrieval_file))
    retrieval_idx = [element['idx'] for element in retrieval]
    num_included_top_10 = 0
    num_included_top_20 = 0
    num_all_included_top_10 = 0
    num_all_included_top_20 = 0

    total = 0
    num_gold_urls = 0
    included_urls_top_10 = 0
    included_urls_top_20 = 0
    for idx, element in enumerate(reference):
        if idx in retrieval_idx:
            indices = [i for i, x in enumerate(retrieval_idx) if x == idx]
            question_info = element['questions']
            questions = []
            gold_urls = []
            for item in question_info:
                question = item['question']
                questions.append(question)
                gold_urls.append(item['answers'][0]['source_url'])
            gold_urls = [url for url in gold_urls if url != '']
            if mode == 'train':
                url_data = list(jsonlines.open(
                    f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json'))
            elif mode == 'dev':
                url_data = list(jsonlines.open(
                    f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json'))
            urls = [item['url'] for item in url_data]
            url_text = [item['url2text'] for item in url_data]
            existing_gold_urls = []

            for url in gold_urls:
                url_idx = urls.index(url)
                if len(url_text[url_idx]) > 0:
                    existing_gold_urls.append(url)
            if len(existing_gold_urls) > 0:
                total += 1
                num_gold_urls += len(existing_gold_urls)
                queries = [retrieval[index]['query'] for index in indices]

                matching = [1 if questions[index] in queries[index] else 0 for index in range(len(queries))]
                if sum(matching) == len(queries):
                    top_10_retrieved_urls = []
                    top_20_retrieved_urls = []

                    for index in indices:
                        top_10_retrieved_urls.append(retrieval[index]['urls'][:10])
                        top_20_retrieved_urls.append(retrieval[index]['urls'][:20])

                    top_10_included = [1 if existing_gold_urls[i] in top_10_retrieved_urls[i] else 0 for i in
                                       range(len(existing_gold_urls))]
                    top_20_included = [1 if existing_gold_urls[i] in top_20_retrieved_urls[i] else 0 for i in
                                       range(len(existing_gold_urls))]
                    included_urls_top_10 += sum(top_10_included)
                    included_urls_top_20 += sum(top_20_included)
                    if sum(top_10_included) > 0:
                        num_included_top_10 += 1
                    if sum(top_10_included) == len(existing_gold_urls):
                        num_all_included_top_10 += 1
                    if sum(top_20_included) > 0:
                        num_included_top_20 += 1
                    if sum(top_20_included) == len(existing_gold_urls):
                        num_all_included_top_20 += 1
    print(f'{num_included_top_10}/{num_included_top_20}/{total}')
    print(f'{num_all_included_top_10}/{num_all_included_top_20}/{total}')
    print(f'{included_urls_top_10}/{included_urls_top_20}/{num_gold_urls}')
    # 512 dev: 105/149/381; 44/73/381; 145/204/756
    # 256 dev: 105/141/381; 46/65/381; 139/191/756
    # 512 train: 544/722/1833; 299/401/1833; 717/972/3407


def evaluate_cross_encoder_retrieval_claim_only(reference_file, retrieval_file, mode):
    with open(reference_file) as file:
        reference_data = json.load(file)
    retrieval = list(jsonlines.open(retrieval_file))
    retrieval_claims = [element['claim'] for element in retrieval]
    num_included = 0
    total = 0
    idx = 0
    for element in tqdm(reference_data):
        claim = element['claim']
        questions = element['questions']
        gold_urls = [question['answers'][0]['source_url'] for question in questions]
        gold_urls = [url for url in gold_urls if url != '']
        if mode == 'train':
            url_data = list(jsonlines.open(
                f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json'))
        elif mode == 'dev':
            url_data = list(jsonlines.open(
                f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json'))
        urls = [item['url'] for item in url_data]
        url_text = [item['url2text'] for item in url_data]
        existing_gold_urls = []

        for url in gold_urls:
            url_idx = urls.index(url)
            if len(url_text[url_idx]) > 0:
                existing_gold_urls.append(url)
        if len(existing_gold_urls) > 0:
            total += 1
            retrieval_idx = retrieval_claims.index(claim)
            retrieved_urls = retrieval[retrieval_idx]['urls'][:10]
            included = [1 if url in retrieved_urls else 0 for url in existing_gold_urls]
            if sum(included) > 0:
                num_included += 1
        idx += 1

    print(f'{num_included}/{total}')


def integrate_question_retrieval(question_file, retrieval_file, output_file):
    question_data = list(jsonlines.open(question_file))
    retrieval_data = list(jsonlines.open(retrieval_file))
    question_idx = 0
    total_idx = []
    for j, element in enumerate(question_data):
        num_questions = len(element['generated_questions'])
        generated_questions = element['generated_questions']
        retrieval_idxes = [question_idx + i for i in range(num_questions)]
        queries = [retrieval_data[idx]['query'] for idx in retrieval_idxes]
        included_indicator = [1 if generated_questions[i] in queries[i] else 0 for i in range(num_questions)]
        if sum(included_indicator) == num_questions:
            element['top_sentence_chunks'] = [retrieval_data[idx]['sentence_chunks'] for idx in retrieval_idxes]
            element['top_scores'] = [retrieval_data[idx]['scores'] for idx in retrieval_idxes]
            element['top_urls'] = [retrieval_data[idx]['urls'] for idx in retrieval_idxes]
            dump_jsonl([element], output_file, append=True)
        else:
            print(f'################{j}###############')

        question_idx += num_questions
        total_idx += retrieval_idxes
    assert total_idx == list(range(len(retrieval_data)))


def combine_retrieval(input_files, reference_file, output_file):
    # retrieval_data = list(jsonlines.open('./data/test/sorted_claim_generated_question_512_test_round1.jsonl'))
    with open(reference_file) as file:
        reference_data = json.load(file)
    reference_claims = [element['claim'] for element in reference_data]
    retrieval_data = []
    for file in input_files:
        retrieval = list(jsonlines.open(file))
        retrieval_data += retrieval
    print(len(retrieval_data))
    retrieval_data.sort(key=lambda x: x['info']['claim_id'])
    finished_idxes = [element['info']['claim_id'] for element in retrieval_data]
    assert finished_idxes == list(range(500))
    retrieval_claims = [element['claim'] for element in retrieval_data]
    assert retrieval_claims == retrieval_claims

    for element in retrieval_data:
        dump_jsonl([element], output_file, append=True)


if __name__ == "__main__":
    '''num_processes = 1
    cross_encoder_retrieval(
        './data/test/llama3_question_generation_test_greedy_sampling_sub_included_w_dev_processed.jsonl',
        max_sequence_length=512,
        process_count=0, mode='test', sampling=True, sampling_index_file='./data/test/further_check_round1.json',
        top_k=3, question_included=True)'''
    '''integrate_question_retrieval(1
        './data/llama3-instruct-lora_70B_question_generation_dev_greedy_sub_included_processed.jsonl',
        './data/sorted_claim_generated_question_separate_ms-marco-l12_512_dev_top50.jsonl',
        './data/claim_generated_question_separate_ms-marco-l12_512_dev_top50.jsonl'
    )'''
    # evaluate_cross_encoder_retrieval('./data/dev.json',
    #                                 './data/sorted_claim_question_separate_ms-marco-l12_dev_top100.jsonl',
    #                                 mode='dev')
    # sort_elements('./data/sorted_claim_only_ms-marco-l12_512_train_top100.jsonl',
    #              './data/sorted_claim_only_ms-marco-l12_train_top100.jsonl')
    combine_retrieval(
        [
            f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/retrieval_chunks/dev/sorted_claim_generated_question_512_dev_round2_all_process_{i}.jsonl'
            for i in range(4)],
        './data/dev.json',
        './data/dev/sorted_claim_generated_question_512_dev_round2_all.jsonl')
    '''index_retrieval(
        [
            f'/hkfs/work/workspace_haic/scratch/vl8701-llm/checkpoint/averitec/retrieval_chunks/sorted_claim_only_rest_chunk_score_pairs_ms-marco-l12_512_train_process_{i}.jsonl'
            for i in range(8)],
        './data/train.json', './data/sorted_claim_only_ms-marco-l12_512_train_top100.jsonl')'''

    '''with parallel_backend('multiprocessing'):
        num_processes = 4
        Parallel(n_jobs=num_processes)(
            delayed(cross_encoder_retrieval)(
                input_file='./data/dev/llama3_question_generation_dev_greedy_sampling_sub_included_wo_dev_processed.jsonl',
                max_sequence_length=512, process_count=i, mode='dev', sampling=True,
                sampling_index_file='./data/dev/further_check_dev_round2_all_claims.json', top_k=3, question_included=True)
            for
            i in range(num_processes))'''
    '''evaluate_retrieve(input_file='./data/train.json',
                      retrieval_claim_file='./data/sorted_claim_url_pairs_SFR-Embedding-Mistral_4096_wo_normalization_train.jsonl',
                      retrieval_claim_question_file='./data/sorted_claim_question_url_pairs_SFR-Embedding-Mistral_4096_wo_normalization_train.jsonl')'''
