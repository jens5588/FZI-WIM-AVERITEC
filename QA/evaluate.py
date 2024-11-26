import json
import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from fairseq.data.data_utils import collate_tokens
from llama_recipes.utils.inference_utils import dump_jsonl
# from retrieval.retrieve import build_sentences_chunking
from sentence_transformers import CrossEncoder


def check_qa_entailment(qa, chunks, nli_model):
    softmax = torch.nn.Softmax(dim=1)
    premise_hypothesis_pairs = [(chunk, qa) for chunk in chunks]
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in premise_hypothesis_pairs]

    batch = collate_tokens(pair_tokens, pad_idx=1, pad_to_length=1024)
    with torch.no_grad():
        batch_logits = nli_model.predict('mnli', batch)
    batch_probs = softmax(batch_logits)
    batch_labels = batch_probs.argmax(dim=1).tolist()
    if 2 in batch_labels:
        return True
    else:
        return False


def check_statement_entailment(chunk_statement_pairs, nli_model):
    softmax = torch.nn.Softmax(dim=1)
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in chunk_statement_pairs]

    batch = collate_tokens(pair_tokens, pad_idx=1, pad_to_length=1024)
    with torch.no_grad():
        batch_logits = nli_model.predict('mnli', batch)
    batch_probs = softmax(batch_logits)
    print(batch_probs)
    batch_labels = batch_probs.argmax(dim=1).tolist()
    label_indicator = [1 if label == 2 else 0 for label in batch_labels]
    return sum(label_indicator), batch_labels


def qa_factuality_evaluation(retrieval_file, reference_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')

    # gold qa_pairs, retrieval
    # gold question, generated answer rag, retrieval
    with open(reference_file) as file:
        reference = json.load(file)
    retrieval = list(jsonlines.open(retrieval_file))
    retrieval_idx = [element['idx'] for element in retrieval]

    gold_qa = []
    retrieval_chunks = []
    for idx, element in enumerate(reference):
        if idx in retrieval_idx:
            indices = [i for i, x in enumerate(retrieval_idx) if x == idx]
            question_info = element['questions']
            questions = []
            for item in question_info:
                question = item['question']
                questions.append(question)
                if '?' not in question[-3:]:
                    question += '?'
                answer = item['answers'][0]['answer']
                qa_pair = f'{question} {answer}'
                gold_qa.append(qa_pair)
            queries = [retrieval[index]['query'] for index in indices]
            matching = [1 if questions[index] in queries[index] else 0 for index in range(len(queries))]
            if sum(matching) == len(queries):
                for index in indices:
                    retrieval_chunks.append(retrieval[index]['sentence_chunks'][:10])
            else:
                print('not matching')

        else:
            print(idx)

    assert len(retrieval_chunks) == len(gold_qa)
    entailed = 0
    error_idx = []
    for idx in tqdm(range(len(gold_qa))):
        try:
            label = check_qa_entailment(gold_qa[idx], retrieval_chunks[idx], nli_model)
            if label:
                entailed += 1
        except:
            error_idx.append(idx)
    print(entailed)
    # dev 512 398
    # dev 256 279
    print(error_idx)


def evaluate_statement_factuality(statement_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')

    statement_data = list(jsonlines.open(statement_file))
    num_entailed = 0
    overall_entiled = 0
    labels = []
    error_idx = []
    for idx in tqdm(range(0, 100, 5)):
        chunk_statement_pairs = []
        for i in range(idx, idx + 5):
            if 'qa_statement' in statement_data[i]:
                statement = statement_data[i]['qa_statement']
                chunk = statement_data[i]['chunk']
                chunk_statement_pairs.append((chunk, statement))
        if len(chunk_statement_pairs) > 0:
            try:
                label_sum, batch_labels = check_statement_entailment(chunk_statement_pairs, nli_model)
                overall_entiled += label_sum
                labels.append(batch_labels)
                if label_sum > 0:
                    num_entailed += 1
            except:
                error_idx.append(idx)
    print(num_entailed)
    print(overall_entiled)
    print(error_idx)
    # print(labels)


def evaluate_qa(reference_file, generation_file):
    with open(reference_file) as file:
        data = json.load(file)
    generation = list(jsonlines.open(generation_file))
    idx = 0
    for element in data:
        claim = element['claim']
        label = element['label']
        questions = element['questions']
        for item in questions:
            question = item['question']
            answers = item['answers']
            answers = [answer['answer'] for answer in answers]
            answer = " ".join(answers)


def truncation(prem_hypo_pair, nli_model):  # truncation strategy only first
    premise, hypothesis = prem_hypo_pair
    num_hypothesis_token = len(nli_model.encode(hypothesis).tolist())
    premise = nli_model.decode(nli_model.encode(premise)[:(1024 - num_hypothesis_token - 10)])

    return nli_model.encode(premise, hypothesis)


def select_best_response(question, response, sentence_chunks, nli_model, top_k):
    qa_pairs = [f'{question} {item}' for item in response]
    prem_hypo_pairs = []
    for qa_pair in qa_pairs:
        for chunk in sentence_chunks:
            prem_hypo_pairs.append((chunk, qa_pair))
    assert len(prem_hypo_pairs) == int(len(sentence_chunks) * len(qa_pairs))
    softmax = torch.nn.Softmax(dim=1)
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in prem_hypo_pairs]
    max_length_exceed_indexes = [i for i in range(len(pair_tokens)) if len(pair_tokens[i]) > 1024]
    for index in max_length_exceed_indexes:
        truncated_tokens = truncation((prem_hypo_pairs[index][0], prem_hypo_pairs[index][1]), nli_model)
        pair_tokens[index] = truncated_tokens[:1024]

    batch = collate_tokens(pair_tokens, pad_idx=1, pad_to_length=1024)
    with torch.no_grad():
        batch_logits = nli_model.predict('mnli', batch)
    batch_probs = softmax(batch_logits)
    scores = []
    for i in range(top_k):
        entailment_scores = 0
        for j in range(int(i * top_k), int((i + 1)) * top_k):
            entailment_scores += batch_probs[j][2]
        scores.append(entailment_scores.cpu())
    max_score_idx = np.argmax(np.asarray(scores))
    return max_score_idx


def answer_question_selection(input_file, reference_file, output_file, top_k):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')

    generation = list(jsonlines.open(input_file))
    with open(reference_file) as file:
        reference_data = json.load(file)

    for idx, element in enumerate(generation):
        claim = element['claim']
        assert claim == reference_data[idx]['claim']
        generated_questions = element['generated_questions']
        top_sentence_chunks = element['top_sentence_chunks']
        responses = element['responses']
        top_urls = element['top_urls']
        num_questions = len(generated_questions)
        best_responses = []
        for i in range(len(generated_questions)):
            question = generated_questions[i]
            response = [item[0] for item in responses[i]]
            # print(response)
            sentence_chunks = top_sentence_chunks[i][:len(response)]
            urls = top_urls[i][:len(response)]
            best_response_idx = select_best_response(question, response, sentence_chunks, nli_model, top_k=5)
            best_response = response[best_response_idx]
            best_response_url = urls[best_response_idx]
            best_responses.append((best_response, best_response_url))
        element['best_responses'] = best_responses
        dump_jsonl([element], output_file, append=True)


def combined_file(file1, file2, file3, reference_file, output_file):
    generation1 = list(jsonlines.open(file1))
    generation2 = list(jsonlines.open(file2))
    generation3 = list(jsonlines.open(file3))
    generation = generation1 + generation2 + generation3
    with open(reference_file) as file:
        reference_data = json.load(file)
    for idx, element in enumerate(generation):
        claim = element['claim']
        assert claim == reference_data[idx]['claim']
    for element in generation:
        dump_jsonl([element], output_file, append=True)


def select_best_statements(nli_model, num_questions, top_sentence_chunks, greedy_statements, sampling_statements,
                           num_sampling=10):
    softmax = torch.nn.Softmax(dim=1)
    for i in range(num_questions):
        total_statements = sampling_statements[i] + [greedy_statements[i]]
        assert len(total_statements) == num_sampling + 1
        sentence_chunks = top_sentence_chunks[i]
        prem_hypo_pairs = []
        for statement in total_statements:
            for chunk in sentence_chunks:
                prem_hypo_pairs.append((statement, chunk))
        pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in prem_hypo_pairs]
        max_length_exceed_indexes = [i for i in range(len(pair_tokens)) if len(pair_tokens[i]) > 1024]
        for index in max_length_exceed_indexes:
            truncated_tokens = truncation((prem_hypo_pairs[index][0], prem_hypo_pairs[index][1]), nli_model)
            pair_tokens[index] = truncated_tokens[:1024]

        batch = collate_tokens(pair_tokens, pad_idx=1, pad_to_length=1024)
        with torch.no_grad():
            batch_logits = nli_model.predict('mnli', batch)
        batch_probs = softmax(batch_logits)
        scores = []
        num_statements = len(total_statements)
        num_chunks = len(sentence_chunks)
        for k in range(num_statements):
            entailment_scores = 0
            for j in range(int(k * num_chunks), int((k + 1)) * num_chunks):
                try:
                    entailment_scores += batch_probs[j][2]
                except:
                    print(f'Error {j} num_statements {num_statements} num chunks {num_chunks}')
            scores.append(entailment_scores.cpu())
        print(f'index {i}')
        print(scores)
        max_score_idx = np.argmax(np.asarray(scores))
        print(f'max score index: {max_score_idx}')


def answer_selection_statements(input_file, reference_file, output_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')

    generation = list(jsonlines.open(input_file))
    with open(reference_file) as file:
        data = json.load(file)
    for idx, element in enumerate(generation[:20]):
        print(f'idx {idx}')
        claim = element['claim']
        assert claim == data[idx]['claim']
        generated_questions = element['generated_questions']
        top_sentence_chunks = element['top_sentence_chunks']
        greedy_statements = [item.replace('\n\n', '') for item in element['greedy_statements']]
        sampling_statements = element['sampling_statements']
        for i in range(len(sampling_statements)):
            for j in range(len(sampling_statements[i])):
                sampling_statements[i][j] = sampling_statements[i][j].replace('\n\n', '')

        assert len(generated_questions) == len(top_sentence_chunks) == len(greedy_statements) == len(
            sampling_statements)
        select_best_statements(nli_model, len(generated_questions), top_sentence_chunks, greedy_statements,
                               sampling_statements)
        print('###########################')


def combined_gold_qa_statement_retrieval(statement_file, retrieval_file, output_file, mode):
    if mode == 'train':
        error_idxes = [2954, 2955, 2956, 2961, 2962]
    else:
        error_idxes = []
    statement_data = list(jsonlines.open(statement_file))

    retrieval_data = list(jsonlines.open(retrieval_file))
    count = 0
    for idx, element in enumerate(statement_data):
        if idx not in error_idxes:
            claim = element['claim']
            questions = [item['question'] for item in element['questions']]
            queries = [f'{claim}  {question}' for question in questions]
            for i in range(len(questions)):
                query = queries[i]
                assert query == retrieval_data[count]['query']
                element['questions'][i]['sentence_chunks'] = retrieval_data[count]['sentence_chunks']
                element['questions'][i]['scores'] = retrieval_data[count]['scores']
                element['questions'][i]['urls'] = retrieval_data[count]['urls']
                count += 1
            dump_jsonl([element], output_file, append=True)

        else:
            dump_jsonl([element], output_file, append=True)


def check_statement_chunks_entailment(prem_hypo_pairs, nli_model, batch_size=8):
    softmax = torch.nn.Softmax(dim=1)
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in prem_hypo_pairs]
    max_length_exceed_indexes = [i for i in range(len(pair_tokens)) if len(pair_tokens[i]) > 1024]
    for index in max_length_exceed_indexes:
        truncated_tokens = truncation((prem_hypo_pairs[index][0], prem_hypo_pairs[index][1]), nli_model)
        pair_tokens[index] = truncated_tokens[:1024]
    chunks = [pair_tokens[x:x + batch_size] for x in range(0, len(pair_tokens), batch_size)]
    labels = torch.tensor([]).to('cuda')

    for chunk in chunks:
        batch = collate_tokens(chunk, pad_idx=1, pad_to_length=1024)
        with torch.no_grad():
            chunk_logits = nli_model.predict('mnli', batch)
        chunk_probs = softmax(chunk_logits)
        chunk_label = chunk_probs.argmax(dim=1)
        labels = torch.cat((labels, chunk_label), dim=0)
    labels = labels.tolist()
    labels = [int(label) for label in labels]
    num_entailments = sum([1 if label == 2 else 0 for label in labels])

    return num_entailments


def select_neutral_chunks(prem_hypo_pairs, nli_model, batch_size=8, top_k=3):
    softmax = torch.nn.Softmax(dim=1)
    pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in prem_hypo_pairs]
    max_length_exceed_indexes = [i for i in range(len(pair_tokens)) if len(pair_tokens[i]) > 1024]
    for index in max_length_exceed_indexes:
        truncated_tokens = truncation((prem_hypo_pairs[index][0], prem_hypo_pairs[index][1]), nli_model)
        pair_tokens[index] = truncated_tokens[:1024]
    chunks = [pair_tokens[x:x + batch_size] for x in range(0, len(pair_tokens), batch_size)]
    probs = torch.tensor([]).to(nli_model.device)

    for chunk in chunks:
        batch = collate_tokens(chunk, pad_idx=1, pad_to_length=1024)
        with torch.no_grad():
            chunk_logits = nli_model.predict('mnli', batch)
        chunk_probs = softmax(chunk_logits)
        probs = torch.cat((probs, chunk_probs), dim=0)
    neutral_probs = [element[1] for element in probs.tolist()]
    top_idxes = sorted(range(len(neutral_probs)), key=lambda k: neutral_probs[k], reverse=True)[:top_k]
    top_probs = [neutral_probs[idx] for idx in top_idxes]
    if all(prob > 0.5 for prob in top_probs):
        return [prem_hypo_pairs[idx][0] for idx in top_idxes]
    else:
        return None


def statement_sentence_chunks_entailment_check(statement_retrieval_file, output_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')
    num_statements = 0
    num_entailments = 0
    entailed = 0
    statement_retrieval_data = list(jsonlines.open(statement_retrieval_file))
    error_idx = []
    for idx, element in enumerate(statement_retrieval_data):
        questions = element['questions']
        for q_idx, part in enumerate(questions):
            if 'sentence_chunks' in part:
                sentence_chunks = part['sentence_chunks'][:3]
                for a_idx, item in enumerate(part['answers']):
                    statement = item['statement']
                    num_entailment = check_statement_chunks_entailment(
                        [(chunk, statement) for chunk in sentence_chunks],
                        nli_model)
                    num_entailments += num_entailment
                    if num_entailment > 0:
                        entailed += 1
                        if item['answer_type'] == 'Boolean':
                            answer = f'{item["answer"]}, {item["boolean_explanation"]}'
                        else:
                            answer = item['answer']
                        dump_jsonl([{'claim': element['claim'], 'label': element['label'], 'idx': idx,
                                     'question': part['question'], 'question_idx': q_idx, 'answer': answer,
                                     'answer_idx': a_idx, 'statement': statement,
                                     'sentence_chunks': sentence_chunks}, ], output_file,
                                   append=True)
                    num_statements += 1
            else:
                error_idx.append(idx)
    print(error_idx)


def get_top_chunks_gold_document(nli_model, encoding_model, max_sequence_length, claim, question, statement, text,
                                 top_k=3):
    query = f'{claim} {question}'
    query_tokens = encoding_model.tokenizer(query, return_tensors='pt').input_ids.shape[1]
    max_length = max_sequence_length - query_tokens - 3
    sentence_chunks = build_sentences_chunking(text, encoding_model.tokenizer, max_length=max_length)

    query_chunk_pairs = [(query, " ".join(chunk)) for chunk in sentence_chunks]
    scores = encoding_model.predict(query_chunk_pairs, batch_size=16, show_progress_bar=False).tolist()
    scores_ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sorted_sentences_chunks = [query_chunk_pairs[i][1] for i in scores_ranking]
    num_entailments = check_statement_chunks_entailment([(chunk, statement) for chunk in sorted_sentences_chunks],
                                                        nli_model)
    if len(sorted_sentences_chunks) <= top_k:
        return sorted_sentences_chunks
    else:
        if num_entailments > 0:
            return sorted_sentences_chunks[:top_k]
        else:
            return None


def check_gold_document_entailment(statement_retrieval_file, output_file):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to('cuda')
    encoding_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512,
                                  device='cuda')
    statement_retrieval_data = list(jsonlines.open(statement_retrieval_file))
    mode = 'train' if 'train' in statement_retrieval_file else 'dev'
    not_entailed = 0
    idx = 0
    total_check = 0
    for element in tqdm(statement_retrieval_data):
        questions = element['questions']
        if mode == 'train':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm2/checkpoint/averitec/data_store/knowledge_store/train/{idx}.json"))
        elif mode == 'dev':
            search_data = list(
                jsonlines.open(
                    f"/hkfs/work/workspace_haic/scratch/vl8701-llm2/checkpoint/averitec/data_store/knowledge_store/output_dev/{idx}.json"))
        valid_url_texts = {}
        for item in search_data:
            if len(item["url2text"]) > 0:
                valid_url_texts[item['url']] = item['url2text']
        for q_idx, part in enumerate(questions):
            answers = part['answers']
            for a_idx, item in enumerate(answers):
                statement = item['statement']
                source_url = item['source_url']
                if source_url in valid_url_texts.keys():
                    top_sentence_chunks = get_top_chunks_gold_document(nli_model=nli_model,
                                                                       encoding_model=encoding_model,
                                                                       max_sequence_length=512, claim=element['claim'],
                                                                       question=part['question'], statement=statement,
                                                                       text=valid_url_texts[source_url], top_k=3)
                    if item['answer_type'] == 'Boolean':
                        answer = f'{item["answer"]}, {item["boolean_explanation"]}'
                    else:
                        answer = item['answer']
                    if top_sentence_chunks is not None:
                        dump_jsonl([{'claim': element['claim'], 'label': element['label'], 'idx': idx,
                                     'question': part['question'], 'question_idx': q_idx, 'answer': answer,
                                     'answer_idx': a_idx, 'statement': statement,
                                     'sentence_chunks': top_sentence_chunks}, ], output_file,
                                   append=True)
                    else:
                        not_entailed += 1
                    total_check += 1

        idx += 1
    print(f'{not_entailed}/{total_check}')


def generate_neutral_examples(gold_entailment_file, retrieval_entailment_file, retrieval_statement_file,
                              output_file, process_count):
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    nli_model.to(f'cuda:{process_count}')

    retrieval_statement_data = list(jsonlines.open(retrieval_statement_file))
    total_pairs = []
    for idx, element in enumerate(retrieval_statement_data):
        for q_idx, question in enumerate(element['questions']):
            for a_idx, answer in enumerate(question['answers']):
                total_pairs.append((idx, q_idx, a_idx))

    gold_entailment_data = list(jsonlines.open(gold_entailment_file))
    retrieval_entailment_data = list(jsonlines.open(retrieval_entailment_file))
    gold_pairs = [(element['idx'], element['question_idx'], element['answer_idx']) for element in gold_entailment_data]
    retrieval_pairs = [(element['idx'], element['question_idx'], element['answer_idx']) for element in
                       retrieval_entailment_data]
    existing_pairs = list(set(gold_pairs + retrieval_pairs))
    missing_pairs = [pair for pair in total_pairs if pair not in existing_pairs]

    num_per_process = len(missing_pairs) // num_processes + 1
    process_idxes = missing_pairs[
                    int(process_count * num_per_process):int((process_count + 1) * num_per_process)]

    for pair in tqdm(process_idxes):
        claim_idx, question_idx, answer_idx = pair
        claim = retrieval_statement_data[claim_idx]['claim']
        label = retrieval_statement_data[claim_idx]['label']

        question = retrieval_statement_data[claim_idx]['questions'][question_idx]['question']
        statement = retrieval_statement_data[claim_idx]['questions'][question_idx]['answers'][answer_idx]['statement']
        answer = retrieval_statement_data[claim_idx]['questions'][question_idx]['answers'][answer_idx]['answer']
        if 'sentence_chunks' in retrieval_statement_data[claim_idx]['questions'][question_idx]:
            sentence_chunks = retrieval_statement_data[claim_idx]['questions'][question_idx]['sentence_chunks'][:50]
            prem_hypo_pairs = [(chunk, statement) for chunk in sentence_chunks]
            top_sentence_chunks = select_neutral_chunks(prem_hypo_pairs, nli_model)
            if top_sentence_chunks is not None:
                dump_jsonl([{'claim': claim, 'label': label, 'idx': claim_idx,
                             'question': question, 'question_idx': question_idx, 'answer': answer,
                             'answer_idx': answer_idx, 'statement': statement,
                             'sentence_chunks': top_sentence_chunks}, ], output_file, append=True)


def evaluate_greedy_sampling_proportion(entailment_check_file):
    generation = list(jsonlines.open(entailment_check_file))
    num_greedy = 0
    num_sampling = 0
    num_no_answer = 0
    num_questions = 0
    for element in generation:
        num_questions += len(element['generated_questions'])
        selected_responses = element['selected_responses']
        checked_responses = element['checked_responses']
        assert len(selected_responses) == len(checked_responses)
        for i in range(len(checked_responses)):
            checked_response = checked_responses[i][0]
            selected_response = selected_responses[i]
            if checked_response is not None:
                if checked_response == selected_response[0]:
                    num_greedy += 1
                else:
                    num_sampling += 1
            else:
                num_no_answer += 1
    print(f'num questions: {num_questions}')
    print(f'num sampling: {num_sampling}')
    print(f'num greedy: {num_greedy}')
    print(f'num no answers: {num_no_answer}')


def combine_question():
    greedy = list(jsonlines.open('./data/dev/sorted_claim_generated_question_512_dev_greedy.jsonl'))
    round1 = list(jsonlines.open('./data/dev/sorted_claim_generated_question_512_dev_round1_all.jsonl'))
    round2 = list(jsonlines.open('./data/dev/sorted_claim_generated_question_512_dev_round2_all.jsonl'))
    all_questions = greedy + round1 + round2
    for element in all_questions:
        dump_jsonl([element], './data/dev/sorted_claim_generated_question_512_dev_all_rounds.jsonl', append=True)


if __name__ == '__main__':
    combine_question()
    # evaluate_greedy_sampling_proportion(
    #    './data/test/qa_generated_questions_combined_test_top3_statements_check_round2.jsonl')
    '''with parallel_backend('multiprocessing'):
        num_processes = 2
        Parallel(n_jobs=num_processes)(
            delayed(generate_neutral_examples)('./data/dev_goldqa_statement_gold_doc_entailed.jsonl',
                                               './data/dev_gold_qa_top3_retrieval_entailed.jsonl',
                                               './data/dev_goldqa_statement_retrieval.jsonl',
                                               f'./data/dev_gold_qa_top3_retrieval_neutral_process{i}.jsonl',
                                               process_count=i) for i in range(num_processes))'''
    # generate_neutral_examples('./data/train_goldqa_statement_gold_doc_entailed.jsonl',
    #                          './data/train_gold_qa_top3_retrieval_entailed.jsonl',
    #                          './data/train_goldqa_statement_retrieval.jsonl',
    #                         './data/train_gold_qa_top3_retrieval_neutral.jsonl')
    # check_gold_document_entailment('./data/train_goldqa_statement_retrieval.jsonl',
    #                               './data/train_goldqa_statement_gold_doc_entailed.jsonl')
    # statement_sentence_chunks_entailment_check('./data/train_goldqa_statement_retrieval.jsonl',
    #                                           './data/train_gold_qa_top3_retrieval_entailed.jsonl')
    # statement_sentence_chunks_entailment_check('./data/train_goldqa_statement_retrieval.jsonl')
    # combined_gold_qa_statement_retrieval('./data/dev_goldqa_statement.jsonl',
    #                                     './data/sorted_claim_question_separate_ms-marco-l12_dev_top100.jsonl',
    #                                     './data/dev_goldqa_statement_retrieval.jsonl',
    #                                     'dev')
    # combined_file('./data/train_goldqa_statement_0.jsonl', './data/train_goldqa_statement_1000.jsonl',
    #              './data/train_goldqa_statement_2000.jsonl', './data/train.json',
    #              './data/train_goldqa_statement.jsonl')
    # qa_factuality_evaluation('./data/sorted_claim_question_separate_ms-marco-l12_256_dev_top100.jsonl',
    #                         './data/dev.json')
    # evaluate_statement_factuality('./data/qa_separate_dev_top5_statements.jsonl')
    # answer_selection('./data/qa_generated_questions_separate_dev_top5.jsonl', './data/dev.json',
    #                 './data/qa_generated_questions_separate_dev_top5_response_selection.jsonl', top_k=5)
    # answer_selection_statements('./data/qa_generated_questions_combined_dev_top3_greedy_sampling_statements_0.jsonl',
    #                            './data/dev.json', None)
