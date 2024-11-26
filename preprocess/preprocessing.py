import json
import random
import pandas as pd
import jsonlines
import spacy
import random
from collections import Counter
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

nlp = spacy.load('en_core_web_sm')


def ner(text):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    return ents


def create_question_instruction_data(input_file):
    random.seed(123)
    output_data = []
    with open(input_file) as file:
        data = json.load(file)
    num_data = len(data)
    print(num_data)
    for element in data:
        claim = element['claim']
        speaker = element['speaker'] if element['speaker'] is not None else 'Unknown'
        questions = [item['question'] for item in element['questions']]
        numeration_question = zip(list(range(1, len(questions) + 1)), questions)
        question_formulation = ''
        for item in numeration_question:
            if '?' not in item[1][-3:]:
                question_formulation += f'{item[0]}: {item[1]}?\n'
            else:
                question_formulation += f'{item[0]}: {item[1]}\n'
        question_formulation = question_formulation[:-1]

        instruction = f"""<|begin_of_text|> You are a fact-checker and your task is to generate relevant questions for verifying the following claim.\nClaimer: {speaker}\nClaim:\n{claim}\nQuestions:"""
        output = f"""{question_formulation} <|end_of_text|>"""
        output_data.append({'instruction': instruction, 'output': output})
    train_index = random.sample(list(range(len(data))), int(len(data) * 0.8))
    dev_index = [i for i in range(len(data)) if i not in train_index]
    train_data = [output_data[idx] for idx in train_index]
    dev_data = [output_data[idx] for idx in dev_index]
    with open('./data/instruction_train.json', 'w') as file:
        json.dump({'data': train_data}, file)
    with open('./data/instruction_test.json', 'w') as file:
        json.dump({'data': dev_data}, file)


def include_qa_briefs():
    processed_data = []
    processed_claims = []
    error_claims = []
    train = pd.read_csv('./data/briefs_train.csv', encoding='latin-1')
    dev = pd.read_csv('./data/briefs_valid.csv', encoding='latin-1')
    test = pd.read_csv('./data/briefs_test.csv', encoding='latin-1')
    df = pd.concat([train, dev, test], ignore_index=True)
    for i in range(df.shape[0]):
        if not (pd.isna(df.loc[i, 'claim_reviewed']) or pd.isna(df.loc[i, 'question'])):
            claim = df.loc[i, 'claim_reviewed'].encode('latin1').decode('unicode-escape').encode('latin1').decode(
                'utf8')
            author = df.loc[i, 'item_reviewed_author_name']
            date_published = df.loc[i, 'item_reviewed_date_published']
            try:
                question = df.loc[i, 'question'].encode('latin1').decode('unicode-escape').encode('latin1').decode(
                    'utf8')
                if claim not in processed_claims:
                    processed_data.append(
                        {'claim': claim, 'author': author, 'date': date_published, 'question': [question]})
                    processed_claims.append(claim)
                else:
                    claim_index = processed_claims.index(claim)
                    assert processed_data[claim_index]['claim'] == claim
                    processed_data[claim_index]['question'].append(question)
            except:
                if claim not in error_claims:
                    error_claims.append(claim)

    assert len(processed_claims) == len(processed_data)
    print(f'processed claims: {len(processed_claims)}')
    print(f'error claims: {len(error_claims)}')
    intersected_claims = list(set(processed_claims).intersection(set(error_claims)))
    print(f'num intersected claims: {len(intersected_claims)}')
    intersected_claim_idxes = [processed_claims.index(claim) for claim in intersected_claims]
    selected_idxes = [i for i in range(len(processed_data)) if i not in intersected_claim_idxes]
    # drop claims with decoding problem
    processed_data = [processed_data[i] for i in selected_idxes]
    processed_claims = [processed_claims[i] for i in selected_idxes]
    print(f'num processed data: {len(processed_data)}')
    print(f'num processed claims: {len(processed_claims)}')
    df = df.loc[df['claim_reviewed'].notna()]
    all_claims = df['claim_reviewed'].apply(
        lambda x: x.encode('latin1').decode('unicode-escape').encode('latin1').decode(
            'utf8')).tolist()
    frequency_dict = dict(Counter(all_claims))
    for idx, item in enumerate(processed_data):
        claim = item['claim']
        if len(item['question']) != frequency_dict[claim]:
            print(idx)
    with open('./data/qa_briefs.json', 'w') as file:
        json.dump(processed_data, file)


def analysis():
    with open('./data/qa_briefs.json') as file:
        qa_briefs = json.load(file)
    with open('./data/train.json') as file:
        averitec_train = json.load(file)
    with open('./data/dev.json') as file:
        averitec_dev = json.load(file)
    beginning = []
    qa_speakers = []
    averitec_speakers = []

    for element in qa_briefs:
        beginning.append(element['claim'].split(' ')[0])
        # qa_speakers.append(str(element['author']) if str(element['author']) != 'nan' else 'Unknown')
        qa_speakers.append(element['author'])
    for element in averitec_train:
        averitec_speakers.append(element['speaker'] if element['speaker'] is not None else 'Unknown')

    from collections import Counter
    d_be = dict(Counter(beginning))
    d_qa_sp = dict(Counter(qa_speakers))
    d_averitec_sp = dict(Counter(averitec_speakers))
    # print({k: v for k, v in sorted(d_be.items(), key=lambda item: item[1], reverse=True)})
    print({k: v for k, v in sorted(d_qa_sp.items(), key=lambda item: item[1], reverse=True)})


def combine_sub_questions():
    train = list(jsonlines.open('./data/train-all.jsonl'))
    dev = list(jsonlines.open('./data/dev-all.jsonl'))
    test = list(jsonlines.open('./data/test-all.jsonl'))
    combined = train + dev + test
    for element in combined:
        venue = element['venue']
    with open('./data/sub_question.json', 'w') as file:
        json.dump(combined, file)


def create_instruction_data(claimer, claim, claim_date, questions):
    prompt = f"""<|begin_of_text|> You are a fact-checker and your task is to generate critical questions for verifying the following claim.\nClaim date: {claim_date}\nClaimer: {claimer}\nClaim: {claim}\nQuestions: """
    print(prompt)
    processed_questions = []
    for question in questions:
        if '?' not in question[-3:]:
            processed_questions.append(f'{question}?')
        else:
            processed_questions.append(question)
    output = (" ".join(processed_questions) + " <|end_of_text|>").replace('\n', " ")
    instruction_output = {'prompt': prompt, 'output': output}

    return instruction_output


def process_qa_briefs_date(claim_date):
    if type(claim_date) == str:
        month, day, year = claim_date.split('/')
        return f'{day}-{month}-20{year}'
    else:
        return 'Unknown'


def process_sub_question_date(claim_date):
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                     'September': 9, 'October': 10, 'November': 11, 'December': 12}

    month, day, year = claim_date.split('on ')[1].split('in')[0].replace(',', '').strip().split(' ')
    return f'{day}-{month_mapping[month]}-{year}'


def process_averitec_date(claim_date):
    if claim_date is None:
        return 'Unknown'
    else:
        day, month, year = claim_date.split('-')
        if year.startswith('19'):
            return 'Unknown'
        else:
            return claim_date


def combine_all_dataset():
    with open('./data/sub_question.json') as file:
        sub_question = json.load(file)
    with open('./data/train.json') as file:
        averitec_train = json.load(file)
    with open('./data/dev.json') as file:
        averitec_dev = json.load(file)
    qa_briefs_output = []
    sub_question_output = []
    averitec_train_output = []
    averitec_dev_output = []

    for element in sub_question:
        claimer = element['person']
        claim = element['claim']
        claim_date = process_sub_question_date(element['venue'])
        questions = element['annotations'][0]['questions']
        sub_question_output.append(create_instruction_data(claimer, claim, claim_date, questions))

    for element in averitec_train:
        claimer = element['speaker'] if element['speaker'] is not None else 'Unknown'
        claim = element['claim']
        claim_date = process_averitec_date(element['claim_date'])
        questions = [item['question'] for item in element['questions']]
        averitec_train_output.append(create_instruction_data(claimer, claim, claim_date, questions))

    for element in averitec_dev:
        claimer = element['speaker'] if element['speaker'] is not None else 'Unknown'
        claim = element['claim']
        claim_date = process_averitec_date(element['claim_date'])
        questions = [item['question'] for item in element['questions']]
        averitec_dev_output.append(create_instruction_data(claimer, claim, claim_date, questions))

    # qa_briefs_train, qa_briefs_test = train_test_split(qa_briefs_output, test_size=0.2, random_state=42)
    # sub_question_train, sub_question_test = train_test_split(sub_question_output, test_size=0.2, random_state=42)
    # averitec_train, averitec_test = train_test_split(averitec_output, test_size=0.2, random_state=42)
    # train = qa_briefs_train + sub_question_train + averitec_train
    sub_included_train = sub_question_output + averitec_train_output
    only_averitec_train = averitec_train_output
    sub_included_train_combined = sub_question_output + averitec_train_output + averitec_dev_output
    only_averitec_train_combined = averitec_train_output + averitec_dev_output
    # train = averitec_train_output
    test = averitec_dev_output
    # test = qa_briefs_test + sub_question_test + averitec_test
    with open('./data/instruction_question_generation_sub_included_train.json', 'w') as file:
        json.dump({'data': sub_included_train}, file)
    with open('./data/instruction_question_generation_sub_included_train_combined.json', 'w') as file:
        json.dump({'data': sub_included_train_combined}, file)
    with open('./data/instruction_question_generation_only_averitec_train.json', 'w') as file:
        json.dump({'data': only_averitec_train}, file)
    with open('./data/instruction_question_generation_only_averitec_train_combined.json', 'w') as file:
        json.dump({'data': only_averitec_train_combined}, file)
    with open('./data/instruction_question_generation_sub_included_test.json', 'w') as file:
        json.dump({'data': test}, file)


def create_verification_dataset(input_file, output_file):
    instruction = f'''<|begin_of_text|> Your task is to verify the claims based on the context information in format of question answer pairs. Verify the claim with justification using the following labels: Supported, Refuted, Not Enough Evidence, Conflicting Evidence/Cherrypicking.\n\n'''
    with open(input_file) as file:
        data = json.load(file)
    processed_examples = []
    for element in data:
        claim = element['claim']
        label = element['label']
        justification = element['justification']
        questions = element['questions']
        qa_pairs = []
        for item in questions:
            question = item['question']
            answers = " ".join([answer['answer'] for answer in item['answers']]).replace('\n\n', '\n')
            qa_pairs.append((question, answers))
        qa_context = ''
        for idx, pair in enumerate(qa_pairs):
            qa_context += f'Question {idx + 1}: {pair[0]}\nAnswer {idx + 1}: {pair[1]}\n'
        prompt = f'{instruction}Claim: {claim}\n{qa_context}'
        output = f'Justification: {justification}\nLabel: {label} <|end_of_text|>'
        processed_examples.append({'prompt': prompt, 'output': output})
    with open(output_file, 'w') as file:
        json.dump({'data': processed_examples}, file)


def create_verification_dataset_source(input_file, output_file):
    instruction = f'''<|begin_of_text|> Your task is to verify the claims based on the context information in format of question answer pairs. Verify the claim with justification using the following labels: Supported, Refuted, Not Enough Evidence, Conflicting Evidence/Cherrypicking.\n\n'''
    with open(input_file) as file:
        data = json.load(file)
    processed_examples = []
    for element in data:
        claim = element['claim']
        label = element['label']
        speaker = element['speaker'] if element['speaker'] is not None else 'Unknown'
        claim_date = process_averitec_date(element['claim_date'])
        reporting_source = element['reporting_source'] if element['reporting_source'] is not None else 'Unknown'
        justification = element['justification']
        questions = element['questions']
        qa_pairs = []
        for item in questions:
            question = item['question']
            answers = " ".join([answer['answer'] for answer in item['answers']]).replace('\n\n', '\n')
            qa_pairs.append((question, answers))
        qa_context = ''
        for idx, pair in enumerate(qa_pairs):
            qa_context += f'Question {idx + 1}: {pair[0]}\nAnswer {idx + 1}: {pair[1]}\n'
        prompt = f'{instruction}Source: {reporting_source}\nClaim data: {claim_date}\nClaimer: {speaker}\nClaim: {claim}\n{qa_context}'
        print(prompt)
        output = f'Justification: {justification}\nLabel: {label} <|end_of_text|>'
        processed_examples.append({'prompt': prompt, 'output': output})
    with open(output_file, 'w') as file:
        json.dump({'data': processed_examples}, file)


def get_formatted_input(messages, context):
    system = "<|begin_of_text|> System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
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


def filter_neutral_data(neutral_data, num_samples, seed=123):
    random.seed(seed)
    selected = []
    selected_idx = []
    filtered_idx = []
    for idx, element in enumerate(neutral_data):
        sentence_chunks = element['sentence_chunks']
        contains_error = False
        for item in sentence_chunks:
            if 'Something went wrong.' in item:
                contains_error = True
                break
        if contains_error:
            filtered_idx.append(idx)
        answer = element['answer'].lower()
        # filter out boolean questions
        if 'yes' in answer or ('no ' in answer and 'no answer could be found' not in answer):
            filtered_idx.append(idx)
    neutral_data = [neutral_data[i] for i in range(len(neutral_data)) if i not in filtered_idx]

    for idx, element in enumerate(neutral_data):
        if 'no answer could be found' in element['answer'].lower():
            selected.append(element)
            selected_idx.append(idx)
    rest_idx = [i for i in range(len(neutral_data)) if i not in selected_idx]
    if num_samples - len(selected_idx) > 0:
        selected_idx += random.sample(rest_idx, num_samples - len(selected_idx))
    selected_data = [neutral_data[idx] for idx in selected_idx]
    return selected_data


def create_qa_instruction_dataset(tokenizer, gold_entailment_file, retrieval_entailment_file, neutral_file,
                                  output_file, seed=123):
    processed_data = []
    gold_entailment_data = list(jsonlines.open(gold_entailment_file))
    retrieval_entailment_data = list(jsonlines.open(retrieval_entailment_file))
    entailment_data = gold_entailment_data + retrieval_entailment_data
    neutral_data = list(jsonlines.open(neutral_file))
    for element in entailment_data:
        question = element['question']
        answer = element['answer']
        if 'claim' in question:
            continue
        if 'no answer could be found' in answer.lower():
            answer = 'Sorry. I cannot find the answer based on the context.'

        sentence_chunks = element['sentence_chunks'][::-1]  # most relevant context closest to question
        context = '\n\n'.join(sentence_chunks)
        message = [
            {"role": "user", "content": question}
        ]
        prompt = get_formatted_input(message, context)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) < 4096:
            output = f'<|begin_of_text|> {answer} <|end_of_text|>'
            processed_data.append({'prompt': prompt, 'output': output})

    neutral_data = filter_neutral_data(neutral_data,
                                       num_samples=min(int(0.05 * len(processed_data)), len(neutral_data)))
    print(len(processed_data))

    for element in neutral_data:
        question = element['question']
        if 'claim' in question:
            continue
        answer = 'Sorry. I cannot find the answer based on the context.'
        sentence_chunks = element['sentence_chunks'][::-1]  # most relevant context closest to question
        context = '\n\n'.join(sentence_chunks)
        message = [
            {"role": "user", "content": question}
        ]
        prompt = get_formatted_input(message, context)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) < 4096:
            output = f'<|begin_of_text|> {answer} <|end_of_text|>'
            processed_data.append({'prompt': prompt, 'output': output})

    print(len(processed_data))
    random.seed(seed)
    random.shuffle(processed_data)

    with open(output_file, 'w') as file:
        json.dump({'data': processed_data}, file)


if __name__ == '__main__':
    #combine_all_dataset()
    # tokenizer = AutoTokenizer.from_pretrained('/hkfs/work/workspace_haic/scratch/vl8701-llm2/Llama3-ChatQA-1.5-70B')
    # create_qa_instruction_dataset(tokenizer,
    #                              './data/train_goldqa_statement_gold_doc_entailed.jsonl',
    #                              './data/train_gold_qa_top3_retrieval_entailed.jsonl',
    #                              './data/train_gold_qa_top3_retrieval_neutral.jsonl',
    #                              './data/qa_instruction_train_few_nei.json')
    # analysis()
    # include_qa_briefs()
    # create_question_instruction_data('./data/train.json')
    # combine_all_dataset()
    # create_verification_dataset('./data/dev.json', './data/claim_verification_justification_verification_dev.json')
    create_verification_dataset_source('./data/dev.json',
                                       './data/claim_verification_justification_verification_source_test.json')
