import json
import jsonlines
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
from llama_recipes.utils.inference_utils import dump_jsonl


def answer_types_analysis(input_file):
    with open(input_file) as file:
        data = json.load(file)
    boolean = []
    abstractive = []
    extractive = []
    unanswerable = []
    for element in data:
        questions = element['questions']
        for question in questions:
            if question['answers'][0]['answer_type'] == 'Boolean':
                boolean.append(question)
            elif question['answers'][0]['answer_type'] == 'Abstractive':
                abstractive.append(question)
            elif question['answers'][0]['answer_type'] == 'Extractive':
                extractive.append(question)
            elif question['answers'][0]['answer_type'] == 'Unanswerable':
                unanswerable.append(question)
    return boolean, abstractive, extractive, unanswerable


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)  # Stop when ALL sequences hit the stopping criteria
        # return True if True in done # Stop when ANY sequence hits the stopping criteria


def create_qa_statements_prompt(with_instruction=True, random_seed=123):
    # prompt from qa type: boolean, abstractive, extractive and unanswerable
    instruction = f'''Your task is to convert question answer pairs into statements. In the following there are some example showing how to convert question answer pairs into statements.\n\n'''
    examples = [
        {
            'question': 'Did Hunter Biden have any experience in the energy sector at the time he joined the board of the  Burisma energy company in 2014',
            'answer': 'No',
            'statement': "Hunter Biden didn't have any experience in the energy sector at the time he joined the board of the  Burisma energy company in 2014"
        },
        {
            'question': 'Did any politicians condemn the rioting after the death of George Floyd in 2020?',
            'answer': 'Yes',
            'statement': 'There are politicians condemn the rioting after the death of George Floyd in 2020.'
        },
        {
            'question': 'Did the 2017 tax bill deliver the largest tax cuts in American history?',
            'answer': 'This tax cut is the 8th largest as a percent of Gross Domestic Product (GDP) since 1918 and the 4th largest in inflation-adjusted dollars.',
            'statement': 'The tax cut in 2017 is the 8th largest as a percent of Gross Domestic Product (GDP) since 1918 and the 4th largest in inflation-adjusted dollars.'
        },
        {
            'question': 'How much of their national budget did the Kenyan judiciary receive in 2021?',
            'answer': 'Budget speeches for 2020/21 show the judiciary received 0.6% of the national budget.',
            'statement': 'Budget speeches for 2020/21 show the Kenyan judiciary received 0.6% of the national budget.'
        },
        {
            'question': 'Was Sanger a racist?',
            'answer': "As her biographer Ellen Chesler told me, she was a progressive who believed in racial integration. She voted for Norman Thomas. She worked with progressive Black people—W.E.B. Du Bois, for example, who along with Mary McCleod Bethune and Adam Clayton Powell Sr. served on the board of the Negro Project, a network of birth control and maternal health clinics Sanger established in Harlem and the South. In 1966, Martin Luther King accepted Planned Parenthood’s first Margaret Sanger Award, and in his statement offered a vigorous endorsement of voluntary birth control.",
            'statement': "As Sanger's biographer Ellen Chesler told me, Sanger was a progressive who believed in racial integration. She voted for Norman Thomas. She worked with progressive Black people—W.E.B. Du Bois, for example, who along with Mary McCleod Bethune and Adam Clayton Powell Sr. served on the board of the Negro Project, a network of birth control and maternal health clinics Sanger established in Harlem and the South. In 1966, Martin Luther King accepted Planned Parenthood’s first Margaret Sanger Award, and in his statement offered a vigorous endorsement of voluntary birth control."
        },
        {
            'question': 'What resolutions did Biden support in favor of US intervention in Iraq',
            'answer': "He supported the H.J.Res.114 - Authorization for Use of Military Force Against Iraq Resolution of 2002 107th Congress (2001-2002)",
            'statement': "Joe Biden supported the H.J.Res.114 - Authorization for Use of Military Force Against Iraq Resolution of 2002 107th Congress (2001-2002)"
        },
        {
            'question': 'Is there a global average for the number of judges compared to population?',
            'answer': 'No answer could be found.',
            'statement': 'No global average for the number of judges compared to population could be found.'
        },
        {
            'question': 'Should counties be chasing the 10% spending target or should it be done at a national level?',
            'answer': 'No answer could be found.',
            'statement': 'No answer could be found regarding whether counties should be chasing the 10% spending target or if it should be done at a national level.'
        }
    ]
    random.seed(random_seed)
    random.shuffle(examples)
    prompt = ''
    if with_instruction:
        prompt = instruction
    for element in examples:
        prompt += f'Question: {element["question"]}\nAnswer: {element["answer"]}\nStatement: {element["statement"]}\n\n'

    return prompt


def generate_statement(example_prompt, question_response_pairs):
    statements = []
    for pair in question_response_pairs:
        # print(f'pair: {pair}')
        question, response = pair
        prompt = [f'{example_prompt}Question: {question}\nAnswer:{response}\nStatement: ']
        inputs = tokenizer(prompt, return_tensors='pt')
        input_length = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(start_length=input_length, eof_strings='\n\n', tokenizer=tokenizer)])
        outputs = model.generate(inputs.input_ids.to('cuda'), attention_mask=inputs.attention_mask.to('cuda'),
                                 max_new_tokens=200, do_sample=False, stopping_criteria=stopping_criteria,
                                 return_dict_in_generate=True, output_scores=True)
        generation = tokenizer.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        statements.append(generation.replace(prompt[0], '').replace('\n\n', ''))
    return statements


def run(input_file, output_file, with_instruction=True):
    if 'jsonl' in input_file:
        data = list(jsonlines.open(input_file))
    else:
        with open(input_file) as file:
            data = json.load(file)
    example_prompt = create_qa_statements_prompt(with_instruction)
    # print(example_prompt)
    num_error = 0
    for element in tqdm(data):
        """if 'greedy_responses' in element.keys():
            generated_questions = element['generated_questions']
            greedy_responses = []
            for item in element['greedy_responses']:
                greedy_responses.append(item[0][0])
            # print(f'greedy responses: {greedy_responses}')
            question_response_pairs = list(zip(generated_questions, greedy_responses))
            greedy_statements = generate_statement(example_prompt, question_response_pairs)
            element['greedy_statements'] = greedy_statements

        if 'sampling_responses' in element.keys():
            generated_questions = element['generated_questions']
            sampling_responses = []
            for item in element['sampling_responses']:
                sampling_responses.append([part[0] for part in item])
            sampling_statements = []
            for i in range(len(generated_questions)):
                question = generated_questions[i]
                responses = sampling_responses[i]
                question_response_pairs = [(question, response) for response in responses]
                statements = generate_statement(example_prompt, question_response_pairs)
                sampling_statements.append(statements)

            element['sampling_statements'] = sampling_statements"""
        if 'selected_responses' in element.keys():
            generated_questions = element['generated_questions']
            selected_responses = element['selected_responses']
            question_response_pairs = list(zip(generated_questions, selected_responses))
            total_statements = []
            for pair in question_response_pairs:
                question, responses = pair
                statements = generate_statement(example_prompt, [(question, response) for response in responses])
                total_statements.append(statements)
            element['selected_statements'] = total_statements

        if "questions" in element.keys():
            questions = element['questions']
            for i, part in enumerate(questions):
                question = part['question']
                answers = part['answers']
                for j, item in enumerate(answers):
                    answer = item['answer']
                    answer_type = item['answer_type']
                    if answer_type == 'Boolean':
                        boolean_explanation = item['boolean_explanation']
                        answer += f', {boolean_explanation}'
                    statement = generate_statement(example_prompt, [(question, answer)])[0]
                    print(f'{i}-{j}-{question}-{answer}')
                    print(f'statement: {statement}')
                    element['questions'][i]['answers'][j]['statement'] = statement

            num_error += 1

        dump_jsonl([element], output_file, append=True)
    print(f'total errors: {num_error}')


if __name__ == '__main__':
    model_id = '/hkfs/work/workspace_haic/scratch/vl8701-llm/llama3/Meta-Llama-3-70B-Instruct'
    model_size = '70B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 128001
    tokenizer.eos_token_id = 128001
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    # model = None

    run('./data/dev/qa_generated_questions_combined_dev_top3_answer_selection_all_rounds.jsonl',
        output_file='./data/dev/qa_generated_questions_combined_dev_top3_statements_all_rounds.jsonl')
