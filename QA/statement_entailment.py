import json
import jsonlines
import random
import torch
import numpy as np
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


def create_entailment_prompt(with_instruction=True, random_seed=123):
    # prompt from qa type: boolean, abstractive, extractive and unanswerable
    instruction = f'''Your task is to check whether the statement is entailed in the corresponding context. There are three labels you can use, Refutes, Supports and Not Enough Information. In the following there are some example showing how to predict the entailment label given context and statement pairs.\n\n'''
    examples = [
        {
            'context': """This is a letter Sean Connery wrote didn't write in response to Steve Jobs after being asked to appear in an Apple ad. Honestly, we're not sure it's legit. The "007" at the bottom-right corner and the "I am fucking JAMES BOND" seem a bit rich. But it's still hilarious. And it's refreshing to see someone respond to Steve Jobs with something other than fawning, drooling admiration. Update: Yep, it's fake. It comes from Scoopertino, The-Onion-for-Apple-news. Thanks Brian Manley! Don't Miss: 11 Tools To Get Your Startup Off The Ground In No Time →\nI am f****** James Bond': Sean Connery letter to Steve Jobs rejecting offer to appear in Apple ad revealed to be fake Thousands of James Bond fans were today taken in by a spoof letter from Sean Conney to Apple boss Steve Jobs in which the film star launches a rant at the computer chief. The faked A4 letter on 007 paper was published online claiming to have been a response from Connery to a request to feature in an advert for Apple. It purportedly shows Connery aggressively declining the offer and telling Jobs 'You are a computer salesman - I am f****** James Bond.' Do you know who I am? A faked letter from James Bond star Sir Sean Connery firmly rejected an apparent advertising role from Apple chief Steve Jobs The letter caused a sensation when it was published on Twitter today with thousands of users believing it to be real. Dated December 1998, it is addressed to Mr Stephen P. Jobs at '1 Infinite Loop, Cupertino, California'. The prankster writes: 'I will say this one more time. Yo do understand English, don't you? I do not sell my soul for Apple or any other company. 'I have no interest in ''changing the world'' as you suggest. You have nothing that I need or want. You are a computer salesman - I am f****** James Bond. 'I can think of no quicker way to destroy my career than to appear in one of your crass adverts. Please do not contact me again.' The person claiming to be the 007 star then signs off with a signature on paper that is headed 'The desk of Sean Connery' and features a 007 gun symbol in the bottom corner. Spoof: The faked letter claims to be from the 'desk of Sean Connery' and is typed on paper with the 007 symbol in the corner The Apple satirical site Scoopertino initially published the fake letter on June 19. They claim that Jobs has been a life-long Bond fan who wanted to call the first computer the Double-O-Mac. Details of how Jobs tried to recruit Connery to flag up struggling sales in 1998 were said to have been revealed in a book on the history of the firm.\nFaux 007 launching expletives at Apple icon fools British social-media guru and becomes meme material. First, the bad news. Sean Connery never actually sent a typewritten letter to Steve Jobs in 1998 refusing to be in an Apple ad. But the awesome news is that quite a few people believe Connery owns personalized stationery with a "007" vanity stamp in the corner and that he would have no qualms using it to dash off a letter dressing down Jobs by declaring "...you are a computer salesman. The letter was actually part of a satirical article on the previously little known (and very specific) humor site, Scoopertino, which peddles Onion-style and tongue-in-cheek "Unreal Apple News." But when British marketing exec John Willshire took the letter seriously and posted it on Twitter and his blog, it started rocketing around Twitter and beyond. At one point early today, Willshire was among the top trending topics on Twitter, beating out even Wimbledon in the U.K. Willshire has since posted an update clarifying that the Connery-to-Jobs letter was in fact a fake and explaining that he had been duped. Ironically, Willshire sells himself and his firm as specializing in social media, which means one of two things--either he's a bit overconfident, or he's some kind of marketing evil genius who had this entire viral strategy planned from the very start as a publicity stunt. If the reality is the latter, there's only one superspy agent I know of who could challenge such an evil mastermind. I've already begun drafting a letter to Mr. Connery...""",
            'statement': """Sean Connery refused to appear in an Apple commercial, as evidenced by a published letter to Apple CEO Steve Jobs.""",
            'entailment': "Refutes"
        },
        {
            'context': """Islamabad, October 25: Pakistan Prime Minister Imran Khan on Sunday hit out at French President Emmanuel Macron "for attacking Islam and hurting sentiments of Muslims". Imran Khan's remarks were in the context of Emmanuel Macron's recent statements after a French teacher was beheaded near Paris after he had shown cartoons of the Prophet Mohammed to his students in a class on freedom of speech. Macron had said that the teacher "was killed because Islamists want our future." "Hallmark of a leader is he unites human beings, as Mandela did, rather than dividing them. This is a time when Pres Macron could have put healing touch & denied space to extremists rather than creating further polarisation & marginalisation that inevitably leads to radicalisation (sic)," Khan said in a series of tweet. Charlie Hebdo's Republication of Cartoons of Prophet Muhammad A 'Provocation', is 'Absolutely Unacceptable', Says Iran. "It is unfortunate that he has chosen to encourage Islamophobia by attacking Islam rather than the terrorists who carry out violence, be it Muslims, White Supremacists or Nazi ideologists. Sadly, President Macron has chosen to deliberately provoke Muslims, incl his own citizens, through encouraging the display of blasphemous cartoons targeting Islam & our Prophet PBUH," the Pakistani Prime Minister opined. He went on to add: "By attacking Islam, clearly without having any understanding of it, President Macron has attacked & hurt the sentiments of millions of Muslims in Europe & across the world. The last thing the world wants or needs is further polarisation. Public statements based on ignorance will create more hate, Islamophobia & space for extremists." It is unfortunate that he has chosen to encourage Islamophobia by attacking Islam rather than the terrorists who carry out violence, be it Muslims, White Supremacists or Nazi ideologists. Sadly, President Macron has chosen to deliberately provoke Muslims, incl his own citizens, The last thing the world wants or needs is further polarisation. Public statements based on ignorance will create more hate, Islamophobia & space for extremists.\nPakistani PM Imran Khan and Turkish President Erdogan slams French President Emmanuel Macron on Sunday for disrespecting Islam. He criticized Macron over his recent anti-Muslim attitude and for hurting the sentiments of millions of Muslims around the world. In a series of tweets, PM Imran Khan gave examples of the iconic leader Nelson Mandela. He said this was a time when President Macron could have put the healing touch. Instead, Macron has created further polarisation and marginalization that inevitably leads to radicalization. Khan added by attacking Islam, Macron has hurt the sentiments of Muslims around the world. He said the last thing the world wants or needs was further polarization. The French president is being criticized by Muslims with protests breaking out in several cities across the world. Macron also accused Muslims of separatism. He vowed not to give up cartoons depicting the Holy Prophet Muhammad (PBUH). His comments came in response to the beheading of a 47-year-old teacher, who was attacked on his way home from the junior high school where he taught in Conflans-Sainte-Honorine, 40 kilometers northwest of Paris. Turkish President Recep Tayyip Erdoğan said on Sunday that his French counterpart Emmanuel Macron needs “mental treatment” over his hostile attitude towards Muslims and Islam. Following Erdogan’s remarks, France recalled its ambassador. “Outrage and insult are not a method,” Macron’s office said. “What is the problem of this person called Macron with Muslims and Islam? Macron needs treatment on a mental level,” Erdogan said in a speech in the central Turkish city of Kayseri. Following President Macron’s comments on Islam, several Muslim countries and trade associations across the world have announced the boycott of French products. They also protested the recent comments made by Macron on Islam. Hashtags such as the #BoycottFrenchProducts and #ShameonYouMacro are trending across social media. In Kuwait, the chairman and members of the board of directors of the Al-Naeem Cooperative Society decided to boycott all French products. They removed the products from supermarket shelves.\nPakistan's Khan slams Macron's views on IslamOctober 26, 2020 Pakistan's Prime Minister Imran Khan joined Turkish President Recep Tayyip Erdogan on Sunday in criticizing French President Emmanuel Macron for his recent comments on Islam. "This is a time when President Macron could have put a healing touch and denied space to extremists rather than creating further polarization and marginalization that inevitably leads to radicalization," Khan wrote on Twitter. "Sadly, President Macron has chosen to deliberately provoke Muslims, including his own citizens, and encouraged the display of blasphemous cartoons targeting Islam and the Holy Prophet [Muhammad]," he added. Macron didn't directly respond to Khan, but issued a tweet later in the day saying that the French government respects all differences in "a spirit of peace." "We do not accept hate speech and defend reasonable debate. We will always be on the side of human dignity and universal values," said Macron. Khan's scathing criticism of Macron follows days after the French president dedicated a high-level ceremony to Samuel Paty, a teacher who was beheaded for showing students caricatures of the Prophet Muhammad. Macron said the teacher "was killed as Islamists want our future," as he pledged to fight "Islamist separatism" that threatened to take hold of some Muslim communities around France. Read more: The French battle for freedom of speech: 'It's all about the principle' Turkish President Erdogan blasted France and Europe on Saturday over what he saw as "rising Islamophobia." "What problem does this person called Macron have with Muslims and Islam? Macron needs treatment on a mental level," Erdogan said in a speech at a provincial congress of his Justice and Development (AK) Party in the central Turkish city of Kayseri. Paris condemned Erdogan's remarks as "unacceptable," adding that it was recalling its envoy to Ankara to discuss the matter. On Sunday, Khan also sought a ban on Islamophobic content on Facebook, similar to the ban Facebook has for content on the Holocaust.""",
            'statement': """No information could be found regarding French authorities deporting 118 Pakistani citizens.""",
            'entailment': 'Supports'
        },
        {
            'context': """- Nadar (also referred to as Nadan, Shanar and Shanan) is a Tamil caste of India. Nadars are predominant in the districts of Kanyakumari, Thoothukudi, Tirunelveli and Virudhunagar. The Nadar community was not a single caste, but developed from an assortment of related subcastes, which in course of time came under the single banner Nadar. Nadar climbers were the largest subsect of today's Nadar community. A few subsects of the Nadar community, such as the Nelamaikkarars, were traditionally wealthy landlords and money lenders. Historically, most Nadars were cultivators of palmyra trees and jaggery and a few were also involved in the toddy trade. Nadar climbers had faced discrimination from major upper castes in some regions. The martial art of Varma Kalai was historically practiced by the Nadars. The socio-economic development achieved by the Nadars in southern India has elicited academic interest. Nadars are classified and listed as an Other Backward Class by the governments of both Tamil Nadu and India.\n, the Nadars are well known for their bravery throughout the southern part of Tamilnadu. The ancient capital city of Pandiya Nadu, Korkai, is predominantly occupied by the Nadars. After successive invasions from the north by the Kalabhras and other Vadugas on the Pandiyan kingdom, the Nadars were forced out of power and almost became extinct in the 18th century Pandyas. Unlike many other ancient communities who were considered as low castes by the brahmanical classes, they fought back and regained their original status. The lords of Pandia dynasty were called Maran, Maranadar/Mara Nadalwar. Nattar, Rayan, Nadar, Kshatriya Nadar, Nadan, Nadavaru Nadava, Alwar, etc Karukkupattayathar( Elite guards ),Kodi Marathar(who defend the flag),Sivanthi( Elite Suicidal army ) Panicker( one who trained in Martial arts ) are some other titles. The soldiers were called Chanar (chan=Iron, those who carried iron implements when iron was rare)( Sanar, Shanar) Pullu Kai Chanar are hired soldiers who threw spears. (Pullu = spear or stake). Enadhi were archers etc. Modern Nadar(caste) community descend from all the elements of Pandiyan kingdom from kings, soldiers and slaves. In recent past, Shanars were also treated as untouchbles, may be due to their slavery jobs. The Ezhavas of Alappey district are called Ezhava Shanars may be because of the intermingling with the Nadars of the earlier Pandiyan era. Later, many of these untouchble shanars converted to Christianity. The community which was known as 'Shanans' till the 19th century came to be known as Nadars. The title Nadar is believed to be derived from the Nadans, the aristocrats and the highest of the old Shanan community. The aristocrats among the Nadars in those days were known as Nadans and the poor among the caste, who did toddy tapping for a living, were known as Shanans. The poor among the Nadars(Shanans) during early times possessed no agricultural lands due to the Nayak invasion.\nThe ancient historian Herodotus tells us that in 400 BC, the Palmyra-Tappers (the Nadars as we call them now) were Valiant Fighters and good Tradesmen, dealing with inter-Continental Trade. By the 15th century AD, Nadars were weakened by the Nayaks because of the in-fights and dis-unity among the Nadar brothers. Following World War 1, the Brahmins began to dominate the Independence movement of the Madras Presidency. The British then tried to secure the non- Brahmin support. Hence they conceded their request of designating ” Nadars” instead of ” Shanars” in the 1921 Census. With the coming of the British rule to the southern districts, roads were improved and better security emerged. The Shanar populace then began to utilize this opportunity and began to move northwards to sell their palmyra- products as well as dried fish and salt. Along their trade routes they established “pettais” or fortified compounds, to protect themselves against thieves and other caste men and also for them to take some rest. From the 16th to 19th century AD, the Nadars had to struggle under the new Caste system imposed and had to fight hard to come up socially, economically and politically. It was during these periods that the great Fights like “The Temple Entry Movement, The Upper Cloth Revolution, The Human Rights Movements” by many leaders like Vellayyan Nadar, Mooka Nadan, WPA.Soundara Pandyan Nadar, Ayya Mudisoodum Perumal, Marshal A.Nesamony, etc., were fought and won over. The 19th century saw the Nadars embracing Christianity in large numbers some out of will and some due to the easier acceptance into Christian schools. Today the distribution of Nadars between the two major religions, Hindu and Christian, are 60% and 40% respectively. Sanskrit dictionary tells Nadar as a ‘Royal Race’. Although the Christian Missionaries have economically helped the Nadars to survive, they have not mentioned the high qualities and the valuable ‘Sastras’ especially the Varma sastra found among Kanniyakumari Nadars. The premium placed on education by the Nadar community resulted in a drastic improvement in the socio-economic landscape of the community, a distinction reserved to upper classes of India.""",
            'statement': """UNESCO is an organization that works to improve education, science, and culture around the world.""",
            'entailment': """Not Enough Information"""
        }
    ]
    random.seed(random_seed)
    random.shuffle(examples)
    prompt = ''
    if with_instruction:
        prompt = instruction
    for element in examples:
        prompt += f'Context:\n{element["context"]}\n\nStatement:\n{element["statement"]}\n\nEntailment:\n{element["entailment"]} ###\n\n'

    return prompt


def generate_entailment_classification(example_prompt, context_statement_pairs):
    entailment_classification = []
    for pair in context_statement_pairs:
        context, statement = pair
        context = context[::-1]
        context = '\n'.join(context)
        prompt = [f'{example_prompt}Context:\n{context}\n\nStatement:\n{statement}\n\nEntailment:\n']
        print(prompt[0])
        inputs = tokenizer(prompt, return_tensors='pt')
        input_length = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList(
            [EndOfFunctionCriteria(start_length=input_length, eof_strings='###', tokenizer=tokenizer)])
        outputs = model.generate(inputs.input_ids.to('cuda'), attention_mask=inputs.attention_mask.to('cuda'),
                                 max_new_tokens=10, do_sample=False, stopping_criteria=stopping_criteria,
                                 return_dict_in_generate=True, output_scores=True)
        generation = tokenizer.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        token_probabilities = []
        generated_tokens = outputs.sequences[:, input_length:]
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            token_probabilities.append((tokenizer.decode(tok), float(round(np.exp(score.cpu().numpy()), 2))))
        entailment_classification.append(
            (generation.replace(prompt[0], '').replace('\n', '').replace('###', '').strip(), token_probabilities))

    return entailment_classification


def run(input_file, output_file, with_instruction=True):
    if 'jsonl' in input_file:
        data = list(jsonlines.open(input_file))
    else:
        with open(input_file) as file:
            data = json.load(file)
    example_prompt = create_entailment_prompt(with_instruction)

    num_error = 0
    for element in tqdm(data[1100:]):
        if 'selected_statements' in element.keys():
            sentence_chunks = element['sentence_chunks']
            selected_statements = element['selected_statements']
            statement_context_pairs = list(zip(selected_statements, sentence_chunks))
            statement_entailment_labels = []
            for pair in statement_context_pairs:
                statements, context = pair
                selected_statement_entailment = generate_entailment_classification(example_prompt,
                                                                                   [(context, statement.strip()) for
                                                                                    statement in
                                                                                    statements])
                statement_entailment_labels.append(selected_statement_entailment)
            element['selected_statement_entailment'] = statement_entailment_labels

            dump_jsonl([element], output_file, append=True)
            print('####################')
        else:
            num_error += 1
    print(f'total errors: {num_error}')


if __name__ == '__main__':
    model_id = '/hkfs/work/workspace_haic/scratch/vl8701-llm2/mixtral/Mixtral-8x7B-Instruct-v0.1'
    # model_id = '/hkfs/work/workspace_haic/scratch/vl8701-llm2/llama3/Meta-Llama-3-70B-Instruct'
    # tokenizer.pad_token_id = 128001
    # tokenizer.eos_token_id = 128001

    model_size = '70B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 2
    tokenizer.eos_token_id = 2
    # model = None
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    # run('./data/qa_generated_questions_combined_dev_top3_greedy_sampling_250.jsonl',
    #    output_file='./data/qa_generated_questions_combined_dev_top3_greedy_sampling_statements_250.jsonl',
    #    with_instruction=True)
    run('./data/test/qa_generated_questions_combined_test_top3_statements.jsonl',
        output_file='./data/test/qa_generated_questions_combined_test_top3_statements_entailment_mixtral_2215.jsonl')
