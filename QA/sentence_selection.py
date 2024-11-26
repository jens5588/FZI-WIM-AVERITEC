import jsonlines
import spacy
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sentence_splitter_wrapper_corenlp_en import CoreNLPTokenizer, corenlp_ssplitter
from llama_recipes.utils.inference_utils import dump_jsonl

nlp = spacy.load('en_core_web_lg')


def cross_encoder_sentence_selection(input_file, output_file, top_k, process_count):
    # tok = CoreNLPTokenizer()
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device=f'cuda:{process_count}')
    retrieval = list(jsonlines.open(input_file))
    idx = 0
    error_idx = []
    for element in tqdm(retrieval):
        query = element['query']
        sentence_chunks = element['sentence_chunks']
        sentences = []
        for i, chunk in enumerate(sentence_chunks[:20]):
            if 'null null' not in chunk:
                try:
                    # split_sentences = corenlp_ssplitter(tok, chunk)
                    split_sentences = nlp(chunk)
                    split_sentences = [str(sent) for sent in split_sentences.sents]
                    sentences += split_sentences
                except:
                    print('error')
                    error_idx.append(idx)

        if len(sentences) > 0:
            query_sentence_pairs = [(query, sentence) for sentence in sentences]
            scores = model.predict(query_sentence_pairs, batch_size=256, show_progress_bar=False).tolist()
            scores_ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            sorted_sentences = [query_sentence_pairs[i][1] for i in scores_ranking][:top_k]
            element['top_sentences'] = sorted_sentences
        dump_jsonl([element], output_file, append=True)
        idx += 1
    print(error_idx)


if __name__ == '__main__':
    cross_encoder_sentence_selection('./data/sorted_claim_question_separate_ms-marco-l12_dev_top100.jsonl',
                                     output_file='./data/sorted_claim_question_separate_ms-marco-l12_dev_top100_sent_selection.jsonl',
                                     top_k=10, process_count=0)
