from datasets import load_dataset
from transformers import AutoTokenizer


def get_preprocessed_averitec_qa(dataset_config, tokenizer, split):
    dataset = load_dataset("json",
                           data_files={
                               'train': './data/qa_instruction_train_few_nei.json',
                               'test': './data/qa_instruction_test_few_nei.json'},
                           field='data', split=split, cache_dir='./data/dataset/')

    def apply_prompt_template(sample):
        return {
            "prompt": sample['prompt'],
            "output": sample["output"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        output = tokenizer.encode(sample["output"], add_special_tokens=False)

        sample = {
            "input_ids": prompt + output,
            "attention_mask": [1] * (len(prompt) + len(output)),
            "labels": [-100] * len(prompt) + output,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


if __name__ == '__main__':
    # Llama3-ChatQA-1.5-70B
    tokenizer = AutoTokenizer.from_pretrained(
        "/hkfs/work/workspace/scratch/vl8701-llm/Llama3-ChatQA-1.5-70B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    get_preprocessed_averitec_qa(None, tokenizer, 'train')
    get_preprocessed_averitec_qa(None, tokenizer, 'test')
