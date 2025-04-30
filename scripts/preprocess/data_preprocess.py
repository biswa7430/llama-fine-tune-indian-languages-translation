import sys
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login


def load_dataset_fun(dataset_name, sample_size=1000):
    """Load a small portion of the dataset in streaming mode."""
    streamed_dataset = load_dataset(dataset_name, split="train", streaming=True)
    subset_list = list(streamed_dataset.take(sample_size))
    dataset = Dataset.from_list(subset_list)
    return dataset


def format_translation_prompts(examples, src_lng, tgt_lng, eos_token, prompt_template):
    instruction = f"Translate the {src_lng} input text into {tgt_lng}."
    texts = [prompt_template.format(instruction, inp, out) + eos_token
             for inp, out in zip(examples[src_lng], examples[tgt_lng])]
    key = f'{src_lng}_to_{tgt_lng}'
    return {key: texts}


if __name__ == "__main__":
    dataset_name = "ai4bharat/wiki-translate"
    model_name = "unsloth/Meta-Llama-3.1-8B"

    # Load and preprocess the dataset
    dataset = load_dataset_fun(dataset_name)
    dataset = dataset.remove_columns(['doc_id', 'url', 'title'])

    # Column renaming
    rename_mapping = {
        "eng_Latn": "English", "asm_Beng": "Assamese", "hin_Deva": "Hindi",
        "mal_Mlym": "Malayalam", "ben_Beng": "Bengali", "guj_Gujr": "Gujarati",
        "san_Deva": "Sanskrit", "kan_Knda": "Kannada", "tel_Telu": "Telugu",
        "mar_Deva": "Marathi", "tam_Taml": "Tamil", "ory_Orya": "Odia",
        "npi_Deva": "Nepali", "pan_Guru": "Punjabi", "urd_Arab": "Urdu"
    }
    dataset = dataset.rename_columns(rename_mapping)


    # Load tokenizer and set up EOS token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    EOS_TOKEN = tokenizer.eos_token

    # Define the prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    # Create translation pairs
    avl_languages = list(rename_mapping.values())
    for source in avl_languages:
        for target in avl_languages:
            if source != target:
                dataset = dataset.map(
                    lambda example: format_translation_prompts(example, src_lng=source, tgt_lng=target, eos_token=EOS_TOKEN, prompt_template=alpaca_prompt),                    batched=True
                )
    dataset = dataset.remove_columns(avl_languages)

    # Create DatasetDict
    dataset_dict = DatasetDict({feature: Dataset.from_dict({'text': dataset[feature]}) for feature in dataset.column_names})

    # Calculate the total size of the dataset
    total_size_bytes = sum(sum(sys.getsizeof(value[column]) for column in value.column_names) for value in dataset_dict.values())
    total_size_mb = total_size_bytes / (1024**2)
    print(f"Total size of DatasetDict: {total_size_mb:.2f} MB")

    # Push to Hugging Face Hub
    login(token="your_token_here")
    dataset_dict.push_to_hub("Biswa46/my-dataset_all_1K")

    # Load back the pushed dataset
    loaded_dataset = load_dataset("Biswa46/my-dataset_all_1K")
    print(loaded_dataset)

    print("Dataset processing and upload complete! 🚀")