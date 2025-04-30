from huggingface_hub import login
import torch
import pandas as pd
from unsloth import FastLanguageModel
from datasets import load_dataset
import sacrebleu
import os

# Login to HuggingFace
login(token="your_token_here")

# Settings
num_data = 50  # Number of samples per pair
model_name = "Biswa46/llama_3_1_eng_tam"
model_short_name = model_name.split("/")[-1]

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Prompt Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Translate the {} input text into {}.
### Input:
{}
### Response:
{}"""

# Language mapping
lang_map = {
    "eng_Latn": "English",
    "asm_Beng": "Assamese",
    "hin_Deva": "Hindi",
    "mal_Mlym": "Malayalam",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "san_Deva": "Sanskrit",
    "kan_Knda": "Kannada",
    "tel_Telu": "Telugu",
    "mar_Deva": "Marathi",
    "tam_Taml": "Tamil",
    "ory_Orya": "Odia",
    "npi_Deva": "Nepali",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu"
}

# Codes
eng_code = [k for k, v in lang_map.items() if v == "English"][0]
hin_code = [k for k, v in lang_map.items() if v == "Hindi"][0]
ben_code = [k for k, v in lang_map.items() if v == "Bengali"][0]
tel_code = [k for k, v in lang_map.items() if v == "Telugu"][0]

# Dataset
dataset = load_dataset("ai4bharat/IN22-Gen", split="test")

# Evaluation function
def evaluate_translation_pairs(source_codes, target_codes, save_prefix):
    results = []
    sample_translations = []

    for src_code in source_codes:
        src_name = lang_map[src_code]

        for tgt_code in target_codes:
            if src_code == tgt_code:
                continue

            tgt_name = lang_map[tgt_code]

            data_pairs = [
                {"source": item[src_code], "target": item[tgt_code]}
                for item in dataset if item[src_code] and item[tgt_code]
            ][:num_data]  # Limit

            if not data_pairs:
                print(f"No data for {src_name} → {tgt_name}, skipping...")
                continue

            print(f"🔍 Evaluating {src_name} → {tgt_name} ({len(data_pairs)} samples)")

            source_sentences = [x["source"] for x in data_pairs]
            reference_translations = [x["target"] for x in data_pairs]
            generated_translations = []

            for src_text in source_sentences:
                prompt = alpaca_prompt.format(src_name, tgt_name, src_text, "")
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=128)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                response_text = decoded.split("### Response:")[-1].strip()
                generated_translations.append(response_text)

            bleu = sacrebleu.corpus_bleu(generated_translations, [reference_translations])
            chrf = sacrebleu.corpus_chrf(generated_translations, [reference_translations])

            # Save all samples
            for src, ref, gen in zip(source_sentences, reference_translations, generated_translations):
                sample_translations.append({
                    "Source Language": src_name,
                    "Target Language": tgt_name,
                    "Input Text": src,
                    "Reference": ref,
                    "Generated": gen
                })

            results.append({
                "Source": src_name,
                "Target": tgt_name,
                "BLEU": bleu.score,
                "chrF": chrf.score,
                "Num_Samples": len(data_pairs)
            })

    # Save sample translations
    sample_df = pd.DataFrame(sample_translations)
    sample_file = f"{model_short_name}_{save_prefix}_sample_translations.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"All sample translations saved to {sample_file}")

    return pd.DataFrame(results)

# Non-English codes
non_eng_codes = [k for k in lang_map.keys() if k != eng_code]
non_hin_codes = [k for k in lang_map.keys() if k != hin_code]
non_ben_codes = [k for k in lang_map.keys() if k != ben_code]
non_tel_codes = [k for k in lang_map.keys() if k != tel_code]

# Source languages to evaluate
source_languages = {
    "english": (eng_code, non_eng_codes),
    "hindi": (hin_code, non_hin_codes),
    "bengali": (ben_code, non_ben_codes),
    "telugu": (tel_code, non_tel_codes)
}

# Evaluate each
for src_name, (src_code, tgt_codes) in source_languages.items():
    print(f"\n===== Evaluating {lang_map[src_code]} → Other Languages =====")
    src_to_other_results = evaluate_translation_pairs([src_code], tgt_codes, f"{src_name}_to_other")
    src_to_other_results = src_to_other_results.sort_values(by="BLEU", ascending=False)
    src_to_other_file = f"{model_short_name}_{src_name}_to_other_translations.csv"
    src_to_other_results.to_csv(src_to_other_file, index=False)
    print(f"Results saved to {src_to_other_file}")

    print(f"\n===== Evaluating Other Languages → {lang_map[src_code]} =====")
    other_to_src_results = evaluate_translation_pairs(tgt_codes, [src_code], f"other_to_{src_name}")
    other_to_src_results = other_to_src_results.sort_values(by="BLEU", ascending=False)
    other_to_src_file = f"{model_short_name}_other_to_{src_name}_translations.csv"
    other_to_src_results.to_csv(other_to_src_file, index=False)
    print(f"Results saved to {other_to_src_file}")

    # Print Summary
    print(f"\n===== SUMMARY for {lang_map[src_code]} =====")
    if not src_to_other_results.empty:
        print(f"Best {lang_map[src_code]} → Other: {src_to_other_results.iloc[0]['Source']} → {src_to_other_results.iloc[0]['Target']} (BLEU: {src_to_other_results.iloc[0]['BLEU']:.2f})")
        print(f"Average BLEU {lang_map[src_code]} → Other: {src_to_other_results['BLEU'].mean():.2f}")
    if not other_to_src_results.empty:
        print(f"Best Other → {lang_map[src_code]}: {other_to_src_results.iloc[0]['Source']} → {other_to_src_results.iloc[0]['Target']} (BLEU: {other_to_src_results.iloc[0]['BLEU']:.2f})")
        print(f"Average BLEU Other → {lang_map[src_code]}: {other_to_src_results['BLEU'].mean():.2f}")

print("\n🎉 Evaluation complete for all source languages!")