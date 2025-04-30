# training/train_llama3_unsloth.py

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import login

# Step 1: Load the model
max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Step 2: Apply LoRA (parameter-efficient fine-tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Step 3: Format prompts
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input_, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Step 4: Authenticate with HuggingFace Hub
login(token="your_token_here")

# Step 5: Load and preprocess dataset
dataset = load_dataset("Biswa46/my-dataset_all_1K")
dataset = concatenate_datasets(list(dataset.values()))
dataset = dataset.map(formatting_prompts_func, batched=True)

# Step 6: Define trainer and training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=150,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# Step 7: Train the model
trainer_stats = trainer.train()

# Step 8: Inference example
FastLanguageModel.for_inference(model)

inputs = tokenizer([
    alpaca_prompt.format(
        "Translate the English input text into Tamil.",
        "Hey Hello, How are you?",
        ""
    )
], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs))

# Step 9: Push to HuggingFace Hub
model.push_to_hub("Biswa46/llama_3_1_eng_tam")
tokenizer.push_to_hub("Biswa46/llama_3_1_eng_tam")
