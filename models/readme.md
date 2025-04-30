# Fine-Tuned Language Models

This directory tracks the development and usage of fine-tuned models based on [Unsloth's Meta-LLaMA 3.1 8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B). The models have been incrementally fine-tuned on an English–Tamil instruction-following dataset from AI4Bharat, as part of this research thesis.

---

## 🧠 Base Model

- **Model Name:** `unsloth/Meta-Llama-3.1-8B`
- **Size:** 8 Billion parameters
- **Provider:** Unsloth (optimized LLaMA v3.1 for faster fine-tuning and inference)
- **Features:**
  - RoPE scaling support for longer sequences
  - 4-bit and 8-bit quantization compatibility
  - Native support for LoRA and SFTTrainer

---

## 📈 Fine-Tuning Phases

### 🔹 First Checkpoint: 50 Steps

- **Model Name:** [`Biswa46/llama_3_1_eng_tam`](https://huggingface.co/Biswa46/llama_3_1_eng_tam)
- **Purpose:** Initial warm-up stage to adapt the model to instruction tuning with English–Tamil translation tasks.
- **Training Configuration:**
  - Steps: `50`
  - LoRA: Applied on transformer projections
  - Optimizer: `adamw_8bit`
  - Scheduler: Linear learning rate
  - Dataset: AI4Bharat (English-Tamil)

### 🔹 Final Checkpoint: 150 Steps

- **Model Name:** [`Biswa46/llama_3_1_FT_10_04_25`](https://huggingface.co/Biswa46/llama_3_1_FT_10_04_25)
- **Purpose:** Final fine-tuned model used for evaluation and deployment.
- **Training Configuration:**
  - Steps: `150` (including the 50-step warm-up)
  - Mixed-precision: `fp16` or `bf16` (based on hardware support)
  - Batch size: 2 (with gradient accumulation)
  - Context length: 4096 tokens
  - Tokenizer: Inherited from base model

---

## 💾 Accessing Models

You can download or load the models using `transformers` or `unsloth`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Biswa46/llama_3_1_FT_10_04_25")
tokenizer = AutoTokenizer.from_pretrained("Biswa46/llama_3_1_FT_10_04_25")
