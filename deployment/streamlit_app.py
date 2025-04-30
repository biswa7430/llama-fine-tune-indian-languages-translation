import streamlit as st
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
from huggingface_hub import login
from transformers import logging

# Suppress warnings
logging.set_verbosity_error()

# Login to Hugging Face
login(token="your_token_here")

# Load model
@st.cache_resource
def load_model():
    model_name = "Biswa46/llama_3_1_eng_tam"
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# Prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")
    instruction_type = st.selectbox("Instruction Type", ["Translate","Chat", ])
    src_lang = st.selectbox("Source Language", ["English","Assamese","Hindi","Malayalam","Bengali","Gujarati","Sanskrit","Kannada","Telugu","Marathi","Tamil","Odia","Nepali","Punjabi","Urdu"])
    tgt_lang = st.selectbox("Target Language", ["Hindi","English","Assamese","Malayalam","Bengali","Gujarati","Sanskrit","Kannada","Telugu","Marathi","Tamil","Odia","Nepali","Punjabi","Urdu"])

# --- MAIN CONTENT ---
st.title("🦙 LLaMA Instruction-Aware Chatbot")

# Form to submit on Enter
with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_area("💬 Enter your message below", "", height=120, placeholder="Type your message and press Enter...")
    submitted = st.form_submit_button("Send")

# When user submits
if submitted:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating response..."):

            # Dynamic instruction
            if instruction_type.lower() == "chat":
                instruction = "Chat with me in English."
            elif instruction_type.lower() == "translate":
                instruction = f"Translate the {src_lang} text into {tgt_lang}."

            prompt = alpaca_prompt.format(instruction, user_input, "")
            #print(prompt)
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache = True
            )

            output_text = tokenizer.batch_decode(generated_ids)
            print(output_text[0])
            response = output_text[0].split("### Response:")[1].strip()

            st.success("🧠 Response:")
            st.write(response)
