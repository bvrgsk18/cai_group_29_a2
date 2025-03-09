import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

class LlmHandler:
    def __init__(self, HUGGINGFACE_TOKEN, MAX_NEW_TOKENS, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        login(token=HUGGINGFACE_TOKEN)
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # Initialize model as None

        # Load model synchronously, blocking execution until it's ready
        self.load_model()
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

class LlmHandler:
    def __init__(self, MAX_NEW_TOKENS, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        huggingface_token = st.secrets["HUGGINGFACE_TOKEN"] 
        login(token=huggingface_token)
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if GPU is available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use FP16 for GPU, FP32 for CPU
            device_map="auto"# if device == "cuda" else None  # Auto device if GPU, otherwise default to CPU
            load_in_4bit=True #if device == "cuda" else False  # 4-bit only for GPU
            low_cpu_mem_usage=True  # Optimize memory
        ).to(device)  # Explicitly move model to device



    @st.cache_resource
    def load_model(_self):
        """Load the model and block execution until it's ready."""
        with st.spinner("⏳ Loading model... Please wait."):
            _self.model = AutoModelForCausalLM.from_pretrained(
                _self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True  # Reduce memory usage
            )
        return _self.model  # Return loaded model

    def generate_response(self, prompt_txt: str) -> str:
        """Generate a response while preventing hallucinations."""
        if not prompt_txt:
            return "I'm sorry, but I couldn't find relevant information in the document."

        prompt = f"[INST] {prompt_txt} [/INST]"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048-self.MAX_NEW_TOKENS).to(self.model.device)

        output_ids = self.model.generate(
            **input_ids,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
        )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip() if "[/INST]" in response else response
