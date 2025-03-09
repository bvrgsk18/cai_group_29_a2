import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


class LlmHandler:
    def __init__(self, MAX_NEW_TOKENS, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        huggingface_token = st.secrets["HUGGINGFACE_TOKEN"] 
        login(token=huggingface_token)

        # Streamlit Cloud runs on CPU only, so enforce CPU usage
        device = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use FP32 for CPU compatibility
            low_cpu_mem_usage=True  # Optimize memory usage for cloud
        ).to(device)  # Ensure the model is explicitly moved to CPU


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
