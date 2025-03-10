import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
#from .config import HUGGINGFACE_TOKEN, MAX_NEW_TOKENS
# Secure Hugging Face login using environment variables
#login(token=os.getenv("HUGGINGFACE_TOKEN"))
#login(token=HUGGINGFACE_TOKEN)
import streamlit as st
class LlmHandler:
    def __init__(self, MAX_NEW_TOKENS,model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        huggingface_token = st.secrets["HUGGINGFACE_TOKEN"] 
        login(token=huggingface_token)
        self.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float16 for efficiency
            device_map="auto",
            #load_in_4bit=True  # Enable 4-bit quantization for low memory usage
        )

    def generate_response(self, prompt_txt: str) -> str:
        """Generate a response using retrieved context chunks while preventing hallucinations."""

        if not prompt_txt:
            return "I'm sorry, but I couldn't find relevant information in the document."

        prompt = f"""[INST] {prompt_txt} [/INST]"""
        # Tokenize & truncate the input
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048-self.MAX_NEW_TOKENS).to(self.model.device)

        # Generate response
        output_ids = self.model.generate(
            **input_ids,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,  # Lower temperature for factual consistency
            top_p=0.7,  # Lower top-p to reduce randomness
            repetition_penalty=1.2,  # Higher penalty to avoid repetitive or generic answers
        )

        # Decode output
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the model-generated answer after the prompt
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        # Enforce strict response control
        #if "I'm sorry, but I couldn't find relevant information in the document." not in response and not any(term.lower() in response.lower() for term in prompt.split()):
        #    return "I cannot find information from the provided financial data."

        return response
