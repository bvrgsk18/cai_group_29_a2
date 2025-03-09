import os
import shutil
from huggingface_hub import snapshot_download
from huggingface_hub import login
HUGGINGFACE_TOKEN="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"
login(token=HUGGINGFACE_TOKEN)
# Define model name and save directory
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_DIR = "./tinyllama_model"

# Download model
print("Downloading TinyLlama model... (This may take some time)")
snapshot_download(repo_id=MODEL_NAME, local_dir=SAVE_DIR)

print(f"Model downloaded ")
