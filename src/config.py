import os
from pathlib import Path

# Use relative paths for Streamlit Cloud compatibility
CACHE_DIR = Path("./cache")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)