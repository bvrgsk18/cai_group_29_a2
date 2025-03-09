import os
from pathlib import Path

# Use relative paths for Streamlit Cloud compatibility
# This contains the config parameters for the application ,
# contains details of the sentence transformer used and the chunk sizes.
CACHE_DIR = Path("./cache")
SEARCH_DIR = Path("./search")
CHUNK_SIZE = 128
CHUNK_OVERLAP = 25
# SBERT_MODEL = "BAAI/bge-base-en-v1.5"
#SBERT_MODEL = "all-mpnet-base-v2"
SBERT_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K = 5

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)