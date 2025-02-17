from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import faiss
import pickle
from pathlib import Path

class EmbeddingManager:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.index = None
        self.chunks = []
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        return self.model.encode(texts, convert_to_tensor=True).numpy()
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def save_index(self, filename: str):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.cache_dir / filename))
        
    def load_index(self, filename: str):
        """Load FAISS index from disk."""
        index_path = self.cache_dir / filename
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            return True
        return False
