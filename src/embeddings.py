from sentence_transformers import SentenceTransformer,util
import numpy as np
from typing import List
import faiss
from pathlib import Path

"""This class contains the methods needed for embedding a chunk """
class EmbeddingManager:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        self.index = None
        self.chunks = []
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        threshold=0.1
        reference_sentence = "This sentence contains meaningful information about a topic."
        sentence_embeddings = self.model.encode(texts, convert_to_tensor=True,normalize_embeddings=True).numpy()
        print(sentence_embeddings)
        reference_embedding = self.model.encode(reference_sentence, convert_to_tensor=True,normalize_embeddings=True).numpy()
        print(reference_embedding)
        similarities = util.cos_sim(sentence_embeddings, reference_embedding).squeeze(1)
        print(similarities)

        return sentence_embeddings
        # return self.model.encode(texts, convert_to_tensor=True,normalize_embeddings=True).numpy()
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search.
        This is the dense vector created as part of embedding."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
    
    def search(self, query: str, k: int = 5) -> tuple[int,int]:
        """Search for most similar chunks."""
        query_embedding = self.embed_texts([query])
        scores, indices = self.index.search(query_embedding, k)
        return (scores,indices)
    
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
