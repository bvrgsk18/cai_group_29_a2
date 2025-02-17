from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np
import pickle
from pathlib import Path

class HybridRetriever:
    def __init__(self, embedding_manager, cache_dir: Path):
        self.embedding_manager = embedding_manager
        self.cache_dir = cache_dir
        self.bm25 = None
        self.chunks = []
        
    def init_bm25(self, chunks: List[str]):
        """Initialize BM25 with tokenized chunks."""
        self.chunks = chunks
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
    def save_state(self, filename: str):
        """Save retriever state to disk."""
        state = {
            'chunks': self.chunks,
            'bm25': self.bm25
        }
        with open(self.cache_dir / filename, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, filename: str):
        """Load retriever state from disk."""
        state_path = self.cache_dir / filename
        if state_path.exists():
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.chunks = state['chunks']
                self.bm25 = state['bm25']
            return True
        return False
        
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Combine dense and sparse retrieval results."""
        dense_indices = self.embedding_manager.search(query, k)
        dense_scores = [1.0 / (i + 1) for i in range(len(dense_indices))]
        
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
        
        combined_scores = {}
        for idx, score in zip(dense_indices, dense_scores):
            combined_scores[idx] = score
        for idx, score in zip(bm25_indices, bm25_scores[bm25_indices]):
            combined_scores[idx] = combined_scores.get(idx, 0) + score
            
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[idx], score) for idx, score in sorted_results]
