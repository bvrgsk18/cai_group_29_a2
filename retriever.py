from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np
import pickle
from pathlib import Path

""" This class is a Hybrid retriever , it uses both dense and sparse vector search
    Dense vector search is done using the vector similarity (FAISS)
    BM25 embedding is used for Sparse vector is. 
    Combing both BM25+Dense vector makes it a hybrid search"""
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

    def bm25_normalize(self,scores):
        """Normalize scores using Min-Max scaling."""
        scores=np.array(scores)
        print("I am from bm25 normalize"+str(type(scores)))
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Combine dense and sparse retrieval results."""
        (dense_scores,dense_indices) = self.embedding_manager.search("query: "+query.lower(), k)
        tokenized_query = query.lower().split()
        bm25_scores_temp = self.bm25.get_scores(tokenized_query)
        bm25_scores = self.bm25_normalize(bm25_scores_temp)
        bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
        combined_scores = {}
        for idx, score in zip(dense_indices[0], dense_scores[0]):
            combined_scores[idx] = score
        for idx, score in zip(bm25_indices, bm25_scores):
            combined_scores[idx] = combined_scores.get(idx, 0) + score

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.chunks[idx], score) for idx, score in sorted_results]