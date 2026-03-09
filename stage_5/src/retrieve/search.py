import sys
import os

# Assuming sys.path includes stage_4 root
try:
    from src.indexer.faiss_engine import FaissEngine
except ImportError:
    pass

class Retriever:
    def __init__(self, config):
        self.config = config
        self.index_file = config['index']['index_file']
        self.meta_file = config['index']['meta_file']
        self.top_k = config['pipeline']['top_k']
        
        self.engine = self._load_engine()

    def _load_engine(self):
        # We assume parameters match what was used in build_index.py
        # Ideally these should also be in config or read from a saved config
        engine = FaissEngine(dim=128, index_type="Flat", metric="L2")
        print(f"Loading index from {self.index_file}")
        engine.load(self.index_file, self.meta_file)
        return engine

    def search(self, embedding):
        """
        Args:
            embedding: (1, dim) np.ndarray
        Returns:
            distances: (1, k)
            results: list of list of metadata
        """
        D, results = self.engine.search(embedding, k=self.top_k)
        return D, results
