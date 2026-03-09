import faiss
import numpy as np
import pickle
import os
from pathlib import Path

class FaissEngine:
    def __init__(self, dim=128, index_type="Flat", metric="L2"):
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.metadata = [] # List to store metadata corresponding to each vector
        
        self._init_index()

    def _init_index(self):
        """Initialize FAISS index based on configuration"""
        if self.index_type == "Flat":
            if self.metric == "L2":
                self.index = faiss.IndexFlatL2(self.dim)
            elif self.metric == "IP": # Inner Product (Cosine Similarity if normalized)
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        else:
            raise NotImplementedError(f"Index type {self.index_type} not implemented yet.")

    def add(self, vectors, metas):
        """
        Add vectors and metadata to the index.
        
        Args:
            vectors (np.ndarray): Shape (N, dim), float32
            metas (list): List of N metadata objects (dicts)
        """
        if len(vectors) != len(metas):
            raise ValueError("Number of vectors and metadata items must match.")
        
        # Ensure vectors are float32
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        
        # Add to FAISS
        self.index.add(vectors)
        
        # Store metadata
        self.metadata.extend(metas)
        
    def search(self, query_vector, k=5):
        """
        Search for top-k nearest neighbors.
        
        Args:
            query_vector (np.ndarray): Shape (1, dim) or (N, dim)
            k (int): Number of neighbors to return
            
        Returns:
            distances (np.ndarray): Shape (N, k)
            results (list of list): Metadata of the neighbors
        """
        query_vector = np.ascontiguousarray(query_vector.astype('float32'))
        
        # D: distances, I: indices
        D, I = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(I)): # For each query
            query_results = []
            for j in range(k): # For each neighbor
                idx = I[i][j]
                if idx != -1: # Valid neighbor
                    meta = self.metadata[idx]
                    query_results.append(meta)
                else:
                    query_results.append(None)
            results.append(query_results)
            
        return D, results

    def save(self, index_path, meta_path):
        """Save index and metadata to disk"""
        index_path = str(index_path)
        meta_path = str(meta_path)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        # Workaround for non-ASCII paths on Windows in FAISS
        try:
            faiss.write_index(self.index, index_path)
        except Exception:
            # Try saving by changing directory
            cwd = os.getcwd()
            dir_name = os.path.dirname(index_path)
            base_name = os.path.basename(index_path)
            try:
                os.chdir(dir_name)
                faiss.write_index(self.index, base_name)
            finally:
                os.chdir(cwd)
        
        # Save metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
    def load(self, index_path, meta_path):
        """Load index and metadata from disk"""
        index_path = str(index_path)
        meta_path = str(meta_path)
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or metadata file not found.")
            
        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
        except Exception:
            # Workaround for non-ASCII paths
            cwd = os.getcwd()
            dir_name = os.path.dirname(index_path)
            base_name = os.path.basename(index_path)
            try:
                os.chdir(dir_name)
                self.index = faiss.read_index(base_name)
            finally:
                os.chdir(cwd)
        
        # Load metadata
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
    def __len__(self):
        return self.index.ntotal
