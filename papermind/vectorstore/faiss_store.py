import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Optional


class FAISSStore:
    """Vector database using FAISS for efficient similarity search."""
    
    def __init__(self, dimension: int = 1536, index_path: str = "papermind/embeddings/faiss.index"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Path to save/load FAISS index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path.replace('.index', '_metadata.pkl')
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        if os.path.exists(index_path):
            self.load()
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        if embeddings.shape[1] != self.dimension:
            self.dimension = embeddings.shape[1]
            old_metadata = self.metadata
            self.index = faiss.IndexFlatL2(self.dimension)
            if old_metadata:
                self.metadata = old_metadata
        
        self.index.add(embeddings)
        
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
            self.metadata.append(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of similar chunks with distances
        """
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(dist)
                results.append(result)
        
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
    
    def clear(self):
        """Clear all data from the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'index_size_mb': os.path.getsize(self.index_path) / 1024 / 1024 if os.path.exists(self.index_path) else 0
        }
