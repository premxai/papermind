import os
from typing import List, Dict
import numpy as np
from openai import OpenAI


class Embedder:
    """Generates embeddings for text chunks using OpenAI or local models."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize embedder.
        
        Args:
            model: Embedding model to use
            api_key: OpenAI API key (or None to read from env)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.use_openai = True
        else:
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_openai = False
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.use_openai:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return np.array(response.data[0].embedding)
        else:
            return self.client.encode(text)
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with embeddings added
        """
        if self.use_openai:
            texts = [chunk['text'] for chunk in chunks]
            
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(embeddings)
            
            for chunk, embedding in zip(chunks, all_embeddings):
                chunk['embedding'] = embedding
        else:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.client.encode(texts, show_progress_bar=True)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
        
        return chunks
