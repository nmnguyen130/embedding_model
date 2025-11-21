"""
Inference utilities for the embedding model.
"""

import torch
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import json


class EmbeddingModel:
    """Inference wrapper for embedding model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize embedding model for inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        # Import here to avoid circular imports
        from src.model import create_model, ModelConfig
        from src.tokenizer import TextTokenizer
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        config = ModelConfig(**checkpoint["config"])
        
        # Create and load model
        self.model = create_model(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = TextTokenizer(tokenizer_path)
        
        self.embedding_dim = config.output_embedding_dim
        
        print(f"Model loaded successfully!")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
    
    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            return_numpy: Whether to return numpy array (otherwise torch tensor)
        Returns:
            Embeddings array/tensor
        """
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Encode in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer.encode(batch_texts)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Encode
            embeddings = self.model(input_ids, attention_mask)
            
            # Additional normalization (model already does this, but ensure it)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings)
        
        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy if requested
        if return_numpy:
            all_embeddings = all_embeddings.cpu().numpy()
            
            # Return single embedding if single text
            if is_single:
                return all_embeddings[0]
        else:
            # Return single embedding if single text
            if is_single:
                return all_embeddings[0]
        
        return all_embeddings
    
    def similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, np.ndarray]:
        """
        Compute cosine similarity between text(s).
        
        Args:
            text1: First text or list of texts
            text2: Second text or list of texts
        Returns:
            Similarity score(s) in range [-1, 1]
        """
        # Encode
        emb1 = self.encode(text1, return_numpy=True, normalize=True)
        emb2 = self.encode(text2, return_numpy=True, normalize=True)
        
        # Handle single vs batch
        if emb1.ndim ==  1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = np.sum(emb1 * emb2, axis=1)
        
        # Return single value if inputs were single texts
        if isinstance(text1, str) and isinstance(text2, str):
            return float(similarities[0])
        
        return similarities
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts from candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
        Returns:
            List of (text, similarity_score) tuples
        """
        # Encode query
        query_emb = self.encode(query, return_numpy=True, normalize=True)
        
        # Encode candidates
        candidate_embs = self.encode(candidates, return_numpy=True, normalize=True)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


def load_model(checkpoint_path: str, tokenizer_path: str, device: Optional[str] = None) -> EmbeddingModel:
    """
    Load embedding model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device to run on
    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(checkpoint_path, tokenizer_path, device)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.tokenizer)
    
    # Test encoding
    print("\n" + "="*60)
    print("Testing encoding...")
    print("="*60)
    
    test_texts = [
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "Natural language processing is a subfield of AI."
    ]
    
    embeddings = model.encode(test_texts)
    print(f"\nEncoded {len(test_texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings are normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
    
    # Test similarity
    print("\n" + "="*60)
    print("Testing similarity...")
    print("="*60)
    
    text1 = "I love machine learning"
    text2 = "Machine learning is great"
    text3 = "The weather is nice today"
    
    sim_12 = model.similarity(text1, text2)
    sim_13 = model.similarity(text1, text3)
    sim_23 = model.similarity(text2, text3)
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print(f"\nSimilarity(1, 2): {sim_12:.4f}")
    print(f"Similarity(1, 3): {sim_13:.4f}")
    print(f"Similarity(2, 3): {sim_23:.4f}")
    
    # Test search
    print("\n" + "="*60)
    print("Testing search...")
    print("="*60)
    
    query = "artificial intelligence"
    candidates = [
        "Machine learning and deep learning",
        "The weather is sunny",
        "AI and neural networks",
        "Cooking recipes",
        "Natural language understanding"
    ]
    
    results = model.find_similar(query, candidates, top_k=3)
    
    print(f"\nQuery: {query}")
    print("\nTop 3 results:")
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. {text} (score: {score:.4f})")
