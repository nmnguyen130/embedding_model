"""
STS Benchmark evaluation.
Evaluates the model on Semantic Textual Similarity Benchmark.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import List, Tuple, Dict
from datasets import load_dataset
from tqdm import tqdm


def load_sts_benchmark(split: str = "test") -> List[Tuple[str, str, float]]:
    """
    Load STS Benchmark dataset.
    
    Args:
        split: Dataset split ("train", "validation", "test")
    Returns:
        List of (sentence1, sentence2, score) tuples
    """
    print(f"Loading STS-B {split} split...")
    
    dataset = load_dataset("mteb/stsbenchmark-sts", split=split)
    
    pairs = []
    for item in dataset:
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        score = item["score"]  # Score from 0 to 5
        
        pairs.append((sentence1, sentence2, score))
    
    print(f"Loaded {len(pairs)} pairs")
    return pairs


@torch.no_grad()
def evaluate_sts(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    split: str = "test",
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate model on STS Benchmark.
    
    Args:
        model: Embedding model
        tokenizer: Text tokenizer
        device: Device to run on
        split: Dataset split
        batch_size: Batch size for encoding
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Load dataset
    pairs = load_sts_benchmark(split)
    
    # Extract sentences and scores
    sentences1 = [p[0] for p in pairs]
    sentences2 = [p[1] for p in pairs]
    true_scores = np.array([p[2] for p in pairs])
    
    # Encode sentences
    print("Encoding sentences...")
    embeddings1 = encode_batch(model, tokenizer, sentences1, device, batch_size)
    embeddings2 = encode_batch(model, tokenizer, sentences2, device, batch_size)
    
    # Compute cosine similarities
    print("Computing similarities...")
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    # Normalize similarities to 0-5 scale (like STS scores)
    # Cosine similarity is in [-1, 1], convert to [0, 5]
    predicted_scores = (similarities + 1) * 2.5
    
    # Compute correlation
    spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
    pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)
    
    results = {
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "num_pairs": len(pairs)
    }
    
    return results


def encode_batch(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 64
) -> np.ndarray:
    """
    Encode a batch of texts.
    
    Args:
        model: Embedding model
        tokenizer: Text tokenizer
        texts: List of texts to encode
        device: Device to run on
        batch_size: Batch size
    Returns:
        Numpy array of embeddings [num_texts, embedding_dim]
    """
    model.eval()
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer.encode(batch_texts)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Encode
        with torch.no_grad():
            batch_embeddings = model(input_ids, attention_mask)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return embeddings


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [N, D]
    Returns:
        Cosine similarities [N]
    """
    # Normalize embeddings
    embeddings1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
    embeddings2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.sum(embeddings1_norm * embeddings2_norm, axis=1)
    
    return similarities


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.model import create_model, ModelConfig
    from src.tokenizer import TextTokenizer
    
    parser = argparse.ArgumentParser(description="Evaluate on STS Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load config
    config = ModelConfig(**checkpoint["config"])
    
    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (step {checkpoint['global_step']})")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = TextTokenizer(args.tokenizer)
    print("Tokenizer loaded")
    
    # Evaluate
    print(f"\nEvaluating on STS-B {args.split} split...")
    print("="*60)
    
    results = evaluate_sts(
        model=model,
        tokenizer=tokenizer,
        device=device,
        split=args.split,
        batch_size=args.batch_size
    )
    
    print("\nResults:")
    print("="*60)
    print(f"Spearman Correlation: {results['spearman_correlation']:.4f}")
    print(f"Pearson Correlation:  {results['pearson_correlation']:.4f}")
    print(f"Number of pairs:      {results['num_pairs']}")
    print("="*60)
