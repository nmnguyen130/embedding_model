"""
PyTorch Dataset and DataLoader for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable, Tuple
import json
import random
from sklearn.model_selection import train_test_split


def load_triplets_with_split(
    triplets_file: str,
    val_size: float = 0.01,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load triplets from file and split into train/validation sets.
    
    Args:
        triplets_file: Path to JSONL file with triplets
        val_size: Fraction of data to use for validation (default: 0.01 = 1%)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_triplets, val_triplets)
    """
    # Load all triplets
    triplets = []
    with open(triplets_file, "r", encoding="utf-8") as f:
        for line in f:
            triplets.append(json.loads(line))
    
    print(f"Loaded {len(triplets):,} total triplets from {triplets_file}")
    
    # Split into train and validation
    train_triplets, val_triplets = train_test_split(
        triplets,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Split: {len(train_triplets):,} train, {len(val_triplets):,} val ({val_size*100:.1f}%)")
    
    return train_triplets, val_triplets


class TripletDataset(Dataset):
    """Dataset for triplet training."""
    
    def __init__(
        self,
        triplets_file: Optional[str] = None,
        triplets: Optional[List[Dict]] = None,
        tokenizer = None,
        max_length: int = 512,
        use_hard_negatives: bool = False
    ):
        """
        Initialize triplet dataset.
        
        Args:
            triplets_file: Path to JSONL file with triplets (optional if triplets provided)
            triplets: Pre-loaded triplets list (optional if triplets_file provided)
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
            use_hard_negatives: Whether dataset includes hard negatives
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_hard_negatives = use_hard_negatives
        
        # Load triplets from file or use provided list
        if triplets is not None:
            self.triplets = triplets
            print(f"Loaded {len(self.triplets):,} triplets from provided list")
        elif triplets_file is not None:
            self.triplets = []
            with open(triplets_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.triplets.append(json.loads(line))
            print(f"Loaded {len(self.triplets):,} triplets from {triplets_file}")
        else:
            raise ValueError("Either triplets_file or triplets must be provided")
        
        # Collect all texts for random negative sampling
        self.all_texts = []
        for triplet in self.triplets:
            self.all_texts.append(triplet["anchor"])
            self.all_texts.append(triplet["positive"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in self.all_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        self.all_texts = unique_texts
        
        print(f"Total unique texts for negative sampling: {len(self.all_texts)}")
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet.
        
        Returns:
            Dictionary with tokenized anchor, positive, and negative
        """
        triplet = self.triplets[idx]
        
        anchor = triplet["anchor"]
        positive = triplet["positive"]
        
        # Get negative
        if self.use_hard_negatives and "hard_negative" in triplet:
            negative = triplet["hard_negative"]
        else:
            # Random negative (different from anchor and positive)
            while True:
                negative = random.choice(self.all_texts)
                if negative != anchor and negative != positive:
                    break
        
        # Tokenize
        anchor_enc = self.tokenizer.encode(anchor, max_length=self.max_length)
        positive_enc = self.tokenizer.encode(positive, max_length=self.max_length)
        negative_enc = self.tokenizer.encode(negative, max_length=self.max_length)
        
        return {
            "anchor_input_ids": anchor_enc["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_enc["attention_mask"].squeeze(0),
            "positive_input_ids": positive_enc["input_ids"].squeeze(0),
            "positive_attention_mask": positive_enc["attention_mask"].squeeze(0),
            "negative_input_ids": negative_enc["input_ids"].squeeze(0),
            "negative_attention_mask": negative_enc["attention_mask"].squeeze(0),
        }


class PairDataset(Dataset):
    """Dataset for evaluation pairs (STS, etc.)."""
    
    def __init__(
        self,
        pairs: List[Dict[str, any]],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize pair dataset.
        
        Args:
            pairs: List of dictionaries with 'text1', 'text2', 'score'
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pair."""
        pair = self.pairs[idx]
        
        # Tokenize
        text1_enc = self.tokenizer.encode(pair["text1"], max_length=self.max_length)
        text2_enc = self.tokenizer.encode(pair["text2"], max_length=self.max_length)
        
        return {
            "text1_input_ids": text1_enc["input_ids"].squeeze(0),
            "text1_attention_mask": text1_enc["attention_mask"].squeeze(0),
            "text2_input_ids": text2_enc["input_ids"].squeeze(0),
            "text2_attention_mask": text2_enc["attention_mask"].squeeze(0),
            "score": torch.tensor(pair.get("score", 0.0), dtype=torch.float)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader with standard settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (faster GPU transfer)
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent training
    )


class InBatchNegativesDataset(Dataset):
    """
    Dataset that only loads anchor-positive pairs.
    Negatives are sampled from within the batch during training (more efficient).
    """
    
    def __init__(
        self,
        triplets_file: Optional[str] = None,
        triplets: Optional[List[Dict]] = None,
        tokenizer = None,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            triplets_file: Path to JSONL file with triplets (optional if triplets provided)
            triplets: Pre-loaded triplets list (optional if triplets_file provided)
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load triplets (but we'll only use anchor and positive)
        self.pairs = []
        
        if triplets is not None:
            # Use provided triplets
            for triplet in triplets:
                self.pairs.append({
                    "anchor": triplet["anchor"],
                    "positive": triplet["positive"]
                })
            print(f"Loaded {len(self.pairs):,} pairs from provided list")
        elif triplets_file is not None:
            # Load from file
            with open(triplets_file, "r", encoding="utf-8") as f:
                for line in f:
                    triplet = json.loads(line)
                    self.pairs.append({
                        "anchor": triplet["anchor"],
                        "positive": triplet["positive"]
                    })
            print(f"Loaded {len(self.pairs):,} pairs from {triplets_file}")
        else:
            raise ValueError("Either triplets_file or triplets must be provided")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an anchor-positive pair."""
        pair = self.pairs[idx]
        
        # Tokenize
        anchor_enc = self.tokenizer.encode(pair["anchor"], max_length=self.max_length)
        positive_enc = self.tokenizer.encode(pair["positive"], max_length=self.max_length)
        
        return {
            "anchor_input_ids": anchor_enc["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_enc["attention_mask"].squeeze(0),
            "positive_input_ids": positive_enc["input_ids"].squeeze(0),
            "positive_attention_mask": positive_enc["attention_mask"].squeeze(0),
        }


if __name__ == "__main__":
    # Test dataset
    from src.tokenizer import TextTokenizer
    
    # Create dummy triplets file for testing
    import tempfile
    import os
    
    dummy_triplets = [
        {"anchor": "This is a test sentence.", "positive": "This is another test sentence."},
        {"anchor": "Machine learning is interesting.", "positive": "Deep learning is a subset of ML."},
        {"anchor": "Python is a programming language.", "positive": "Python is great for data science."},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        for triplet in dummy_triplets:
            f.write(json.dumps(triplet) + '\n')
        temp_file = f.name
    
    print(f"Created temporary file: {temp_file}")
    
    # Note: This test requires a trained tokenizer
    # For now, we'll just print the structure
    print("\nDataset structure created successfully!")
    print(f"  TripletDataset: {TripletDataset.__name__}")
    print(f"  PairDataset: {PairDataset.__name__}")
    print(f"  InBatchNegativesDataset: {InBatchNegativesDataset.__name__}")
    
    # Clean up
    os.unlink(temp_file)
