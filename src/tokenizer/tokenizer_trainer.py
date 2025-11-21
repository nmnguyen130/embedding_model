"""
WordPiece tokenizer trainer.
Trains a tokenizer from scratch on the provided corpus.
"""

import os
from typing import List, Optional, Iterator
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tqdm import tqdm


class TokenizerTrainer:
    """Train WordPiece tokenizer from scratch."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: Special tokens to include
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        if special_tokens is None:
            self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        else:
            self.special_tokens = special_tokens
        
        # Initialize tokenizer with WordPiece model
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
        # Set normalizer (lowercase + unicode normalization)
        self.tokenizer.normalizer = Sequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])
        
        # Set pre-tokenizer (split on whitespace and punctuation)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Set post-processor (add [CLS] and [SEP] tokens)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.special_tokens.index("[CLS]")),
                ("[SEP]", self.special_tokens.index("[SEP]")),
            ],
        )
    
    def train_from_iterator(
        self,
        text_iterator: Iterator[str],
        show_progress: bool = True
    ) -> Tokenizer:
        """
        Train tokenizer from text iterator.
        
        Args:
            text_iterator: Iterator yielding text strings
            show_progress: Whether to show progress bar
        Returns:
            Trained tokenizer
        """
        # Create trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress
        )
        
        # Train
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        
        return self.tokenizer
    
    def train_from_files(
        self,
        files: List[str],
        show_progress: bool = True
    ) -> Tokenizer:
        """
        Train tokenizer from text files.
        
        Args:
            files: List of file paths
            show_progress: Whether to show progress bar
        Returns:
            Trained tokenizer
        """
        # Create trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress
        )
        
        # Train
        self.tokenizer.train(files, trainer=trainer)
        
        return self.tokenizer
    
    def save(self, path: str):
        """Save trained tokenizer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str) -> Tokenizer:
        """Load trained tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)
        return self.tokenizer


def train_tokenizer_on_datasets(
    datasets: List[str],
    data_dir: str,
    output_path: str,
    vocab_size: int = 30000,
    max_samples: Optional[int] = None
) -> Tokenizer:
    """
    Train tokenizer on multiple datasets.
    
    Args:
        datasets: List of dataset names (e.g., ["wikipedia", "snli"])
        data_dir: Directory containing dataset files
        output_path: Path to save trained tokenizer
        vocab_size: Target vocabulary size
        max_samples: Maximum samples to use (for testing)
    Returns:
        Trained tokenizer
    """
    from datasets import load_dataset
    
    def text_iterator():
        """Generator that yields text from all datasets."""
        for dataset_name in datasets:
            print(f"Loading {dataset_name}...")
            
            if dataset_name == "wikipedia":
                try:
                    # Use new wikimedia/wikipedia dataset (not deprecated script)
                    dataset = load_dataset(
                        "wikimedia/wikipedia",
                        "20231101.en",  # Latest snapshot
                        split="train",
                        streaming=True,
                        trust_remote_code=False
                    )
                    count = 0
                    for item in dataset:
                        yield item["text"]
                        count += 1
                        if max_samples and count >= max_samples:
                            break
                except Exception as e:
                    print(f"Warning: Could not load Wikipedia: {e}")
                    print("Skipping Wikipedia dataset...")
            
            elif dataset_name == "snli":
                # Load SNLI dataset
                dataset = load_dataset("snli", split="train")
                count = 0
                for item in dataset:
                    if item["label"] != -1:  # Skip invalid samples
                        yield item["premise"]
                        yield item["hypothesis"]
                        count += 1
                        if max_samples and count >= max_samples:
                            break
            
            elif dataset_name == "quora":
                # Load Quora Question Pairs
                try:
                    dataset = load_dataset("quora", split="train")
                    count = 0
                    for item in dataset:
                        yield item["questions"]["text"][0]
                        yield item["questions"]["text"][1]
                        count += 1
                        if max_samples and count >= max_samples:
                            break
                except Exception as e:
                    print(f"Warning: Could not load Quora: {e}")
                    print("Skipping Quora dataset...")
            
            else:
                print(f"Unknown dataset: {dataset_name}")
    
    # Train tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    tokenizer = trainer.train_from_iterator(text_iterator(), show_progress=True)
    
    # Save tokenizer
    trainer.save(output_path)
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train WordPiece tokenizer")
    parser.add_argument("--output", type=str, default="./tokenizer/tokenizer.json",
                        help="Output path for trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=30000,
                        help="Vocabulary size")
    parser.add_argument("--datasets", type=str, nargs="+", default=["wikipedia", "snli"],
                        help="Datasets to train on")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    
    args = parser.parse_args()
    
    # Train tokenizer
    tokenizer = train_tokenizer_on_datasets(
        datasets=args.datasets,
        data_dir=args.data_dir,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_samples=args.max_samples
    )
    
    # Test tokenizer
    print("\n" + "="*50)
    print("Testing tokenizer...")
    print("="*50)
    
    test_texts = [
        "This is a simple test sentence.",
        "How does the tokenizer handle this?",
        "Machine learning and natural language processing."
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"\nText: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
