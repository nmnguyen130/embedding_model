"""
Tokenizer utilities for encoding and decoding text.
"""

import torch
from typing import List, Union, Optional
from tokenizers import Tokenizer
import os


class TextTokenizer:
    """Wrapper around HuggingFace tokenizer for easy usage."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_path: Path to trained tokenizer JSON file
            max_length: Maximum sequence length
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
        # Enable padding and truncation
        self.tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
            length=max_length
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        
        # Get special token IDs
        self.pad_token_id = 0
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        
        # Vocab size
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> dict:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add [CLS] and [SEP]
            return_tensors: Return type ("pt" for PyTorch, "np" for NumPy, None for list)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length (overrides default)
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_length
        
        # Handle single text vs batch
        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]
        
        # Encode
        encodings = self.tokenizer.encode_batch(
            text,
            add_special_tokens=add_special_tokens
        )
        
        # Extract IDs and attention masks
        input_ids = [enc.ids for enc in encodings]
        attention_mask = [enc.attention_mask for enc in encodings]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np
            input_ids = np.array(input_ids)
            attention_mask = np.array(attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(
        self,
        token_ids_batch: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs.
        
        Args:
            token_ids_batch: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        
        return self.tokenizer.decode_batch(token_ids_batch, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> dict:
        """Shortcut for encode."""
        return self.encode(text, **kwargs)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> dict:
        """Get vocabulary mapping."""
        return self.tokenizer.get_vocab()


if __name__ == "__main__":
    # Test tokenizer
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py <tokenizer_path>")
        sys.exit(1)
    
    tokenizer_path = sys.argv[1]
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(tokenizer_path, max_length=128)
    
    print(f"Tokenizer loaded from {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max length: {tokenizer.max_length}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"CLS token ID: {tokenizer.cls_token_id}")
    print(f"SEP token ID: {tokenizer.sep_token_id}")
    
    # Test encoding
    print("\n" + "="*50)
    print("Testing encoding...")
    print("="*50)
    
    test_texts = [
        "This is a test sentence.",
        "Another test with different length.",
        "Short one."
    ]
    
    encoded = tokenizer.encode(test_texts)
    
    print(f"\nInput texts: {test_texts}")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"\nFirst sequence IDs: {encoded['input_ids'][0]}")
    print(f"First sequence mask: {encoded['attention_mask'][0]}")
    
    # Test decoding
    print("\n" + "="*50)
    print("Testing decoding...")
    print("="*50)
    
    decoded = tokenizer.batch_decode(encoded['input_ids'])
    for original, decoded_text in zip(test_texts, decoded):
        print(f"Original: {original}")
        print(f"Decoded:  {decoded_text}")
        print()
