"""
Data preprocessing utilities.
"""

import re
import unicodedata
from typing import List, Optional
import html


class TextPreprocessor:
    """Text preprocessing pipeline for training data."""
    
    def __init__(
        self,
        lowercase: bool = False,  # Keep casing for better semantics
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        min_length: int = 10,
        max_length: int = 10000
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Whether to convert to lowercase
            remove_html: Whether to remove HTML tags
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_unicode: Whether to normalize Unicode characters
            min_length: Minimum text length
            max_length: Maximum text length
        """
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and unescape HTML entities."""
        # Unescape HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        return text
    
    def clean_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub(' ', text)
    
    def clean_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub(' ', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def normalize_unicode_text(self, text: str) -> str:
        """Normalize Unicode characters."""
        # NFKC normalization (compatibility decomposition, then canonical composition)
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def process(self, text: str) -> Optional[str]:
        """
        Process a single text.
        
        Args:
            text: Input text
        Returns:
            Processed text or None if filtered out
        """
        if not text or not isinstance(text, str):
            return None
        
        # Remove HTML
        if self.remove_html:
            text = self.clean_html(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.clean_urls(text)
        
        # Remove emails
        if self.remove_emails:
            text = self.clean_emails(text)
        
        # Normalize Unicode
        if self.normalize_unicode:
            text = self.normalize_unicode_text(text)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Filter by length
        if len(text) < self.min_length or len(text) > self.max_length:
            return None
        
        return text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
        Returns:
            List of processed texts (excluding filtered ones)
        """
        processed = []
        for text in texts:
            result = self.process(text)
            if result is not None:
                processed.append(result)
        return processed


def deduplicate_texts(texts: List[str], similarity_threshold: float = 0.95) -> List[str]:
    """
    Remove near-duplicate texts using simple character-level similarity.
    
    Args:
        texts: List of texts
        similarity_threshold: Minimum similarity to consider duplicate
    Returns:
        Deduplicated list of texts
    """
    from difflib import SequenceMatcher
    
    if len(texts) <= 1:
        return texts
    
    unique_texts = []
    
    for text in texts:
        is_duplicate = False
        
        # Check against existing unique texts
        for unique_text in unique_texts:
            similarity = SequenceMatcher(None, text, unique_text).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_texts.append(text)
    
    return unique_texts


def filter_language(texts: List[str], language: str = "en") -> List[str]:
    """
    Filter texts by language (simple heuristic).
    
    Args:
        texts: List of texts
        language: Target language code
    Returns:
        Filtered texts
    """
    try:
        from langdetect import detect
    except ImportError:
        print("Warning: langdetect not installed. Skipping language filter.")
        return texts
    
    filtered = []
    for text in texts:
        try:
            if detect(text) == language:
                filtered.append(text)
        except:
            # Skip texts where language detection fails
            pass
    
    return filtered


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "<p>This is a <b>test</b> with HTML tags.</p>",
        "Check out this link: https://example.com and email me@example.com",
        "   Multiple    spaces   should   be   normalized   ",
        "Short",  # Too short
        "Unicode test: café, naïve, 日本語",
        "Normal sentence that should pass through."
    ]
    
    print("Testing preprocessor:")
    print("="*50)
    for text in test_texts:
        processed = preprocessor.process(text)
        print(f"Original:  {repr(text)}")
        print(f"Processed: {repr(processed)}")
        print()
