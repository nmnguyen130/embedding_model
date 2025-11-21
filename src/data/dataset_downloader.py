"""
Dataset downloader for training data.
Downloads Wikipedia, SNLI, Quora, and MS MARCO datasets.
"""

import os
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json


class DatasetDownloader:
    """Download and prepare datasets for training."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Directory to save downloaded datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_wikipedia(
        self,
        max_samples: int = None,
        language: str = "en"
    ) -> Dataset:
        """
        Download Wikipedia dataset.
        
        Args:
            max_samples: Maximum number of samples (None for all)
            language: Language code
        Returns:
            Wikipedia dataset or None if failed
        """
        print(f"Downloading Wikipedia ({language})...")
        
        cache_file = os.path.join(self.data_dir, f"wikipedia_{language}.jsonl")
        
        # Check if cache exists and is valid
        if os.path.exists(cache_file):
            # Validate cache file is not empty
            if os.path.getsize(cache_file) > 100:  # At least 100 bytes
                print(f"Loading from cache: {cache_file}")
                try:
                    dataset = load_dataset("json", data_files=cache_file, split="train")
                    if len(dataset) > 0:
                        print(f"Wikipedia: {len(dataset)} samples (from cache)")
                        return dataset
                except Exception as e:
                    print(f"Cache file corrupted: {e}")
            
            # Remove invalid cache
            print("Removing invalid cache file...")
            os.remove(cache_file)
        
        # Download with streaming for large dataset
        # Use new wikimedia/wikipedia format (not deprecated script)
        try:
            print("Downloading from wikimedia/wikipedia...")
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",  # Latest snapshot
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            
            # Process and save
            samples = []
            for i, item in enumerate(tqdm(dataset, desc="Processing Wikipedia")):
                if max_samples and i >= max_samples:
                    break
                
                text = item["text"].strip()
                if len(text) > 50:  # Filter very short texts
                    samples.append({"text": text, "source": "wikipedia"})
            
            if len(samples) == 0:
                print("Warning: No valid samples from Wikipedia")
                return None
            
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            dataset = load_dataset("json", data_files=cache_file, split="train")
            print(f"Wikipedia: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"Error loading Wikipedia: {e}")
            print("Skipping Wikipedia dataset...")
            return None
    
    def download_snli(self, split: str = "train") -> Dataset:
        """
        Download SNLI dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
        Returns:
            SNLI dataset
        """
        print(f"Downloading SNLI ({split})...")
        
        cache_file = os.path.join(self.data_dir, f"snli_{split}.jsonl")
        
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            dataset = load_dataset("json", data_files=cache_file, split="train")
            return dataset
        
        # Download SNLI
        dataset = load_dataset("snli", split=split)
        
        # Process: create pairs from entailment/contradiction
        samples = []
        for item in tqdm(dataset, desc=f"Processing SNLI {split}"):
            if item["label"] == -1:  # Skip invalid samples
                continue
            
            premise = item["premise"].strip()
            hypothesis = item["hypothesis"].strip()
            label = item["label"]  # 0: entailment, 1: neutral, 2: contradiction
            
            if len(premise) > 10 and len(hypothesis) > 10:
                samples.append({
                    "text1": premise,
                    "text2": hypothesis,
                    "label": label,
                    "source": "snli"
                })
        
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        dataset = load_dataset("json", data_files=cache_file, split="train")
        print(f"SNLI {split}: {len(dataset)} samples")
        return dataset
    
    def download_quora(self, split: str = "train") -> Dataset:
        """
        Download Quora Question Pairs dataset.
        
        Args:
            split: Dataset split
        Returns:
            Quora dataset
        """
        print(f"Downloading Quora Question Pairs...")
        
        cache_file = os.path.join(self.data_dir, f"quora_{split}.jsonl")
        
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            dataset = load_dataset("json", data_files=cache_file, split="train")
            return dataset
        
        try:
            # Download Quora dataset (might require authentication)
            dataset = load_dataset("quora", split=split)
        except Exception as e:
            print(f"Could not download Quora dataset: {e}")
            print("Skipping Quora dataset...")
            return None
        
        # Process: create question pairs
        samples = []
        for item in tqdm(dataset, desc=f"Processing Quora {split}"):
            questions = item["questions"]["text"]
            is_duplicate = item["is_duplicate"]
            
            if len(questions) == 2:
                q1, q2 = questions
                if len(q1) > 10 and len(q2) > 10:
                    samples.append({
                        "text1": q1.strip(),
                        "text2": q2.strip(),
                        "label": 1 if is_duplicate else 0,  # 1: duplicate, 0: not duplicate
                        "source": "quora"
                    })
        
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        dataset = load_dataset("json", data_files=cache_file, split="train")
        print(f"Quora {split}: {len(dataset)} samples")
        return dataset
    
    def download_msmarco(
        self,
        max_samples: int = None,
        subset: str = "v1.1"
    ) -> Dataset:
        """
        Download MS MARCO dataset.
        
        Args:
            max_samples: Maximum number of samples
            subset: Dataset version
        Returns:
            MS MARCO dataset
        """
        print(f"Downloading MS MARCO...")
        
        cache_file = os.path.join(self.data_dir, f"msmarco_{subset}.jsonl")
        
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            dataset = load_dataset("json", data_files=cache_file, split="train")
            return dataset
        
        try:
            # MS MARCO is large, use streaming
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
        except Exception as e:
            print(f"Could not download MS MARCO dataset: {e}")
            print("Skipping MS MARCO dataset...")
            return None
        
        # Process: create query-passage pairs
        samples = []
        for i, item in enumerate(tqdm(dataset, desc="Processing MS MARCO")):
            if max_samples and i >= max_samples:
                break
            
            query = item["query"].strip()
            passages = item["passages"]["passage_text"]
            is_selected = item["passages"]["is_selected"]
            
            # Find positive passage
            for passage, selected in zip(passages, is_selected):
                if selected == 1 and len(query) > 10 and len(passage) > 20:
                    samples.append({
                        "text1": query,
                        "text2": passage.strip(),
                        "label": 1,  # Relevant pair
                        "source": "msmarco"
                    })
                    break
        
        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        dataset = load_dataset("json", data_files=cache_file, split="train")
        print(f"MS MARCO: {len(dataset)} samples")
        return dataset
    
    def download_all(
        self,
        include_wikipedia: bool = True,
        include_snli: bool = True,
        include_quora: bool = True,
        include_msmarco: bool = False,
        max_wikipedia_samples: int = 100000,
        max_msmarco_samples: int = 50000
    ) -> Dict[str, Dataset]:
        """
        Download all datasets.
        
        Args:
            include_wikipedia: Whether to include Wikipedia
            include_snli: Whether to include SNLI
            include_quora: Whether to include Quora
            include_msmarco: Whether to include MS MARCO
            max_wikipedia_samples: Max Wikipedia samples
            max_msmarco_samples: Max MS MARCO samples
        Returns:
            Dictionary of datasets
        """
        datasets = {}
        
        if include_wikipedia:
            wiki = self.download_wikipedia(max_samples=max_wikipedia_samples)
            if wiki is not None:
                datasets["wikipedia"] = wiki
        
        if include_snli:
            datasets["snli"] = self.download_snli(split="train")
        
        if include_quora:
            quora = self.download_quora(split="train")
            if quora is not None:
                datasets["quora"] = quora
        
        if include_msmarco:
            msmarco = self.download_msmarco(max_samples=max_msmarco_samples)
            if msmarco is not None:
                datasets["msmarco"] = msmarco
        
        # Print summary
        print("\n" + "="*50)
        print("Download Summary:")
        print("="*50)
        for name, dataset in datasets.items():
            print(f"{name}: {len(dataset):,} samples")
        print(f"Total: {sum(len(d) for d in datasets.values()):,} samples")
        
        return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to save datasets")
    parser.add_argument("--wikipedia", action="store_true",
                        help="Download Wikipedia")
    parser.add_argument("--snli", action="store_true",
                        help="Download SNLI")
    parser.add_argument("--quora", action="store_true",
                        help="Download Quora")
    parser.add_argument("--msmarco", action="store_true",
                        help="Download MS MARCO")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--max-wiki-samples", type=int, default=100000,
                        help="Max Wikipedia samples")
    parser.add_argument("--max-msmarco-samples", type=int, default=50000,
                        help="Max MS MARCO samples")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(data_dir=args.data_dir)
    
    if args.all:
        downloader.download_all(
            include_wikipedia=True,
            include_snli=True,
            include_quora=True,
            include_msmarco=True,
            max_wikipedia_samples=args.max_wiki_samples,
            max_msmarco_samples=args.max_msmarco_samples
        )
    else:
        if args.wikipedia:
            downloader.download_wikipedia(max_samples=args.max_wiki_samples)
        if args.snli:
            downloader.download_snli()
        if args.quora:
            downloader.download_quora()
        if args.msmarco:
            downloader.download_msmarco(max_samples=args.max_msmarco_samples)
