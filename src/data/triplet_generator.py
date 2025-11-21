"""
Triplet generation for contrastive learning.
Creates (anchor, positive, negative) triplets from datasets.
"""

import random
from typing import List, Tuple, Dict, Any, Optional
from datasets import Dataset
import numpy as np
from tqdm import tqdm


class TripletGenerator:
    """Generate training triplets for contrastive learning."""
    
    def __init__(
        self,
        hard_negative_ratio: float = 0.5,
        use_bm25: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize triplet generator.
        
        Args:
            hard_negative_ratio: Ratio of hard negatives vs random negatives
            use_bm25: Whether to use BM25 for hard negative mining
            random_seed: Random seed for reproducibility
        """
        self.hard_negative_ratio = hard_negative_ratio
        self.use_bm25 = use_bm25
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if use_bm25:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25_class = BM25Okapi
            except ImportError:
                print("Warning: rank-bm25 not installed. Using random negatives only.")
                self.use_bm25 = False
    
    def from_wikipedia(self, dataset: Dataset, max_triplets: int = None) -> List[Dict[str, str]]:
        """
        Generate triplets from Wikipedia.
        Strategy: Use sentences from same article as positives.
        
        Args:
            dataset: Wikipedia dataset
            max_triplets: Maximum number of triplets to generate
        Returns:
            List of triplets
        """
        triplets = []
        
        for item in tqdm(dataset, desc="Generating Wikipedia triplets"):
            text = item["text"]
            
            # Split into sentences (simple splitting)
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            if len(sentences) < 2:
                continue
            
            # Create pairs from consecutive sentences (as they are semantically related)
            for i in range(len(sentences) - 1):
                anchor = sentences[i]
                positive = sentences[i + 1]
                
                # Random negative from different article (will be sampled during training)
                triplets.append({
                    "anchor": anchor,
                    "positive": positive,
                    "source": "wikipedia"
                })
                
                if max_triplets and len(triplets) >= max_triplets:
                    return triplets
        
        return triplets
    
    def from_snli(self, dataset: Dataset) -> List[Dict[str, str]]:
        """
        Generate triplets from SNLI.
        Strategy: Entailment as positive, contradiction/neutral as negative.
        
        Args:
            dataset: SNLI dataset
        Returns:
            List of triplets
        """
        triplets = []
        
        # Group by premise
        premise_groups = {}
        for item in dataset:
            premise = item["text1"]
            hypothesis = item["text2"]
            label = item["label"]
            
            if premise not in premise_groups:
                premise_groups[premise] = {"entailment": [], "contradiction": [], "neutral": []}
            
            if label == 0:  # Entailment
                premise_groups[premise]["entailment"].append(hypothesis)
            elif label == 1:  # Neutral
                premise_groups[premise]["neutral"].append(hypothesis)
            elif label == 2:  # Contradiction
                premise_groups[premise]["contradiction"].append(hypothesis)
        
        # Create triplets
        for premise, groups in tqdm(premise_groups.items(), desc="Generating SNLI triplets"):
            # Use entailment as positive
            for positive in groups["entailment"]:
                triplets.append({
                    "anchor": premise,
                    "positive": positive,
                    "source": "snli"
                })
        
        return triplets
    
    def from_quora(self, dataset: Dataset) -> List[Dict[str, str]]:
        """
        Generate triplets from Quora.
        Strategy: Duplicate questions as positive pairs.
        
        Args:
            dataset: Quora dataset
        Returns:
            List of triplets
        """
        triplets = []
        
        for item in tqdm(dataset, desc="Generating Quora triplets"):
            if item["label"] == 1:  # Duplicate questions
                triplets.append({
                    "anchor": item["text1"],
                    "positive": item["text2"],
                    "source": "quora"
                })
        
        return triplets
    
    def from_msmarco(self, dataset: Dataset) -> List[Dict[str, str]]:
        """
        Generate triplets from MS MARCO.
        Strategy: Query and relevant passage as positive pair.
        
        Args:
            dataset: MS MARCO dataset
        Returns:
            List of triplets
        """
        triplets = []
        
        for item in tqdm(dataset, desc="Generating MS MARCO triplets"):
            if item["label"] == 1:  # Relevant pair
                triplets.append({
                    "anchor": item["text1"],  # Query
                    "positive": item["text2"],  # Passage
                    "source": "msmarco"
                })
        
        return triplets
    
    def generate_triplets(
        self,
        datasets: Dict[str, Dataset],
        max_triplets_per_dataset: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Generate triplets from all datasets.
        
        Args:
            datasets: Dictionary of datasets
            max_triplets_per_dataset: Max triplets per dataset
        Returns:
            Combined list of triplets
        """
        all_triplets = []
        
        for name, dataset in datasets.items():
            print(f"\nProcessing {name}...")
            
            if name == "wikipedia":
                triplets = self.from_wikipedia(dataset, max_triplets=max_triplets_per_dataset)
            elif name == "snli":
                triplets = self.from_snli(dataset)
            elif name == "quora":
                triplets = self.from_quora(dataset)
            elif name == "msmarco":
                triplets = self.from_msmarco(dataset)
            else:
                print(f"Unknown dataset: {name}")
                continue
            
            all_triplets.extend(triplets)
            print(f"  Generated {len(triplets):,} triplets from {name}")
        
        # Shuffle all triplets
        random.shuffle(all_triplets)
        
        print(f"\nTotal triplets: {len(all_triplets):,}")
        return all_triplets
    
    def add_hard_negatives_bm25(
        self,
        triplets: List[Dict[str, str]],
        corpus_texts: List[str],
        top_k: int = 100
    ) -> List[Dict[str, str]]:
        """
        Add hard negatives using BM25 ranking.
        
        Args:
            triplets: List of triplets (without negatives)
            corpus_texts: All available texts for negative sampling
            top_k: Number of top BM25 results to consider
        Returns:
            Triplets with hard negatives added
        """
        if not self.use_bm25:
            print("BM25 not available. Skipping hard negative mining.")
            return triplets
        
        print("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in tqdm(corpus_texts)]
        bm25 = self.bm25_class(tokenized_corpus)
        
        print("Finding hard negatives...")
        for triplet in tqdm(triplets):
            anchor = triplet["anchor"]
            positive = triplet["positive"]
            
            # Get BM25 scores for anchor
            tokenized_query = anchor.lower().split()
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[-top_k:]
            
            # Find hard negative (similar to anchor but not the positive)
            candidates = [corpus_texts[idx] for idx in top_indices
                         if corpus_texts[idx] != positive and corpus_texts[idx] != anchor]
            
            if candidates:
                # Pick a random hard negative from top candidates
                triplet["hard_negative"] = random.choice(candidates[:20])
        
        return triplets


def create_training_triplets(
    datasets: Dict[str, Dataset],
    output_file: str,
    max_triplets_per_dataset: Optional[int] = None,
    use_hard_negatives: bool = False
) -> List[Dict[str, str]]:
    """
    Create training triplets and save to file.
    
    Args:
        datasets: Dictionary of datasets
        output_file: Path to save triplets
        max_triplets_per_dataset: Max triplets per dataset
        use_hard_negatives: Whether to mine hard negatives
    Returns:
        List of triplets
    """
    import json
    import os
    
    generator = TripletGenerator(use_bm25=use_hard_negatives)
    triplets = generator.generate_triplets(datasets, max_triplets_per_dataset)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet) + "\n")
    
    print(f"\nTriplets saved to {output_file}")
    return triplets


if __name__ == "__main__":
    # Test triplet generation
    from datasets import load_dataset
    
    print("Loading test dataset...")
    snli = load_dataset("snli", split="train[:1000]")  # Small subset for testing
    
    # Convert to expected format
    formatted_data = []
    for item in snli:
        if item["label"] != -1:
            formatted_data.append({
                "text1": item["premise"],
                "text2": item["hypothesis"],
                "label": item["label"],
                "source": "snli"
            })
    
    from datasets import Dataset
    test_dataset = Dataset.from_list(formatted_data)
    
    # Generate triplets
    generator = TripletGenerator()
    triplets = generator.from_snli(test_dataset)
    
    print(f"\nGenerated {len(triplets)} triplets")
    print("\nExample triplets:")
    for i, triplet in enumerate(triplets[:3]):
        print(f"\nTriplet {i+1}:")
        print(f"  Anchor:   {triplet['anchor']}")
        print(f"  Positive: {triplet['positive']}")
