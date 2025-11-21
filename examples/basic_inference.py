"""
Example: Basic usage of the embedding model for inference.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_model


def main():
    """Demonstrate basic usage of the embedding model."""
    
    print("="*70)
    print("Text Embedding Model - Basic Inference Example")
    print("="*70)
    
    # Note: Update these paths to your actual checkpoint and tokenizer
    checkpoint_path = "./outputs/best_model/checkpoint.pt"
    tokenizer_path = "./data/tokenizer/tokenizer.json"
    
    # Check if files exist
    if not Path(checkpoint_path).exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python src/training/train.py --quick-start")
        return
    
    if not Path(tokenizer_path).exists():
        print(f"\nError: Tokenizer not found at {tokenizer_path}")
        print("Please train the model first to generate the tokenizer.")
        return
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, tokenizer_path)
    
    # Example 1: Encode single text
    print("\n" + "="*70)
    print("Example 1: Encode single text")
    print("="*70)
    
    text = "Machine learning is a fascinating field of artificial intelligence."
    embedding = model.encode(text)
    
    print(f"\nText: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 dimensions: {embedding[:10]}")
    
    # Example 2: Encode multiple texts
    print("\n" + "="*70)
    print("Example 2: Encode batch of texts")
    print("="*70)
    
    texts = [
        "Natural language processing enables computers to understand human language.",
        "Deep learning has revolutionized computer vision.",
        "The weather is beautiful today."
    ]
    
    embeddings = model.encode(texts)
    
    print(f"\nEncoded {len(texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}")
    
    for i, text in enumerate(texts):
        print(f"\n{i+1}. {text}")
        print(f"   Embedding preview: {embeddings[i][:5]}...")
    
    # Example 3: Compute similarity
    print("\n" + "="*70)
    print("Example 3: Compute text similarity")
    print("="*70)
    
    text1 = "I love machine learning and AI"
    text2 = "Machine learning is my favorite topic"
    text3 = "The cat is sleeping on the couch"
    
    sim_12 = model.similarity(text1, text2)
    sim_13 = model.similarity(text1, text3)
    sim_23 = model.similarity(text2, text3)
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    print(f"\nSimilarity scores (range: -1 to 1):")
    print(f"  Text 1 ↔ Text 2: {sim_12:.4f} (should be high - similar topics)")
    print(f"  Text 1 ↔ Text 3: {sim_13:.4f} (should be low - different topics)")
    print(f"  Text 2 ↔ Text 3: {sim_23:.4f} (should be low - different topics)")
    
    # Example 4: Semantic search
    print("\n" + "="*70)
    print("Example 4: Semantic search")
    print("="*70)
    
    query = "artificial intelligence and neural networks"
    
    candidates = [
        "Machine learning algorithms for data analysis",
        "Best pizza recipes in Italy",
        "Deep learning and convolutional neural networks",
        "How to train your dog",
        "Natural language processing with transformers",
        "Gardening tips for beginners",
        "Computer vision applications in robotics",
        "Travel destinations in Europe"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nSearching through {len(candidates)} candidates...")
    
    results = model.find_similar(query, candidates, top_k=5)
    
    print(f"\nTop 5 results:")
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. {text}")
        print(f"     Score: {score:.4f}")
    
    # Example 5: Batch similarity
    print("\n" + "="*70)
    print("Example 5: Batch similarity computation")
    print("="*70)
    
    queries = [
        "Python programming",
        "Healthy eating",
        "Space exploration"
    ]
    
    documents = [
        "Learn to code in Python",
        "Nutrition and diet tips",
        "NASA Mars mission"
    ]
    
    print("\nComputing similarity matrix...\n")
    print("Queries vs Documents:")
    print("-" * 70)
    
    for i, query in enumerate(queries):
        scores = []
        for j, doc in enumerate(documents):
            score = model.similarity(query, doc)
            scores.append(score)
        
        print(f"\nQuery {i+1}: {query}")
        for j, (doc, score) in enumerate(zip(documents, scores)):
            print(f"  → Doc {j+1}: {score:.4f}  ({doc})")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
