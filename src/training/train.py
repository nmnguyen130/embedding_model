"""
Main training script.
Optimized for Google Colab T4 GPU.

Usage:
    python train.py --config config/train_config.yaml
    or
    python train.py --quick-start  # For quick testing with small dataset
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelConfig, TrainingConfig, create_model
from src.tokenizer import TextTokenizer, train_tokenizer_on_datasets
from src.data import DatasetDownloader, TripletGenerator, InBatchNegativesDataset, create_dataloader
from src.training.losses import create_loss_function
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import EmbeddingTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train text embedding model")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    
    # Quick start mode
    parser.add_argument("--quick-start", action="store_true",
                        help="Quick start with small dataset for testing")
    
    # Model config
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=384)
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true", default=True)
    
    # Optimizer config
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "hybrid"],
                        help="Optimizer: 'adamw' (pure AdamW) or 'hybrid' (Muon+AdamW)")
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Learning rate for Muon in hybrid mode")
    parser.add_argument("--num-kv-heads", type=int, default=2,
                        help="Number of KV heads for GQA")
    
    # Data config
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--triplets-file", type=str, default=None)
    
    # Dataset selection
    parser.add_argument("--use-wikipedia", action="store_true", default=True)
    parser.add_argument("--use-snli", action="store_true", default=True)
    parser.add_argument("--use-quora", action="store_true", default=False)
    parser.add_argument("--max-wiki-samples", type=int, default=100000)
    
    # Resume training
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Quick start mode
    if args.quick_start:
        print("="*60)
        print("QUICK START MODE - Using small dataset for testing")
        print("="*60)
        args.max_wiki_samples = 10000
        args.num_epochs = 2
        args.max_steps = 1000
        args.batch_size = 16
        args.grad_accum_steps = 4
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train or load tokenizer
    print("\n" + "="*60)
    print("STEP 1: Tokenizer")
    print("="*60)
    
    tokenizer_path = args.tokenizer_path or f"{args.data_dir}/tokenizer/tokenizer.json"
    
    if not Path(tokenizer_path).exists():
        print(f"Training tokenizer (vocab size: {args.vocab_size})...")
        
        # Download small dataset for tokenizer training
        datasets_for_tokenizer = ["wikipedia", "snli"]
        tokenizer = train_tokenizer_on_datasets(
            datasets=datasets_for_tokenizer,
            data_dir=args.data_dir,
            output_path=tokenizer_path,
            vocab_size=args.vocab_size,
            max_samples=args.max_wiki_samples if args.quick_start else None
        )
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
    
    tokenizer = TextTokenizer(tokenizer_path, max_length=512)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Step 2: Download and prepare datasets
    print("\n" + "="*60)
    print("STEP 2: Download Datasets")
    print("="*60)
    
    downloader = DatasetDownloader(data_dir=args.data_dir)
    datasets = downloader.download_all(
        include_wikipedia=args.use_wikipedia,
        include_snli=args.use_snli,
        include_quora=args.use_quora,
        include_msmarco=False,  # Skip MS MARCO for T4 (too large)
        max_wikipedia_samples=args.max_wiki_samples,
        max_msmarco_samples=0
    )
    
    # Step 3: Generate triplets
    print("\n" + "="*60)
    print("STEP 3: Generate Training Triplets")
    print("="*60)
    
    triplets_file = args.triplets_file or f"{args.data_dir}/triplets.jsonl"
    
    if not Path(triplets_file).exists():
        print("Generating triplets...")
        generator = TripletGenerator()
        triplets = generator.generate_triplets(
            datasets,
            max_triplets_per_dataset=args.max_wiki_samples if args.quick_start else None
        )
        
        # Save triplets
        import json
        with open(triplets_file, "w", encoding="utf-8") as f:
            for triplet in triplets:
                f.write(json.dumps(triplet) + "\n")
        
        print(f"Triplets saved to {triplets_file}")
    else:
        print(f"Using existing triplets from {triplets_file}")
    
    # Step 4: Create datasets and dataloaders
    print("\n" + "="*60)
    print("STEP 4: Create DataLoaders")
    print("="*60)
    
    train_dataset = InBatchNegativesDataset(
        triplets_file=triplets_file,
        tokenizer=tokenizer,
        max_length=512
    )
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} pairs")
    print(f"Train batches: {len(train_dataloader)}")
    
    # Step 5: Create model
    print("\n" + "="*60)
    print("STEP 5: Create Model")
    print("="*60)
    
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        output_embedding_dim=args.output_dim,
        max_seq_length=512,
        pad_token_id=tokenizer.pad_token_id
    )
    
    model = create_model(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")
    
    # Step 6: Create optimizer and scheduler
    print("\n" + "="*60)
    print("STEP 6: Create Optimizer & Scheduler")
    print("="*60)
    
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        muon_lr=args.muon_lr
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader) // args.grad_accum_steps
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    opt_name = "HYBRID (Muon+AdamW)" if args.optimizer == "hybrid" else "AdamW"
    opt_lr = args.muon_lr if args.optimizer == "hybrid" else args.learning_rate
    print(f"Optimizer: {opt_name} (lr={opt_lr})")
    print(f"Scheduler: Cosine with warmup")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    
    # Step 7: Create loss function
    print("\n" + "="*60)
    print("STEP 7: Create Loss Function")
    print("="*60)
    
    loss_fn = create_loss_function("mnr", temperature=0.05)
    print("Loss function: Multiple Negatives Ranking Loss (temperature=0.05)")
    
    # Step 8: Create trainer
    print("\n" + "="*60)
    print("STEP 8: Initialize Trainer")
    print("="*60)
    
    trainer = EmbeddingTrainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        eval_dataloader=None,  # Can add validation set later
        output_dir=args.output_dir,
        logging_steps=100,
        save_steps=5000,
        eval_steps=2500,
        max_grad_norm=1.0,
        use_fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_total_limit=3
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    print("Trainer initialized!")
    
    # Step 9: Train!
    print("\n" + "="*60)
    print("STEP 9: Training")
    print("="*60)
    
    trainer.train(
        num_epochs=args.num_epochs,
        max_steps=args.max_steps
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Best model: {Path(args.output_dir) / 'best_model'}")


if __name__ == "__main__":
    main()
