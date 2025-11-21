"""
Model configuration for the text embedding model.
Optimized for Google Colab T4 GPU (16GB VRAM).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer-based embedding model with modern architecture."""
    
    # Vocabulary and embeddings
    vocab_size: int = 30000
    embedding_dim: int = 512
    max_seq_length: int = 512
    pad_token_id: int = 0
    
    # Transformer architecture
    num_layers: int = 6  # Reduced from 12 for T4 GPU
    num_attention_heads: int = 8  # Query heads
    num_kv_heads: int = 2  # KV heads for Grouped Query Attention (NEW!)
    feedforward_dim: int = 2048
    
    # Output embedding
    output_embedding_dim: int = 384  # Final embedding size
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Activation
    activation: str = "gelu"
    
    # RoPE settings
    rope_base: int = 10000
    rope_scaling_factor: float = 1.0  # YaRN scaling
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.embedding_dim % self.num_attention_heads == 0, \
            f"Embedding dim {self.embedding_dim} must be divisible by num heads {self.num_attention_heads}"
        assert self.num_attention_heads % self.num_kv_heads == 0, \
            f"Query heads {self.num_attention_heads} must be divisible by KV heads {self.num_kv_heads}"
        assert self.activation in ["gelu", "relu", "swish"], \
            f"Activation must be one of: gelu, relu, swish"


@dataclass
class TrainingConfig:
    """Training configuration optimized for T4 GPU."""
    
    # Batch sizes (with gradient accumulation for effective batch size)
    per_device_batch_size: int = 32  # Smaller for T4 memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 256
    
    # Learning rate
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1  # 10% of total steps
    weight_decay: float = 0.01
    
    # Training duration
    num_epochs: int = 10
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Mixed precision (essential for T4)
    use_fp16: bool = True
    
    # Loss function
    loss_type: str = "mnr"  # "mnr" (Multiple Negatives Ranking) or "infonce"
    temperature: float = 0.05
    
    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 2500
    logging_steps: int = 100
    save_total_limit: int = 5  # Keep only 5 best checkpoints
    
    # Paths
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    
    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Dataset paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    
    # Tokenizer
    tokenizer_path: Optional[str] = None  # Path to trained tokenizer
    vocab_size: int = 30000
    
    # Data processing
    num_workers: int = 4
    max_seq_length: int = 512
    
    # Dataset selection
    use_wikipedia: bool = True
    use_snli: bool = True
    use_quora: bool = True
    use_msmarco: bool = False  # Large dataset, set to False for faster training
    
    # Sampling
    max_samples_per_dataset: Optional[int] = None  # Limit for testing
    
    # Hard negative mining
    use_hard_negatives: bool = True
    hard_negative_ratio: float = 0.5  # 50% hard negatives, 50% random
    
    # Validation split
    validation_split: float = 0.05  # 5% for validation


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # STS Benchmark
    evaluate_sts: bool = True
    sts_batch_size: int = 64
    
    # BEIR datasets
    evaluate_beir: bool = True
    beir_datasets: list = field(default_factory=lambda: [
        "trec-covid",
        "nfcorpus",
        "fiqa"
    ])
    beir_batch_size: int = 128
    
    # Metrics
    compute_cosine_similarity: bool = True
    compute_ranking_metrics: bool = True
