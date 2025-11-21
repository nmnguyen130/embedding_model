"""
Transformer-based text embedding model built from scratch.
No pretrained weights - trained from random initialization.

Modern architecture improvements:
- RMSNorm instead of LayerNorm (more efficient)
- Rotary Position Embedding (RoPE) with YaRN scaling (better than sinusoidal)
- Grouped Query Attention (GQA) (more efficient than multi-head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., dim]
        Returns:
            Normalized tensor [..., dim]
        """
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with YaRN scaling.
    Better than sinusoidal for long sequences.
    
    Reference: https://arxiv.org/abs/2104.09864 (RoPE)
               https://arxiv.org/abs/2309.00071 (YaRN)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float)
        # YaRN scaling: scale positions for better extrapolation
        t = t / scaling_factor
        freqs = torch.outer(t, inv_freq)
        
        # Create rotation matrices
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[1]
        
        # Get cached sin/cos
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Positional encoding [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.positional_embeddings(positions)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - more efficient than Multi-Head Attention.
    
    GQA reduces the number of key/value heads while keeping multiple query heads.
    This significantly reduces memory and computation while maintaining quality.
    
    Example: 8 query heads, 2 KV heads -> each KV head shared by 4 query heads
    
    Reference: https://arxiv.org/abs/2305.13245
    """
    
    def __init__(
        self,
        d_model: int,
        num_query_heads: int = 8,
        num_kv_heads: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_query_heads // num_kv_heads  # How many Q heads per KV head
        self.head_dim = d_model // num_query_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (full number of heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Key/Value projections (reduced number of heads)
        kv_dim = self.head_dim * num_kv_heads
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE for positional encoding
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=2048
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Mask [batch_size, seq_len] (1 for valid, 0 for padding)
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)  # [batch, seq_len, kv_dim]
        v = self.v_proj(x)  # [batch, seq_len, kv_dim]
        
        # Reshape for multi-head attention
        # Q: [batch, seq_len, num_query_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        # K, V: [batch, seq_len, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Transpose for attention computation
        # [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat KV heads to match Q heads (grouped attention)
        # [batch, num_query_heads, seq_len, head_dim]
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Scaled dot-product attention
        # [batch, num_query_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        # [batch, num_query_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_probs, v)
        
        # Concatenate heads
        # [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with modern improvements."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Grouped Query Attention (more efficient than MHA)
        num_kv_heads = getattr(config, 'num_kv_heads', max(1, config.num_attention_heads // 4))
        self.attention = GroupedQueryAttention(
            d_model=config.embedding_dim,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            dropout=config.attention_dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=config.embedding_dim,
            d_ff=config.feedforward_dim,
            dropout=config.dropout,
            activation=config.activation
        )
        
        # RMSNorm (more efficient than LayerNorm)
        self.norm1 = RMSNorm(config.embedding_dim)
        self.norm2 = RMSNorm(config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm architecture (norm before attention)
        normed = self.norm1(x)
        attn_output = self.attention(normed, attention_mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class TextEmbeddingModel(nn.Module):
    """Complete text embedding model with modern architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id
        )
        
        # Transformer encoder (RoPE is integrated in GQA)
        self.encoder = TransformerEncoder(config)
        
        # Projection head (reduce dimensionality)
        self.projection = nn.Linear(config.embedding_dim, config.output_embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, RMSNorm):
                # Initialize RMSNorm weight to 1
                nn.init.constant_(module.weight, 1.0)
    
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over sequence, considering attention mask.
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            Pooled embeddings [batch_size, d_model]
        """
        # Expand mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Count valid tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        return sum_embeddings / sum_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        Returns:
            L2-normalized embeddings [batch_size, output_embedding_dim]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        
        # Token embeddings
        embeddings = self.token_embeddings(input_ids)  # [batch_size, seq_len, d_model]
        embeddings = self.dropout(embeddings)
        
        # Transformer encoder (RoPE applied inside GQA)
        encoder_output = self.encoder(embeddings, attention_mask)
        
        # Mean pooling
        pooled_output = self.mean_pooling(encoder_output, attention_mask)
        
        # Projection
        projected = self.projection(pooled_output)
        
        # L2 normalization for cosine similarity
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized
    
    def get_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config


def create_model(config: ModelConfig) -> TextEmbeddingModel:
    """
    Create a text embedding model with the given configuration.
    
    Args:
        config: Model configuration
    Returns:
        TextEmbeddingModel instance
    """
    model = TextEmbeddingModel(config)
    return model


if __name__ == "__main__":
    # Test the model
    config = ModelConfig(
        vocab_size=30000,
        embedding_dim=512,
        num_layers=6,
        num_attention_heads=8,
        output_embedding_dim=384
    )
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward
    embeddings = model(input_ids, attention_mask)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output is normalized: {torch.allclose(torch.norm(embeddings, dim=1), torch.ones(batch_size))}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
