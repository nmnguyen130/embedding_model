"""
Loss functions for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (InfoNCE with in-batch negatives).
    
    Given a batch of (anchor, positive) pairs, treats all other positives
    in the batch as negatives. This is very efficient and works well for
    embedding models.
    
    Reference: https://arxiv.org/abs/1705.00652
    """
    
    def __init__(self, temperature: float = 0.05):
        """
        Initialize loss.
        
        Args:
            temperature: Temperature for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive embeddings [batch_size, embedding_dim]
        Returns:
            Loss value
        """
        batch_size = anchor_embeddings.size(0)
        
        # Normalize embeddings (already done in model, but ensure it)
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Compute similarity matrix: [batch_size, batch_size]
        # similarity[i, j] = cosine_sim(anchor[i], positive[j])
        similarities = torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature
        
        # Labels: anchor[i] should be most similar to positive[i]
        # So the correct class for row i is index i
        labels = torch.arange(batch_size, device=similarities.device)
        
        # Cross-entropy loss
        # For each anchor, we want its positive to have the highest similarity
        loss = self.cross_entropy(similarities, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (same as Multiple Negatives Ranking Loss).
    Kept as separate class for clarity.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.mnr_loss = MultipleNegativesRankingLoss(temperature=temperature)
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.mnr_loss(anchor_embeddings, positive_embeddings)


class TripletLoss(nn.Module):
    """
    Traditional triplet loss with margin.
    Less efficient than MNR loss but useful as baseline.
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative distances
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive embeddings [batch_size, embedding_dim]
            negative_embeddings: Negative embeddings [batch_size, embedding_dim]
        Returns:
            Loss value
        """
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute distances (using 1 - cosine_similarity as distance)
        pos_distance = 1 - F.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_distance = 1 - F.cosine_similarity(anchor_embeddings, negative_embeddings)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairs.
    Pulls similar pairs closer, pushes dissimilar pairs apart.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embeddings [batch_size, embedding_dim]
            embedding2: Second embeddings [batch_size, embedding_dim]
            label: 1 for similar, 0 for dissimilar [batch_size]
        Returns:
            Loss value
        """
        # Normalize embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Loss
        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance (up to margin)
        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = loss_similar + loss_dissimilar
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """
    Simple cosine similarity loss.
    Minimizes 1 - cosine_similarity for similar pairs.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        target_similarity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            embedding1: First embeddings [batch_size, embedding_dim]
            embedding2: Second embeddings [batch_size, embedding_dim]
            target_similarity: Target similarity scores (optional)
        Returns:
            Loss value
        """
        # Normalize embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2)
        
        if target_similarity is not None:
            # MSE loss between predicted and target similarity
            loss = F.mse_loss(similarity, target_similarity)
        else:
            # Maximize similarity (minimize 1 - similarity)
            loss = (1 - similarity).mean()
        
        return loss


def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Create loss function by name.
    
    Args:
        loss_type: Type of loss ("mnr", "infonce", "triplet", "contrastive", "cosine")
        **kwargs: Additional arguments for loss function
    Returns:
        Loss function module
    """
    if loss_type == "mnr":
        return MultipleNegativesRankingLoss(**kwargs)
    elif loss_type == "infonce":
        return InfoNCELoss(**kwargs)
    elif loss_type == "triplet":
        return TripletLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastiveLoss(**kwargs)
    elif loss_type == "cosine":
        return CosineSimilarityLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    embedding_dim = 384
    
    # Create dummy embeddings
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    
    print("Testing loss functions:")
    print("="*50)
    
    # MNR Loss
    mnr_loss = MultipleNegativesRankingLoss(temperature=0.05)
    loss_value = mnr_loss(anchor, positive)
    print(f"MNR Loss: {loss_value.item():.4f}")
    
    # InfoNCE Loss
    infonce_loss = InfoNCELoss(temperature=0.05)
    loss_value = infonce_loss(anchor, positive)
    print(f"InfoNCE Loss: {loss_value.item():.4f}")
    
    # Triplet Loss
    triplet_loss = TripletLoss(margin=0.5)
    loss_value = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss_value.item():.4f}")
    
    # Contrastive Loss
    contrastive_loss = ContrastiveLoss(margin=1.0)
    labels = torch.ones(batch_size)  # All pairs are similar
    loss_value = contrastive_loss(anchor, positive, labels)
    print(f"Contrastive Loss: {loss_value.item():.4f}")
    
    # Cosine Similarity Loss
    cosine_loss = CosineSimilarityLoss()
    loss_value = cosine_loss(anchor, positive)
    print(f"Cosine Similarity Loss: {loss_value.item():.4f}")
    
    print("\nAll loss functions work correctly!")
