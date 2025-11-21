"""
Optimizer and learning rate scheduler setup.
Includes AdamW and Muon optimizers.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, List


class Muon(Optimizer):
    """
    Muon optimizer - momentum orthogonalized by Newton-schulz.
    More efficient than Adam for large models.
    
    Reference: https://arxiv.org/abs/2402.03496
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[List] = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_wd: float = 0.0
    ):
        """
        Args:
            params: Model parameters for Muon
            lr: Learning rate for Muon
            momentum: Momentum coefficient
            nesterov: Whether to use Nesterov momentum
            ns_steps: Newton-Schulz iteration steps
            adamw_params: Separate parameters for AdamW (for embeddings, norms, etc.)
            adamw_lr: Learning rate for AdamW parameters
            adamw_betas: Betas for AdamW
            adamw_wd: Weight decay for AdamW
        """
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        
        # Optionally create AdamW for specific params
        self.adamw = None
        if adamw_params is not None:
            self.adamw = torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=adamw_wd
            )
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                # Newton-Schulz orthogonalization
                if grad.dim() >= 2:
                    buf = self._newton_schulz(buf, steps=group['ns_steps'])
                
                if group['nesterov']:
                    grad = grad.add(buf, alpha=group['momentum'])
                else:
                    grad = buf
                
                p.data.add_(grad, alpha=-group['lr'])
        
        # Step AdamW if exists
        if self.adamw is not None:
            self.adamw.step()
        
        return loss
    
    def _newton_schulz(self, G, steps=5, eps=1e-7):
        """Newton-Schulz iteration for orthogonalization."""
        a, b, c = (3.4445, -4.7750, 2.0315)  # Coefficients
        X = G.bfloat16() / (G.norm() + eps)
        
        if G.size(0) > G.size(1):
            X = X.T
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if G.size(0) > G.size(1):
            X = X.T
        
        return X.to(G.dtype)
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_lr: float = 0.02
) -> Optimizer:
    """
    Create optimizer (AdamW or Hybrid Muon+AdamW).
    
    Args:
        model: Model to optimize
        optimizer_type: "adamw" or "hybrid" (Muon+AdamW)
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay coefficient
        adam_beta1: Adam beta1 parameter
        adam_beta2: Adam beta2 parameter
        adam_epsilon: Adam epsilon parameter
        muon_momentum: Momentum for Muon
        muon_lr: Learning rate for Muon in hybrid mode
    Returns:
        Optimizer instance
    """
    # Separate parameters for weight decay
    # Don't apply weight decay to bias and RMSNorm parameters
    no_decay = ["bias", "RMSNorm.weight", "rms_norm.weight"]
    
    if optimizer_type == "adamw":
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon
        )
    
    elif optimizer_type == "hybrid":
        # Muon for weight matrices, AdamW for norms/embeddings
        muon_params = []
        adamw_params = []
        
        for name, p in model.named_parameters():
            # Use Muon for large weight matrices
            if p.ndim >= 2 and not any(nd in name for nd in no_decay):
                muon_params.append(p)
            else:
                adamw_params.append(p)
        
        optimizer = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            adamw_params=adamw_params,
            adamw_lr=learning_rate,
            adamw_wd=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 default for one half-cycle)
        last_epoch: The index of last epoch
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and then constant.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    **kwargs
) -> Optional[LambdaLR]:
    """
    Create learning rate scheduler by type.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ("linear", "cosine", "constant", "none")
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: The index of last epoch
        **kwargs: Additional arguments for scheduler
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch=last_epoch, **kwargs
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps, last_epoch
        )
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
