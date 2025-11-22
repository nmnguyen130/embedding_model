"""
Main trainer class for training the embedding model.
Optimized for Google Colab T4 GPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Fixed deprecation warnings - use torch.amp instead
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
import json
from pathlib import Path


class EmbeddingTrainer:
    """Trainer for text embedding model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str = "./outputs",
        logging_steps: int = 100,
        save_steps: int = 5000,
        eval_steps: int = 2500,
        max_grad_norm: float = 1.0,
        use_fp16: bool = True,
        gradient_accumulation_steps: int = 1,
        save_total_limit: int = 5
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            eval_dataloader: Evaluation data loader (optional)
            output_dir: Directory to save checkpoints
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            max_grad_norm: Maximum gradient norm for clipping
            use_fp16: Whether to use mixed precision training
            gradient_accumulation_steps: Number of gradient accumulation steps
            save_total_limit: Maximum number of checkpoints to keep
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.max_grad_norm = max_grad_norm
        self.use_fp16 = use_fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_total_limit = save_total_limit
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if use_fp16 else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.train_loss_history = []
        self.eval_loss_history = []
        
        # Checkpoint tracking
        self.saved_checkpoints = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of data
        Returns:
            Loss value
        """
        # Move batch to device
        anchor_input_ids = batch["anchor_input_ids"].to(self.device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
        positive_input_ids = batch["positive_input_ids"].to(self.device)
        positive_attention_mask = batch["positive_attention_mask"].to(self.device)
        
        # Forward pass with mixed precision
        if self.use_fp16:
            with torch.amp.autocast('cuda', enabled=True):
                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)
                loss = self.loss_fn(anchor_embeddings, positive_embeddings)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
        else:
            # Regular forward pass
            anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = self.model(positive_input_ids, positive_attention_mask)
            loss = self.loss_fn(anchor_embeddings, positive_embeddings)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_fp16:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single evaluation step.
        
        Args:
            batch: Batch of data
        Returns:
            Loss value
        """
        self.model.eval()
        
        # Move batch to device
        anchor_input_ids = batch["anchor_input_ids"].to(self.device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
        positive_input_ids = batch["positive_input_ids"].to(self.device)
        positive_attention_mask = batch["positive_attention_mask"].to(self.device)
        
        # Forward pass
        if self.use_fp16:
            with autocast():
                anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = self.model(positive_input_ids, positive_attention_mask)
                loss = self.loss_fn(anchor_embeddings, positive_embeddings)
        else:
            anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = self.model(positive_input_ids, positive_attention_mask)
            loss = self.loss_fn(anchor_embeddings, positive_embeddings)
        
        self.model.train()
        return loss.item()
    
    def evaluate(self) -> float:
        """
        Run evaluation on eval dataset.
        
        Returns:
            Average evaluation loss
        """
        if self.eval_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        eval_pbar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)
        for batch in eval_pbar:
            loss = self.eval_step(batch)
            total_loss += loss
            num_batches += 1
            
            eval_pbar.set_postfix({"eval_loss": f"{loss:.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.model.train()
        
        return avg_loss
    
    def save_checkpoint(self, save_dir: Path, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint
            is_best: Whether this is the best model so far
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.model.get_config().__dict__
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = save_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save model config
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.model.get_config().__dict__, f, indent=2)
        
        print(f"Checkpoint saved to {save_dir}")
        
        # Track saved checkpoints
        self.saved_checkpoints.append(save_dir)
        
        # Remove old checkpoints if exceeding limit
        if len(self.saved_checkpoints) > self.save_total_limit:
            oldest_checkpoint = self.saved_checkpoints.pop(0)
            if oldest_checkpoint.exists() and not is_best:
                import shutil
                shutil.rmtree(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")
        
        # Save best model separately
        if is_best:
            best_model_dir = self.output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_model_dir / "checkpoint.pt")
            with open(best_model_dir / "config.json", "w") as f:
                json.dump(self.model.get_config().__dict__, f, indent=2)
            print(f"Best model saved to {best_model_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(
        self,
        num_epochs: int,
        max_steps: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            max_steps: Maximum number of steps (overrides num_epochs if set)
        """
        self.model.train()
        
        print(f"Starting training...")
        print(f"  Num epochs: {num_epochs}")
        print(f"  Batch size: {self.train_dataloader.batch_size}")
        print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps per epoch: {len(self.train_dataloader) // self.gradient_accumulation_steps}")
        print(f"  FP16 training: {self.use_fp16}")
        print(f"  Device: {self.device}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(pbar):
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Perform optimizer step after accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        
                        self.train_loss_history.append({
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": current_lr
                        })
                    
                    # Evaluation
                    is_best = False  # Initialize here to avoid UnboundLocalError
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        print(f"\nStep {self.global_step}: Eval loss = {eval_loss:.4f}")
                        
                        self.eval_loss_history.append({
                            "step": self.global_step,
                            "loss": eval_loss
                        })
                        
                        # Save best model
                        is_best = eval_loss < self.best_eval_loss
                        if is_best:
                            self.best_eval_loss = eval_loss
                            print(f"New best model! Eval loss: {eval_loss:.4f}")
                    
                    # Save checkpoint (ALWAYS runs at save_steps interval)
                    if self.global_step % self.save_steps == 0:
                        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
                        self.save_checkpoint(checkpoint_dir, is_best=is_best)
                        print(f"Checkpoint saved at step {self.global_step}")
                    
                    # Check max steps
                    if max_steps is not None and self.global_step >= max_steps:
                        print(f"\nReached max steps ({max_steps}). Stopping training.")
                        return
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch+1}"
            self.save_checkpoint(checkpoint_dir, is_best=False)
        
        print("\nTraining completed!")
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "train_loss": self.train_loss_history,
                "eval_loss": self.eval_loss_history
            }, f, indent=2)
        
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    print("Trainer module created successfully!")
    print("Use this module to train your embedding model.")
