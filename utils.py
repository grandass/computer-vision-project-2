"""
Utility Functions for Training and Evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime


class EarlyStopping:
    """
    Early Stopping Mechanism
    Stop training when validation loss stops improving
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience (int): Number of epochs to tolerate without improvement
            min_delta (float): Minimum improvement amount
            mode (str): 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if early stopping should be triggered
        
        Args:
            score (float): Current metric value
        
        Returns:
            bool: Whether to trigger early stopping
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """Calculate and store average and current values"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda or cpu)
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average loss
    """
    model.train()
    losses = AverageMeter()
    
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.update(loss.item(), images.size(0))
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {losses.val:.4f} (Avg: {losses.avg:.4f})')
    
    return losses.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Validate model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
    
    Returns:
        avg_loss: Average loss
        metrics: Evaluation metrics dictionary
    """
    model.eval()
    losses = AverageMeter()
    
    all_predictions = []
    all_targets = []
    
    for images, targets, _ in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Record
        losses.update(loss.item(), images.size(0))
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    # Calculate evaluation metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    return losses.avg, metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        loss: Current loss
        metrics: Evaluation metrics
        filepath: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Checkpoint path
        device: Device
    
    Returns:
        epoch: Epoch number
        loss: Loss
        metrics: Evaluation metrics
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    metrics = checkpoint.get('metrics', {})
    
    print(f'Checkpoint loaded from {filepath}')
    return epoch, loss, metrics


def plot_training_history(train_losses, val_losses, val_metrics, save_path='training_history.png'):
    """
    Plot training history
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_metrics: List of validation metrics dictionaries
        save_path: Save path
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE curve
    mae_values = [m['mae'] for m in val_metrics]
    axes[0, 1].plot(epochs, mae_values, 'g-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].grid(True)
    
    # RMSE curve
    rmse_values = [m['rmse'] for m in val_metrics]
    axes[1, 0].plot(epochs, rmse_values, 'm-')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation RMSE')
    axes[1, 0].grid(True)
    
    # R2 curve
    r2_values = [m['r2'] for m in val_metrics]
    axes[1, 1].plot(epochs, r2_values, 'c-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Validation R² Score')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history plot saved to {save_path}')
    plt.close()


def plot_predictions(predictions, targets, save_path='predictions.png'):
    """
    Plot scatter plot of predictions vs true values
    
    Args:
        predictions: Predictions array
        targets: True values array
        save_path: Save path
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=30)
    
    # Perfect prediction line (y=x)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.title('Predictions vs True Values', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Predictions plot saved to {save_path}')
    plt.close()


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    """Test utility functions"""
    
    # Test EarlyStopping
    print("Testing EarlyStopping...")
    early_stopping = EarlyStopping(patience=3, mode='min')
    losses = [0.5, 0.4, 0.3, 0.31, 0.32, 0.33]
    for i, loss in enumerate(losses):
        should_stop = early_stopping(loss)
        print(f"Epoch {i+1}: Loss={loss}, Should stop={should_stop}")
    
    # Test AverageMeter
    print("\nTesting AverageMeter...")
    meter = AverageMeter()
    for val in [1.0, 2.0, 3.0]:
        meter.update(val)
        print(f"Current={val}, Average={meter.avg}")
    
    print("\nUtils are working correctly!")

