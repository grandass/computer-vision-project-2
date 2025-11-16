"""
Main Training Script for Medical Image Regression Task

Usage:
python main.py --model simple_cnn --epochs 50 --batch_size 32 --lr 0.001

Student Tasks:
1. Run baseline model (SimpleCNN)
2. Understand the training process
3. Implement your own model in model.py
4. Tune hyperparameters for better performance
5. Analyze results and write report
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime

from dataset import create_dataloaders
from model import get_model
from utils import (
    train_one_epoch, validate, save_checkpoint, load_checkpoint,
    plot_training_history, plot_predictions, EarlyStopping,
    get_lr, count_parameters
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Medical Image Regression Training')
    
    # Data parameters
    parser.add_argument('--csv_file', type=str, default='data.csv',
                        help='Path to CSV file')
    parser.add_argument('--train_dir', type=str, default='sub_train',
                        help='Path to training images directory')
    parser.add_argument('--val_dir', type=str, default='sub_val',
                        help='Path to validation images directory')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'student'],
                        help='Model type')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Early stopping and checkpoints
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Early stopping patience (0 means no early stopping)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main function"""
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.model}_{timestamp}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\n" + "="*60)
    print("Creating data loaders...")
    print("="*60)
    train_loader, val_loader = create_dataloaders(
        csv_file=args.csv_file,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    model = get_model(args.model)
    model = model.to(device)
    
    # Print model information
    params = count_parameters(model)
    print(f"Model: {args.model}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Define loss function
    criterion = nn.MSELoss()  # MSE loss for regression task
    
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                             momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=5)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, mode='min')
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1

    # Training history
    train_losses = []
    val_losses = []
    val_metrics_history = []
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Val R²: {val_metrics['r2']:.4f}")
        print(f"Learning Rate: {get_lr(optimizer):.6f}")
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, 
                          best_checkpoint_path)
            print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
        
        # Save latest model
        latest_checkpoint_path = os.path.join(exp_dir, 'latest_model.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, 
                       latest_checkpoint_path)
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        if epoch == 5:
            model.unfreeze_backbone()
    
    # Training complete
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Plot training history
    plot_path = os.path.join(exp_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, val_metrics_history, plot_path)
    
    # Load best model for final evaluation
    print("\nPerforming final evaluation with best model...")
    load_checkpoint(model, None, best_checkpoint_path, device)
    
    # Final validation
    final_loss, final_metrics = validate(model, val_loader, criterion, device)
    
    print("\nFinal Validation Results:")
    print(f"Loss: {final_loss:.4f}")
    print(f"MAE: {final_metrics['mae']:.4f}")
    print(f"RMSE: {final_metrics['rmse']:.4f}")
    print(f"R²: {final_metrics['r2']:.4f}")
    
    # Generate prediction plot
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    pred_plot_path = os.path.join(exp_dir, 'predictions.png')
    plot_predictions(all_preds, all_targets, pred_plot_path)
    
    # Save configuration and results
    results_path = os.path.join(exp_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("Training Configuration:\n")
        f.write("="*60 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\nModel Information:\n")
        f.write("="*60 + "\n")
        f.write(f"Total parameters: {params['total']:,}\n")
        f.write(f"Trainable parameters: {params['trainable']:,}\n")
        
        f.write("\nFinal Results:\n")
        f.write("="*60 + "\n")
        f.write(f"Val Loss: {final_loss:.4f}\n")
        f.write(f"Val MAE: {final_metrics['mae']:.4f}\n")
        f.write(f"Val RMSE: {final_metrics['rmse']:.4f}\n")
        f.write(f"Val R²: {final_metrics['r2']:.4f}\n")
    
    print(f"\nResults saved to: {exp_dir}")
    print(f"- Best model: {best_checkpoint_path}")
    print(f"- Training history plot: {plot_path}")
    print(f"- Predictions plot: {pred_plot_path}")
    print(f"- Results file: {results_path}")


if __name__ == '__main__':
    main()
