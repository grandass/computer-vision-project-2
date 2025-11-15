"""
Inference Script for Medical Image Regression Task

Usage:
python inference.py --checkpoint checkpoints/best_model.pth --image_path test_image.jpg
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms

from model import get_model
from utils import load_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Medical Image Regression Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image (optional)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to image directory (optional)')
    parser.add_argument('--model', type=str, default='simple_cnn',
                        choices=['simple_cnn', 'student'],
                        help='Model type')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output CSV file')
    
    return parser.parse_args()


def get_inference_transform(image_size=224):
    """
    Get image transformations for inference
    
    Args:
        image_size (int): Image size
    
    Returns:
        transform: Image transformation
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def predict_single_image(model, image_path, transform, device):
    """
    Predict on a single image
    
    Args:
        model: PyTorch model
        image_path (str): Path to image
        transform: Image transformation
        device: Device
    
    Returns:
        prediction (float): Predicted value
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    return prediction.item()


def predict_batch(model, image_paths, transform, device, batch_size=32):
    """
    Batch prediction on multiple images
    
    Args:
        model: PyTorch model
        image_paths (list): List of image paths
        transform: Image transformation
        device: Device
        batch_size (int): Batch size
    
    Returns:
        predictions (list): List of predicted values
    """
    model.eval()
    predictions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load and preprocess images
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                predictions.append(None)
                continue
        
        if not batch_images:
            continue
        
        # Batch prediction
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_predictions = model(batch_tensor)
        
        predictions.extend(batch_predictions.cpu().numpy().tolist())
        
        # Print progress
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return predictions


def main():
    """Main function"""
    
    # Parse arguments
    args = parse_args()
    
    # Check inputs
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Must provide either --image_path or --image_dir")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(args.model)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device)
    
    # Get image transformation
    transform = get_inference_transform(args.image_size)
    
    # Single image prediction
    if args.image_path is not None:
        print(f"\nPredicting on image: {args.image_path}")
        prediction = predict_single_image(model, args.image_path, transform, device)
        print(f"Predicted score: {prediction:.2f}")
        
        # Save results
        results_df = pd.DataFrame({
            'image_path': [args.image_path],
            'predicted_score': [prediction]
        })
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
    
    # Batch prediction
    elif args.image_dir is not None:
        print(f"\nBatch prediction on images in directory: {args.image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        image_names = []
        
        for filename in os.listdir(args.image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(args.image_dir, filename))
                image_names.append(filename)
        
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No image files found!")
            return
        
        # Batch prediction
        predictions = predict_batch(model, image_paths, transform, device)
        
        # Save results
        results_df = pd.DataFrame({
            'image_name': image_names,
            'image_path': image_paths,
            'predicted_score': predictions
        })
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
        
        # Print statistics
        valid_predictions = [p for p in predictions if p is not None]
        if valid_predictions:
            print(f"\nPrediction Statistics:")
            print(f"Min: {min(valid_predictions):.2f}")
            print(f"Max: {max(valid_predictions):.2f}")
            print(f"Mean: {np.mean(valid_predictions):.2f}")
            print(f"Std: {np.std(valid_predictions):.2f}")


if __name__ == '__main__':
    main()
