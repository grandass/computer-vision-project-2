"""
Dataset Module for Medical Image Regression Task
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MedicalImageDataset(Dataset):
    """
    Medical Image Dataset Class
    
    Args:
        csv_file (str): Path to CSV file
        img_dir (str): Path to image directory
        split (str): 'train' or 'val'
        transform (callable, optional): Image transformations
    """
    
    def __init__(self, csv_file, img_dir, split='train', transform=None):
        """
        Initialize dataset
        """
        self.data = pd.read_csv(csv_file)
        # Select only data for the specified split
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        print(f"Loaded {len(self.data)} samples for {split} set")
        print(f"Score range: [{self.data['score_avg'].min():.2f}, {self.data['score_avg'].max():.2f}]")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get one sample
        
        Returns:
            image (torch.Tensor): Image tensor
            score (torch.Tensor): Score label
            info (dict): Additional information (exam_number)
        """
        # Get image path
        img_name = self.data.iloc[idx]['image_filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label (regression target)
        score = torch.tensor(self.data.iloc[idx]['score_avg'], dtype=torch.float32)
        
        # Get additional information
        info = {
            'exam_number': self.data.iloc[idx]['exam_number']
        }
        
        return image, score, info


def get_transforms(image_size=224, is_train=True):
    """
    Get image preprocessing transformations
    
    Args:
        image_size (int): Image size
        is_train (bool): Whether for training set
    
    Returns:
        transforms.Compose: Composition of image transformations
    """
    if is_train:
        # Training set: includes data augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation set: basic preprocessing only
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(csv_file, train_dir, val_dir, batch_size=32, 
                       image_size=224, num_workers=4):
    """
    Create DataLoaders for training and validation sets
    
    Args:
        csv_file (str): Path to CSV file
        train_dir (str): Path to training images directory
        val_dir (str): Path to validation images directory
        batch_size (int): Batch size
        image_size (int): Image size
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader: DataLoaders for training and validation sets
    """
    # Create datasets
    train_dataset = MedicalImageDataset(
        csv_file=csv_file,
        img_dir=train_dir,
        split='train',
        transform=get_transforms(image_size, is_train=True)
    )
    
    val_dataset = MedicalImageDataset(
        csv_file=csv_file,
        img_dir=val_dir,
        split='val',
        transform=get_transforms(image_size, is_train=False)
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    """Test dataset loading"""
    import matplotlib.pyplot as plt
    
    # Test dataset
    csv_file = 'data.csv'
    train_dir = 'sub_train'
    val_dir = 'sub_val'
    
    # Create dataset
    train_dataset = MedicalImageDataset(
        csv_file=csv_file,
        img_dir=train_dir,
        split='train',
        transform=get_transforms(224, is_train=True)
    )
    
    # Test getting one sample
    image, score, info = train_dataset[0]
    print(f"\nSample info:")
    print(f"Image shape: {image.shape}")
    print(f"Score: {score.item()}")
    print(f"Patient info: {info}")
    
    # Visualize first image
    print(f"\nDataset is working correctly!")

