"""
Model Module for Medical Image Regression Task

Student Tasks:
1. Understand the SimpleCNN architecture
2. Design and implement your own model architecture
3. Try different network depths, widths, and architectures
4. You can try using pretrained models (ResNet, EfficientNet, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN Baseline Model
    This is a basic convolutional neural network as a starting point for students
    
    Network Architecture:
    - 4 convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)
    - 2 fully connected layers
    - Final output: single regression value
    """
    
    def __init__(self, num_channels=3):
        super(SimpleCNN, self).__init__()
        
        # Convolutional block 1: 3 -> 32
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolutional block 4: 128 -> 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Input: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14
        # 14 * 14 * 256 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Regression task: output 1 value
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
        
        Returns:
            Output regression value [batch_size]
        """
        # Convolutional block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 112, 112]
        
        # Convolutional block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 56, 56]
        
        # Convolutional block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 128, 28, 28]
        
        # Convolutional block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # [B, 256, 14, 14]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 50176]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # [B, 512]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 1]
        
        return x.squeeze(1)  # [B]


# ==================== Student Implementation Area ====================
# TODO: Implement your own model below
# Hints:
# 1. You can increase network depth (more convolutional layers)
# 2. You can use residual connections (ResNet style)
# 3. You can use attention mechanisms
# 4. You can use pretrained models (torchvision.models)
# 5. You can try different activation functions, regularization methods, etc.

class StudentModel(nn.Module):
    """
    Student Custom Model
    
    TODO: Implement your model architecture here
    """
    
    def __init__(self, num_channels=3):
        super(StudentModel, self).__init__()
        
        # TODO: Define your network layers here
        # Example: You can use pretrained ResNet
        # import torchvision.models as models
        # self.backbone = models.resnet18(pretrained=True)
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

        pass
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
        
        Returns:
            Output regression value [batch_size]
        """
        # TODO: Implement forward pass
        pass


def get_model(model_name='simple_cnn', **kwargs):
    """
    Model factory function
    
    Args:
        model_name (str): Model name
        **kwargs: Model parameters
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'simple_cnn':
        return SimpleCNN(**kwargs)
    elif model_name == 'student':
        return StudentModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    """Test models"""

    # Test SimpleCNN
    print("Testing SimpleCNN...")
    model = SimpleCNN()
    x = torch.randn(4, 3, 224, 224)  # batch_size=4
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nModels are working correctly!")
