import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------
# Attention modules (CBAM-style)
# -----------------------

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # along channel axis
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, max_], dim=1)
        out = self.conv(cat)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.sa(x)
        return x

# -----------------------
# StudentModel
# -----------------------
class StudentModel(nn.Module):
    """
    Student custom model using EfficientNet-B0 backbone + CBAM attention + regression head.
    - By default most backbone feature layers are frozen (only last block trainable).
    - Call `unfreeze_backbone()` to unfreeze entire backbone for fine-tuning.
    """

    def __init__(self, num_channels=3, pretrained=True, dropout=0.3):
        super().__init__()

        # Load EfficientNet-B0
        if num_channels != 3:
            # torchvision's EfficientNet expects 3 channels; if different, create a small conv stem
            self.stem = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1, bias=False)
        else:
            self.stem = None

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Freeze backbone features by default (fine tune only last block)
        for p in self.backbone.features.parameters():
            p.requires_grad = False

        # Unfreeze the last two feature blocks to allow some adaptation
        # EfficientNet features is an nn.Sequential of MBConv blocks; unfreeze last two
        try:
            for p in self.backbone.features[-2:].parameters():
                p.requires_grad = True
        except Exception:
            # defensive: if indexing fails, unfreeze last block
            for p in self.backbone.features[-1].parameters():
                p.requires_grad = True

        # Attach CBAM attention to the last feature channel size
        last_chan = self.backbone.features[-1][0].conv[0][0].out_channels \
            if hasattr(self.backbone.features[-1][0], 'conv') else 1280
        # Fallback to 1280 which EfficientNet-B0 uses
        if not isinstance(last_chan, int):
            last_chan = 1280
        self.cbam = CBAM(last_chan, reduction=16, kernel_size=7)

        # Regression head: global pool -> FC -> dropout -> final linear
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(last_chan, 256)
        self.fc2 = nn.Linear(256, 1)

        self.unfrozen = False

        # Initialize small head
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def unfreeze_backbone(self):
        """Unfreeze entire backbone for fine-tuning (call when ready)."""
        if not self.unfrozen:
            for p in self.backbone.features.parameters():
                p.requires_grad = True
            # also unfreeze stem if present
            if self.stem is not None:
                for p in self.stem.parameters():
                    p.requires_grad = True
            self.unfrozen = True
            print(">>> StudentModel: backbone unfrozen for fine-tuning.")

    def forward(self, x):
        # optional stem to convert num_channels -> 3
        if self.stem is not None:
            x = self.stem(x)

        # EfficientNet feature extractor (all layers except classifier)
        x = self.backbone.features(x)   # [B, C, H, W]

        # attention on top of last feature map
        x = self.cbam(x)

        # global pooling and head
        x = self.pool(x)               # [B, C, 1, 1]
        x = torch.flatten(x, 1)        # [B, C]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                # [B, 1]
        return x.squeeze(1)            # [B]

def get_model(model_name='simple_cnn', **kwargs):
    return StudentModel(**kwargs)
