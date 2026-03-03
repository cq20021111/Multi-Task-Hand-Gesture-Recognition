import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiTaskHandNet(nn.Module):
    """
    Multi-task hand gesture recognition network (based on ResNet18 Backbone).
    Contains a shared encoder (ResNet18) and three task-specific heads.
    """
    def __init__(self, num_classes=10):
        super(MultiTaskHandNet, self).__init__()
        
        # --- Encoder: ResNet18 ---
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first convolution layer to accept 4-channel input (RGB + Depth)
        original_conv1 = resnet.conv1
        self.enc1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the weights of the new convolution layer
        with torch.no_grad():
            # Copy RGB channel weights
            self.enc1.weight[:, :3, :, :] = original_conv1.weight
            # Initialize Depth channel weights (e.g., using the mean of RGB channels)
            self.enc1.weight[:, 3:4, :, :] = torch.mean(original_conv1.weight, dim=1, keepdim=True)
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.enc2 = resnet.layer1 # 64 -> 64
        self.enc3 = resnet.layer2 # 64 -> 128
        self.enc4 = resnet.layer3 # 128 -> 256
        self.enc5 = resnet.layer4 # 256 -> 512
        
        # --- Decoder / Segmentation Head ---
        # U-Net style, needs to adapt to ResNet feature map channels
        # enc5: 512, enc4: 256, enc3: 128, enc2: 64, enc1: 64 (after maxpool)
        
        self.dec1 = self.upconv_block(512, 256)      # 512 -> 256
        self.dec2 = self.upconv_block(256 + 256, 128) # Cat(dec1, enc4) -> 128
        self.dec3 = self.upconv_block(128 + 128, 64)  # Cat(dec2, enc3) -> 64
        self.dec4 = self.upconv_block(64 + 64, 32)    # Cat(dec3, enc2) -> 32
        # Add an upsampling layer to fuse x0 (64 channels)
        self.dec5 = self.upconv_block(32 + 64, 16)    # Cat(dec4, x0) -> 16
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # --- Classification Head ---
        # ResNet18 layer4 output is 512 channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_cls = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # --- Bounding Box Head ---
        # Improvement: Instead of using pure GAP, preserve spatial information for regression
        # e5 output size is (B, 512, 7, 7) (for 224x224 input)
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Flattened in forward pass: 64 * 7 * 7 = 3136
        )
        
        self.fc_bbox = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 4),
            nn.Sigmoid() # Normalize to [0, 1] range as coordinates are normalized
        )

    def upconv_block(self, in_channels, out_channels):
        """Upsampling block"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder (ResNet18) ---
        # x: (B, 4, 224, 224)
        x0 = self.enc1(x)       # (B, 64, 112, 112)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        e1 = self.maxpool(x0)   # (B, 64, 56, 56)
        
        e2 = self.enc2(e1)      # (B, 64, 56, 56)
        e3 = self.enc3(e2)      # (B, 128, 28, 28)
        e4 = self.enc4(e3)      # (B, 256, 14, 14)
        e5 = self.enc5(e4)      # (B, 512, 7, 7)
        
        # --- Classification ---
        pooled = self.global_pool(e5).view(x.size(0), -1) # (B, 512)
        cls_logits = self.fc_cls(pooled)
        
        # --- Bounding Box Regression ---
        bbox_feat = self.bbox_conv(e5)
        bbox_feat = bbox_feat.view(bbox_feat.size(0), -1) # Flatten
        bbox_coords = self.fc_bbox(bbox_feat)
        
        # --- Segmentation (U-Net Decoder) ---
        d1 = self.dec1(e5)              # (B, 256, 14, 14)
        d1 = torch.cat([d1, e4], dim=1) # (B, 512, 14, 14)
        
        d2 = self.dec2(d1)              # (B, 128, 28, 28)
        d2 = torch.cat([d2, e3], dim=1) # (B, 256, 28, 28)
        
        d3 = self.dec3(d2)              # (B, 64, 56, 56)
        d3 = torch.cat([d3, e2], dim=1) # (B, 128, 56, 56)
        
        d4 = self.dec4(d3)
        # Fuse x0 (64 channels, 112x112)
        # x0 is output after conv1(stride=2), size is 112x112
        d4 = torch.cat([d4, x0], dim=1) # (B, 32+64, 112, 112)
        
        d5 = self.dec5(d4) # (B, 16, 224, 224)
        
        mask_logits = self.final_conv(d5) # (B, 1, 224, 224)
        
        return cls_logits, bbox_coords, mask_logits
