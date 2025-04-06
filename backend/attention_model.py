import torch
import torch.nn as nn
import torch.nn.functional as F

# this class defines the spacial mechanism for the attention model
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.eye = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.squash = nn.Sigmoid()

    # so this is the foward pass which the image in go through to the model
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        stuff = torch.cat([avg_out, max_out], dim=1)

        attention_map = self.eye(stuff)
        attention_map = self.squash(attention_map)
        return attention_map


# This is the backbone of the model
class SimpleAttentionCNN(nn.Module):
    def __init__(self):
        super(SimpleAttentionCNN, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3),
        )

        self.attention = SpatialAttention(kernel_size=7)

        # this is the classification head or the model brin
        self.brain = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.backbone(x)
        
        attn_map = self.attention(feats)
        
        focused_feats = feats * attn_map
        
        out = self.brain(focused_feats)
        
        return out, attn_map
