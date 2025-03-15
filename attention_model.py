import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    - Takes an input feature map of shape [B, C, H, W].
    - Produces a single-channel attention map of shape [B, 1, H, W].
    - We'll multiply this attention map by the input feature map 
      to highlight important spatial regions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels=2,   # avg_out + max_out stacked along dim=1
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        input_tensor: shape [B, C, H, W]
        Returns:
            attention_map: shape [B, 1, H, W]
        """
        # Compute average and max along the channel dimension
        avg_out = torch.mean(input_tensor, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(input_tensor, dim=1, keepdim=True)  # [B,1,H,W]
        
        # Concatenate along the channel dimension
        combined = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]

        # Pass through a conv layer -> single-channel map -> sigmoid
        attention_map = self.conv2d(combined)            # [B,1,H,W]
        attention_map = self.sigmoid(attention_map)      # [B,1,H,W]
        return attention_map


class SimpleAttentionCNN(nn.Module):
    """
    - Three convolution blocks (Conv->ReLU->MaxPool).
    - A SpatialAttention module inserted after the final conv layer.
    - A simple MLP classifier (Flatten->Linear->ReLU->Dropout->Linear->Sigmoid).
    - Returns (output, attention_map) where:
        output: shape [B, 1]  (the TB probability for each image)
        attention_map: shape [B, 1, H', W'] 
    """
    def __init__(self):
        super(SimpleAttentionCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.spatial_attention = SpatialAttention(kernel_size=7)

        # After 3x (Conv->Pool), input 256x256 becomes 30x30 
        # with 64 feature maps: shape [B, 64, 30, 30].
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: shape [B,1,256,256] (batch of single-channel images).
        Returns:
            output: shape [B,1] (probability of TB).
            attention_map: shape [B,1,H',W'] (attention map after the final conv block).
        """
        features = self.feature_extractor(x)
        attention_map = self.spatial_attention(features)
        attended_features = features * attention_map
        
        output = self.classifier(attended_features)
        return output, attention_map
