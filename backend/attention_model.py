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
    A variant of the SimpleAttentionCNN that matches exactly the saved model architecture.
    Structure adjusted based on error messages to ensure compatibility with saved weights.
    """
    def __init__(self):
        super(SimpleAttentionCNN, self).__init__()

        # Adjusted to match the architecture of the saved model
        self.feature_extractor = nn.Sequential(
            # First block - conv, bn, relu, pool
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [0]
            nn.BatchNorm2d(16),                                   # [1]
            nn.ReLU(),                                            # [2]
            nn.MaxPool2d(kernel_size=2),                          # [3]
            
            # Second block - conv, bn, relu, pool, dropout
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [4]
            nn.BatchNorm2d(32),                                   # [5]
            nn.ReLU(),                                            # [6]
            nn.MaxPool2d(kernel_size=2),                          # [7]
            nn.Dropout2d(0.2),                                    # [8]
            
            # Third block - conv, bn, relu, pool, dropout
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [9]
            nn.BatchNorm2d(64),                                   # [10]
            nn.ReLU(),                                            # [11]
            nn.MaxPool2d(kernel_size=2),                          # [12]
            nn.Dropout2d(0.3),                                    # [13]
        )

        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Adjusted classifier to match expected dimensions
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 64),  # Dimensions match the saved model
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.7),
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
