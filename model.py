import torch
import torch.nn as nn
import os

# Define your model architecture (same as during training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
def load_model():
    # Get the base directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the model path relative to the base directory
    model_path = os.path.join(base_dir, 'model', 'tb_chest_xray_cnn_best.pt')
    
    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model
