import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

class PCEFeatureExtractor:
    def __init__(self, model_path):
        # force CPU — avoid OOM issues
        self.device = torch.device("cpu")

        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Identity()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    
    def extract_features(self, image_bytes):
        """Extract features from image bytes"""
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_tensor = self.transform(image).unsqueeze(0)  # No .to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.numpy().squeeze()

