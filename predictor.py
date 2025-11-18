import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import joblib

class SimplePCEClassifier:
    def __init__(self, classifier_model_path):
        """Simplified classifier using only online ResNet"""
        # Load pre-trained ResNet from torchvision
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()  # Remove last layer for features
        self.resnet.eval()
        self.resnet = self.resnet.to('cpu')  # Force CPU
        
        # Load your custom classifier
        self.model = joblib.load(classifier_model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, image_bytes):
        """Extract features using online ResNet"""
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_tensor = self.transform(image).unsqueeze(0).to('cpu')
        
        with torch.no_grad():
            features = self.resnet(image_tensor)
            
        return features.numpy().squeeze()
    
    def predict(self, image_bytes):
        """Predict classification"""
        features = self.extract_features(image_bytes).reshape(1, -1)
        pred = self.model.predict(features)
        prob = self.model.predict_proba(features).max()
        label = "High" if pred[0] == 1 else "Low"

        return {"prediction": label, "confidence": float(prob)}
