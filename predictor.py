import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import joblib

class PCEFeatureExtractor:
    def __init__(self, model_path):
        # ✅ Force CPU to avoid CUDA OOM
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
        """Extract features from image bytes (CPU only)."""
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_tensor = self.transform(image).unsqueeze(0)  # 👈 no .to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.numpy().squeeze()

class PCEClassifier:
    def __init__(self, feature_model_path, classifier_model_path):
        self.feature_extractor = PCEFeatureExtractor(feature_model_path)
        self.model = joblib.load(classifier_model_path)
    
    def predict(self, image_bytes):
        features = self.feature_extractor.extract_features(image_bytes).reshape(1, -1)
        pred = self.model.predict(features)
        prob = self.model.predict_proba(features).max()
        label = "High" if pred[0] == 1 else "Low"

        return {"prediction": label, "confidence": float(prob)}
