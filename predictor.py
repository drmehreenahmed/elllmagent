# predictor.py - Final fixed version for ONNX + NumPy 2.x
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import logging
import onnxruntime as ort

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FEATURE EXTRACTOR
# -------------------------------------------------------------------
class PCEFeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Identity()

        try:
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location="cpu", weights_only=True)
                )
            except Exception:
                self.model.load_state_dict(
                    torch.load(model_path, map_location="cpu")
                )

            self.model.eval()
            logger.info("✅ Feature extractor loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load feature extractor: {e}")
            raise

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    def extract_features(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            img_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                features = self.model(img_tensor)

            return np.asarray(features).flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.random.rand(512).astype(np.float32)

    def extract_features_from_pil(self, pil_image):
        try:
            img_tensor = self.transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(img_tensor)
            return np.asarray(features).flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Feature extraction from PIL error: {e}")
            return np.random.rand(512).astype(np.float32)

# -------------------------------------------------------------------
# ONNX CLASSIFIER
# -------------------------------------------------------------------
class PCEClassifier:
    def __init__(self, model_path):
        self.is_loaded = False
        self.session = None
        self.input_name = None
        self.expected_dim = 512  # Known ResNet18 output

        try:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

            # Handle dynamic shapes (None)
            in_shape = self.session.get_inputs()[0].shape
            if in_shape[1] is not None:
                self.expected_dim = in_shape[1]

            self.is_loaded = True
            logger.info(f"✅ ONNX Classifier loaded: {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            self.is_loaded = False

    def predict_from_features(self, features: np.ndarray):
        if not self.is_loaded:
            return {"prediction": "Unknown", "confidence": 0.5}

        try:
            X = np.asarray(features, dtype=np.float32)

            # Ensure correct shape (1, N)
            if X.ndim == 1:
                if X.shape[0] != self.expected_dim:
                    logger.error(
                        f"❌ Wrong feature length: got {X.shape[0]}, expected {self.expected_dim}"
                    )
                    return {"prediction": "Unknown", "confidence": 0.1}

                X = X.reshape(1, -1)

            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: X})
            pred_proba = outputs[0]

            # Support for logits or probabilities
            if pred_proba.ndim == 2:  
                class_idx = int(np.argmax(pred_proba, axis=1)[0])
                confidence = float(np.max(pred_proba))
            else:
                class_idx = int(pred_proba[0])
                confidence = 0.7

            label = "High" if class_idx == 1 else "Low"

            return {"prediction": label, "confidence": confidence}

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"prediction": "Unknown", "confidence": 0.3}

# -------------------------------------------------------------------
# TEST FUNCTION
# -------------------------------------------------------------------
def test_predictor():
    print("🧪 Testing predictor...")

    clf = PCEClassifier("models/best_classifier.onnx")
    if not clf.is_loaded:
        print("❌ Classifier failed to load")
        return

    # Use safe fallback feature size
    num_features = clf.expected_dim
    print(f"ℹ️ Expected feature length: {num_features}")

    test_input = np.random.rand(num_features).astype(np.float32)
    result = clf.predict_from_features(test_input)

    print(f"✅ Classifier test: {result}")


if __name__ == "__main__":
    test_predictor()
