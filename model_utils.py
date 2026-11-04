import os
import json
import base64
import io
import logging
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger("petvision.model")

# Standard ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


class GradCAM:
    """
    Grad-CAM implementation for PyTorch convolutional networks.
    Computes class activation map with respect to target convolutional layer (e.g. layer4).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        target_score = output[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling across H, W of gradients
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize 0 to 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam


class PetClassifier:
    """
    Loads fine-tuned ResNet50 model and provides breed inference + Grad-CAM generation.
    """
    def __init__(self, model_path: str = "model/pet_model.pt", class_names_path: str = "model/class_names.json"):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.class_names: List[str] = []
        self.model: nn.Module = None
        self.grad_cam: GradCAM = None
        self.load()

    def _load_class_names(self):
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded {len(self.class_names)} classes from {self.class_names_path}")
        else:
            logger.warning(f"Class names file '{self.class_names_path}' not found! Creating default 132 classes.")
            self.class_names = [f"dog_Breed_{i}" for i in range(120)] + [f"cat_Breed_{i}" for i in range(12)]

    def load(self):
        self._load_class_names()
        num_classes = len(self.class_names) if self.class_names else 132

        # Initialize ResNet50 architecture
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=torch.device("cpu"))
                if isinstance(state, dict):
                    if "state_dict" in state:
                        state = state["state_dict"]
                    elif "model" in state:
                        state = state["model"]

                # Clean any 'module.' prefix if saved from DataParallel
                cleaned_state = {}
                for k, v in state.items():
                    key = k.replace("module.", "") if k.startswith("module.") else k
                    cleaned_state[key] = v

                self.model.load_state_dict(cleaned_state)
                logger.info(f"Successfully loaded model weights from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights from {self.model_path}: {e}. Running with initialized weights.")
        else:
            logger.warning(
                f"Model file '{self.model_path}' not found yet. Initialized ResNet50 architecture ready for weights."
            )

        self.model.eval()
        # Initialize Grad-CAM on layer4
        self.grad_cam = GradCAM(self.model, self.model.layer4)

    @staticmethod
    def format_breed_name(raw_name: str) -> Tuple[str, str]:
        """
        Parses class name format like 'dog_German_Shepherd' or 'cat_Persian'.
        Returns (animal_type, human_readable_breed).
        """
        raw_name = raw_name.strip()
        lower_name = raw_name.lower()

        if lower_name.startswith("dog_"):
            animal = "Dog"
            breed = raw_name[4:].replace("_", " ").strip()
        elif lower_name.startswith("cat_"):
            animal = "Cat"
            breed = raw_name[4:].replace("_", " ").strip()
        else:
            animal = "Pet"
            breed = raw_name.replace("_", " ").strip()

        return animal, breed

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Runs image classification, returns top3 predictions, animal type, and Grad-CAM base64 PNG.
        """
        # Ensure RGB
        rgb_image = pil_image.convert("RGB")
        orig_w, orig_h = rgb_image.size

        # Preprocess for model input
        input_tensor = transform_pipeline(rgb_image).unsqueeze(0)  # Shape [1, 3, 224, 224]

        # Forward pass for prediction probabilities
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)  # Shape [num_classes]

        top_k = min(3, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, top_k)

        top_indices = top_indices.tolist()
        top_probs = top_probs.tolist()

        # Format Top 3 predictions
        top3 = []
        for idx, prob in zip(top_indices, top_probs):
            class_raw = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
            _, breed_name = self.format_breed_name(class_raw)
            top3.append({
                "breed": breed_name,
                "confidence": round(float(prob), 4)
            })

        # Top predicted class
        top_idx = top_indices[0]
        top_raw_class = self.class_names[top_idx] if top_idx < len(self.class_names) else "Unknown"
        animal_type, _ = self.format_breed_name(top_raw_class)

        # Generate Grad-CAM heatmap for the top class
        # Input tensor requires grad for backward hook pass in Grad-CAM
        grad_input = input_tensor.clone().requires_grad_(True)
        cam_heatmap = self.grad_cam.generate_heatmap(grad_input, top_idx)

        # Create overlay with the original image
        overlay_base64 = self._create_gradcam_overlay(rgb_image, cam_heatmap)

        return {
            "animal": animal_type,
            "top3": top3,
            "gradcam": overlay_base64
        }

    def _create_gradcam_overlay(self, original_image: Image.Image, cam_heatmap: np.ndarray) -> str:
        """
        Resizes heatmap to original image size, applies colormap, overlays with original image,
        and returns base64 PNG string data URI.
        """
        orig_w, orig_h = original_image.size
        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(cam_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend original image and heatmap
        orig_np = np.array(original_image)
        blended = np.uint8(0.55 * orig_np + 0.45 * heatmap_rgb)

        # Save to PNG in-memory buffer
        blended_pil = Image.fromarray(blended)
        buffer = io.BytesIO()
        blended_pil.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


# Global singleton instance loaded once at startup
_classifier_instance: PetClassifier = None


def get_classifier() -> PetClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PetClassifier()
    return _classifier_instance
