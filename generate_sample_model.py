"""
Utility script to generate an initialized sample model weights file: model/pet_model.pt
Matches the exact architecture required: PyTorch ResNet50 with final fc layer replaced by 132 outputs.
This allows PetVision to run and be tested immediately before placing final trained weights.
"""
import os
import torch
import torch.nn as nn
from torchvision import models

def generate_sample_model(output_path="model/pet_model.pt", num_classes=132):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Instantiating ResNet50 architecture for {num_classes} classes...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    print(f"Saving state_dict to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("Sample model state_dict created successfully!")

if __name__ == "__main__":
    generate_sample_model()
