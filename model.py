import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=4):
    # Use pretrained ResNet18 (transfer learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer with our classifier
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


# Class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]