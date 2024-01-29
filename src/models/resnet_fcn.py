import torch.nn as nn
from torchvision import models


class ResNetFCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFCN, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet50 = models.resnet50(pretrained=True)

        # Remove the fully connected layer and avgpool
        self.features = nn.Sequential(*list(self.resnet50.children())[:-2])

        # Add new layers, including convolution for class prediction and upsampling
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)  # Example classifier layer
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # Example upsampling layer

    def forward(self, x):
        x = self.features(x)  # Extract features
        x = self.classifier(x)  # Classify each pixel
        x = self.upsample(x)  # Upsample to original image size
        return x
