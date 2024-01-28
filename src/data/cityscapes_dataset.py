# Define the necessary imports for the Cityscapes dataset
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, ToTensor, Normalize
from pathlib import Path
import os


class CityscapesDataLoader:
    def __init__(self, data_dir, split='train', mode='fine', target_type='semantic', batch_size=4, shuffle=True):
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        self.target_type = target_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
        ])

        # Initialize the Cityscapes dataset
        cityscapes_dataset = Cityscapes(root=self.data_dir, split=self.split, mode=self.mode, target_type=self.target_type, transform=self.transform)

        # DataLoader
        self.dataloader = DataLoader(cityscapes_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_dataloader(self):
        return self.dataloader

