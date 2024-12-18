import os
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# Define image transformation
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


# Create a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, dataset, images, transform: transforms.Compose = transform):
        df = pd.read_csv(dataset)
        images = Path(images) if isinstance(images, str) else images
        df["IMAGE_PATH"] = df["IMAGE_PATH"].apply(lambda x: images / x)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image using PIL
        labels = self.data.iloc[idx]
        path = labels["IMAGE_PATH"]
        assert os.path.exists(path), f"Image {path} does not exist"
        image = Image.open(path).convert("RGB")
        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
        return image, labels

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), pd.concat(labels, axis=1).T
