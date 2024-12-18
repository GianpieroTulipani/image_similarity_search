import os
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Define image transformation
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


# Create a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform: transforms.Compose = None):
        self.data = data
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


def get_dataset(
    path: str,
    df_path: str,
) -> DataLoader:
    """
    Returns a DataLoader object for the dataset at the specified path.
    """
    df = pd.read_csv(df_path)
    path = Path(path) if isinstance(path, str) else path
    df["IMAGE_PATH"] = df["IMAGE_PATH"].apply(lambda x: path / x)
    return ImageDataset(df, transform=transform)
