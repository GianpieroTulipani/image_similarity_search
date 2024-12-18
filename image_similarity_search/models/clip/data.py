import os
from pathlib import Path

import clip
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
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
        description = labels["DESCRIPTION"]

        assert os.path.exists(path), f"Image {path} does not exist"
        image = Image.open(path).convert("RGB")
        description = clip.tokenize([description], truncate=True)
        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
        return image, description

    @staticmethod
    def collate_fn(batch):
        images, description = zip(*batch)
        return torch.stack(images), torch.vstack(description)


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
