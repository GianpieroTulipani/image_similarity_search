import argparse

import pandas as pd
import torch
from autoencoder import Autoencoder
from core.eval import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loader import ImageDataset, transform


def infer(weights_path, image_path_col):
    image_paths = ImageDataset(image_path_col, transform=transform)
    test_loader = DataLoader(image_paths, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    embeddings = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            embedding, _ = model(data)
            embeddings.append(embedding.cpu())

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights_path", "-w", type=str)
    parser.add_argument("--testset_path", "-t", type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.testset_path, index_col=0)
    image_path_col = df["IMAGE_PATH"]
    labels = df["product_type"]

    embeddings = infer(args.weights_path, image_path_col)
    evaluate(embeddings, labels, runId="322c2ae52e554172ae185d586271fa2f")
