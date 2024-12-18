import argparse
import json
import os
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from image_similarity_search.models.clip.data import get_dataset
from image_similarity_search.utils.torch import select_device


def load_trained_model(model_path, device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, preprocess


def dump(model, dataset, images, device, save, batch_size: int = 32, workers: int = 8):
    save = Path(save) if isinstance(save, str) else save
    device = select_device(device)
    model, preprocess = load_trained_model(model, device)
    ds = get_dataset(images, dataset)
    nd = torch.cuda.device_count()  # number of CUDA devices
    bs = min(batch_size, len(ds))
    nw = min(
        [os.cpu_count() // max(nd, 1), bs if bs > 1 else 0, workers]
    )  # number of workers
    dl = DataLoader(
        ds, batch_size=bs, num_workers=nw, pin_memory=True, collate_fn=ds.collate_fn
    )
    embeddings = compute_embeddings(model, dl, device)
    np.save(save / "embeddings.npy", embeddings)
    similarities = cosine_similarity(embeddings)
    results = {}
    for i, row in enumerate(similarities):
        # Get top 10 most similar IDs (excluding the image itself)
        top_indices = np.argsort(row)[::-1][1:11]
        top_similarities = row[top_indices]

        results[i] = {
            "top_10": top_indices.tolist(),
            "top_10_similarities": top_similarities.tolist(),
        }

    with open(save / "sim.json", "w", encoding="utf-8") as f:
        json.dump(results, f)


def compute_embeddings(model, dataloader, device):
    embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing Embeddings"):
            images = images.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=1, keepdim=True
            )  # Normalize embeddings
            embeddings.append(image_features.cpu().numpy())
    return np.vstack(embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/clip/weights/best.pth")
    parser.add_argument("--save", default="models/clip")
    parser.add_argument("--dataset", default="data/interim/dataset.csv")
    parser.add_argument("--images", default="data/interim/images/images")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    dump(
        model=args.model,
        dataset=args.dataset,
        images=args.images,
        device=args.device,
        save=args.save,
    )
