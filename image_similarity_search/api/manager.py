import base64
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from clip import clip
from PIL import Image

from image_similarity_search.utils import LOGGER, ROOT


def load_model(
    root: Path = ROOT.parent.joinpath("models", "clip", "pretrained"),
    device: torch.device | str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    Load CLIP pre-trained model and preprocess function.
    """
    LOGGER.info(f"Loading model from root {root} on device {device}")
    # Load the fine-tuned model
    model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False, download_root=root
    )
    return model, preprocess


def load_finetuned_model(model_path, device):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, preprocess


def response_from_dataframe(df: pd.DataFrame, indices: list[int]) -> dict:
    rows = df.iloc[indices]
    results = rows.to_dict(orient="records")
    for i, result in enumerate(results):
        path = results[i].pop("IMAGE_PATH")
        path = ROOT.parent.joinpath(
            "data", "interim", "images", "images", path.replace("/", os.sep)
        ).as_posix()
        results[i]["index"] = indices[i]
        buffered = BytesIO()
        image = Image.open(path)
        image.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue())
        results[i]["image"] = encoded.decode("utf-8")
    return results
