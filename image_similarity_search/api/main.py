import io
import json

import dagshub
import mlflow
import numpy as np
import pandas as pd
import PIL
import torch
from clip.clip import _transform
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from image_similarity_search.api.manager import response_from_dataframe
from image_similarity_search.utils import ROOT

MODEL_NAME = "CLIP"
MODEL_VERSION_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_VERSION_ALIAS}"
RESOLUTION = 224


dagshub.init(repo_owner="se4ai2425-uniba", repo_name="image_similarity_search", mlflow=True)

app = FastAPI(title="Product Image Similarity API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocess = _transform(RESOLUTION)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    model = mlflow.pytorch.load_model(MODEL_URI)
    assert hasattr(model, "encode_image"), "Model must have an `encode_image` method"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # embeddings = np.load(ROOT.parent.joinpath("models", "clip", "embeddings.npy"))

    # add model and other preprocess tools too app state
    app.package = {
        "model": model,
        "device": device,
        # "embeddings": embeddings,
        "data": pd.read_csv(ROOT.parent.joinpath("data", "interim", "dataset.csv")),
    }
    with open(ROOT.parent.joinpath("models", "clip", "sim.json"), "r", encoding="utf-8") as file:
        app.package["embeddings"] = np.load(
            ROOT.parent.joinpath("models", "clip", "embeddings.npy")
        )
        app.package["similarity"] = json.load(file)


@app.post("/api/v1/embedding")
async def embedding(immagine: UploadFile = File(...)):
    """
    Generate an embedding for the uploaded image.

    Args:
        immagine (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing the image embedding as a list,
                      or an error message if an exception occurs.

    Raises:
        PIL.UnidentifiedImageError: If the uploaded file is not a valid image.
        io.UnsupportedOperation: If there is an issue with reading the file.
        torch.TorchError: If there is an error during the model inference.
    """
    try:
        # Leggi i dati binari del file caricato
        content = await immagine.read()
        # Use Pillow to get the image
        image = PIL.Image.open(io.BytesIO(content))
        image = preprocess(image).unsqueeze(0)
        image.to(app.package["device"])
        with torch.inference_mode():
            feats = app.package["model"].encode_image(image).cpu().numpy()
        return JSONResponse(content={"embedding": feats.tolist()})
    except (PIL.UnidentifiedImageError, io.UnsupportedOperation, RuntimeError) as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v1/similarity")
async def similarity(immagine: UploadFile = File(...)):
    """
    Process the uploaded image file to find similar images.

    Args:
        immagine (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: A JSON response containing the list of similar images
            and their similarity scores.

    Raises:
        PIL.UnidentifiedImageError: If the uploaded file is not a valid image.
        io.UnsupportedOperation: If there is an issue with reading the file.
        torch.TorchError: If there is an error during the model inference.
        KeyError: If there is a missing key in the app package.
        ValueError: If there is an issue with the value during processing.
    """
    try:
        # Leggi i dati binari del file caricato
        content = await immagine.read()
        # Use Pillow to get the image
        image = PIL.Image.open(io.BytesIO(content))
        image = preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            feats = app.package["model"].encode_image(image).cpu().numpy()
            feats = feats / np.linalg.norm(feats)
        similarities = np.dot(app.package["embeddings"], feats.T)
        neighbors_index = np.argsort(similarities[:, 0])[::-1][1:11]
        neighbors_score = similarities[neighbors_index]
        results = response_from_dataframe(app.package["data"], neighbors_index.tolist())
        for i in enumerate(results):
            results[i]["similarity"] = neighbors_score[i].item()
        return JSONResponse(content=results)
    except (
        PIL.UnidentifiedImageError,
        io.UnsupportedOperation,
        torch.TorchError,
        KeyError,
        ValueError,
    ) as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v1/top10")
async def topk(index: str):
    if index not in app.package["similarity"]:
        return JSONResponse(status_code=404, content=json.dumps({"error": "ID not found"}))

    df = app.package["data"]
    neighbors_index = app.package["similarity"][index]["top_10"]
    return JSONResponse(content=response_from_dataframe(df, neighbors_index))
