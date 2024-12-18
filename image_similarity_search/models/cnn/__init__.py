from functools import partial

from .model import AutoEncoder
from .train import train


def build(h, d, **kwargs):
    emb_dim = h.get("emb_dim", 512)
    emb_dim = kwargs.get("emb_dim", emb_dim)
    return AutoEncoder(embedding_dim=emb_dim)


MODELS = {
    "CNN/AE": {"build": build, "train": train},
    "CNN/AE_128": {"build": partial(build, embedding_dim=128), "train": train},
    "CNN/AE_256": {"build": partial(build, embedding_dim=128), "train": train},
}
