import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier


def evaluate(
    embeddings: np.ndarray, df: pd.DataFrame, n_neighbors: int = 10, seed: int = 1337
):
    # TODO: Remove hardcoded features
    features = [
        "item_age_range_category",
        "product_gender_unified",
        "product_top_category",
        "product_type",
    ]
    metrics = {}

    metrics["fitness"] = 0
    for feature in features:
        labels = df[feature].to_list()
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(embeddings, labels)
        y_pred = knn.predict(embeddings)
        metrics[f"{feature}/accuracy"] = accuracy_score(labels, y_pred)
        metrics[f"{feature}/precision"] = precision_score(
            labels, y_pred, average="macro", zero_division=1
        )
        metrics[f"{feature}/recall"] = recall_score(
            labels, y_pred, average="macro", zero_division=1
        )
        metrics[f"{feature}/f1"] = f1_score(labels, y_pred, average="macro")
        metrics[f"{feature}/silhouette_score"] = silhouette_score(
            embeddings, labels, random_state=1337
        )
        # metrics[f'{feature}/confusion_matrix'] = confusion_matrix(labels, y_pred, labels=df[feature].unique())
        metrics[f"{feature}/fitness"] = (
            1
            - (
                (1 - metrics[f"{feature}/f1"])
                + (
                    1 - (metrics[f"{feature}/silhouette_score"] + 1) / 2
                )  # Normalized silhouette
            )
            / 2
        )  # Average distance from perfect scores
        metrics["fitness"] += metrics[f"{feature}/fitness"]
    metrics["fitness"] /= len(features)

    return metrics
