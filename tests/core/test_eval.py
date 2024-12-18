import numpy as np
import pandas as pd
import pytest
from image_similarity_search.core.eval import evaluate

@pytest.fixture
def setup_data():
    # Create dummy embeddings
    embeddings = np.random.rand(10, 5)
    
    # Create dummy dataframe
    data = {
        "item_age_range_category": np.random.choice(["young", "adult", "senior"], 10),
        "product_gender_unified": np.random.choice(["male", "female"], 10),
        "product_top_category": np.random.choice(["clothing", "accessories"], 10),
        "product_type": np.random.choice(["shirt", "pants", "hat"], 10),
    }
    df = pd.DataFrame(data)
    
    return embeddings, df

def test_evaluate(setup_data):
    embeddings, df = setup_data
    metrics = evaluate(embeddings, df, n_neighbors=3, seed=42)
    
    # Check if metrics dictionary is not empty
    assert len(metrics) > 0
    expected_metrics = []
    for feature in df.columns:
        expected_metrics.extend([
            f"{feature}/accuracy",
            f"{feature}/precision",
            f"{feature}/recall",
            f"{feature}/f1",
            f"{feature}/silhouette_score",
            f"{feature}/fitness",
        ])
    # Check if all expected metrics are present    
    for metric in expected_metrics:
        assert metric in metrics
