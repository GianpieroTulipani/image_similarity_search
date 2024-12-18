# Model Card: Image Similarity Search


## Table of Contents
1. [Model Details](#model-details)
2. [Intended Use](#intended-use)
3. [Factors](#factors)
4. [Metrics](#metrics)
5. [Evaluation Data](#evaluation-data)
6. [Training Data](#training-data)
7. [Quantitative Analyses](#quantitative-analyses)
8. [Ethical Considerations](#ethical-considerations)
9. [Caveats and Recommendations](#caveats-and-recommendations)


## Model Details

- **Person or organization developing the model:** Francesco Manco,Gianpiero Tulipani, Gianluca Calò, Domenico Roberto
- **Model version:** 1.0
- **Model type:** Fine-tuned version of OpenAI's CLIP model. 


## Intended Use

- **Primary intended uses:**
  - Enable users to find visually similar products on an e-commerce platform.
  - Support refined searches using product description filters.
- **Primary intended users:**
  - E-commerce customers.
- **Out-of-scope use cases:**
  - Applications requiring high precision for overlapping or ambiguous items.

## Factors

- **Relevant factors:**
  - Variability in product image quality (e.g., lighting, resolution).
  
- **Evaluation factors:**
  - Dataset diversity (e.g., including different product categories and styles).
  - Evaluation of classification fitness and silhouette scores for multiple product attributes.

## Metrics

- **Model performance measures:**
  - Accuracy: Computed using K-Nearest Neighbors (KNN) classifier with cosine similarity.
  - Precision: Calculated as macro-averaged for each product attribute.
  - Recall: Macro-averaged recall for all product categories.
  - F1-score: Harmonic mean of precision and recall across diverse features.
  - Silhouette score: Measures embedding quality and cluster separation.
- **Decision thresholds:**
  - KNN classification uses 10 neighbors.
  - Fitness score incorporates a weighted average of F1 and normalized silhouette metrics.
- **Variation approaches:**
  - Evaluated across different product attributes such as age range, gender, and category.

## Evaluation Data

- **Datasets:**
  - Dataset name: Giorgio Armani's Catalogue.
  - Source: Private Dataset.
- **Motivation:** To ensure robust performance in finding similar products of the Armani's catalogue.
- **Preprocessing:**
  - Images resized to 224x224 pixels.
  - Normalization of RGB values.

## Training Data

- **Datasets:** Giorgio Armani's Catalogue.
- **Details:**
  - Training dataset split: 80% training, 20% testing.
  - Data distribution: Equal representation of major product categories.

## Quantitative Analyses

- **Unitary results:**
  - Performance across individual categories (e.g., precision for clothing).
- **Intersectional results:**
  - Accuracy for images combining multiple factors (e.g., products with multiple attributes).

## Ethical Considerations

- Potential biases due to imbalanced data (e.g., underrepresentation of certain product styles).


## Caveats and Recommendations

- Model performance may degrade with low-resolution or poorly lit images.
- Ensure periodic evaluation to maintain fairness and robustness as the dataset evolves.

