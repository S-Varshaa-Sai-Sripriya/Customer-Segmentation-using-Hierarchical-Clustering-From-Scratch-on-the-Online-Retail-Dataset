"""Evaluation runner placeholder for hierarchical_customer_segmentation"""
from src.agglomerative import AgglomerativeClustering
from src.feature_engineering import compute_rfm, scale_features
from src.evaluation import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd


def run_evaluation_demo():
    """Small demo using synthetic data to show evaluation metrics."""
    # synthetic 2D points forming two clusters
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [5.0, 5.0], [5.1, 5.0], [4.9, 5.2]])

    model = AgglomerativeClustering(n_clusters=2, linkage="single")
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print("Evaluation demo")
    print("Labels:", labels)
    print(f"Silhouette score: {sil:.4f}")
    print(f"Davies-Bouldin index: {db:.4f}")


if __name__ == "__main__":
    run_evaluation_demo()
