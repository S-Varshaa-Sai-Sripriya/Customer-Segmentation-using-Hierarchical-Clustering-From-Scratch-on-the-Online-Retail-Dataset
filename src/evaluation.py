"""Evaluation metrics: Silhouette, Davies-Bouldin Index (placeholder)"""
import numpy as np


def silhouette_score(X, labels):
	"""Compute silhouette score using scikit-learn if available.

	X : array-like, shape (n_samples, n_features)
	labels : array-like, shape (n_samples,)
	"""
	try:
		from sklearn.metrics import silhouette_score as sk_silhouette

		return float(sk_silhouette(X, labels))
	except Exception as e:
		raise RuntimeError("silhouette_score requires scikit-learn: %s" % e)


def davies_bouldin_score(X, labels):
	"""Compute Davies-Bouldin score using scikit-learn if available."""
	try:
		from sklearn.metrics import davies_bouldin_score as sk_db

		return float(sk_db(X, labels))
	except Exception as e:
		raise RuntimeError("davies_bouldin_score requires scikit-learn: %s" % e)


__all__ = ["silhouette_score", "davies_bouldin_score"]
