"""Distance metrics: Euclidean, Manhattan, Cosine (placeholder)"""
import numpy as np
from numpy.linalg import norm


def _to_1d(a):
	arr = np.asarray(a)
	if arr.ndim == 0:
		return arr.reshape(1)
	return arr.reshape(-1)


def euclidean(a, b):
	"""Return Euclidean distance between two 1-D arrays or lists."""
	a1 = _to_1d(a)
	b1 = _to_1d(b)
	return float(np.linalg.norm(a1 - b1))


def manhattan(a, b):
	"""Return Manhattan (L1) distance between two 1-D arrays or lists."""
	a1 = _to_1d(a)
	b1 = _to_1d(b)
	return float(np.sum(np.abs(a1 - b1)))


def cosine(a, b):
	"""Return cosine distance (1 - cosine similarity) between two 1-D arrays."""
	a1 = _to_1d(a).astype(float)
	b1 = _to_1d(b).astype(float)
	na = norm(a1)
	nb = norm(b1)
	if na == 0 or nb == 0:
		# define cosine distance as 1 when one vector is zero (no similarity)
		return 1.0
	cos_sim = float(np.dot(a1, b1) / (na * nb))
	# numerical safety
	cos_sim = max(min(cos_sim, 1.0), -1.0)
	return 1.0 - cos_sim


def pairwise_distances(X, metric="euclidean"):
	"""Compute pairwise distance matrix for 2D array X.

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	metric : 'euclidean'|'manhattan'|'cosine'

	Returns
	-------
	D : ndarray, shape (n_samples, n_samples)
	"""
	X = np.asarray(X)
	n = X.shape[0]
	D = np.zeros((n, n), dtype=float)
	for i in range(n):
		for j in range(i + 1, n):
			if metric == "euclidean":
				d = euclidean(X[i], X[j])
			elif metric == "manhattan":
				d = manhattan(X[i], X[j])
			elif metric == "cosine":
				d = cosine(X[i], X[j])
			else:
				raise ValueError("Unknown metric: %s" % metric)
			D[i, j] = D[j, i] = d
	return D


__all__ = ["euclidean", "manhattan", "cosine", "pairwise_distances"]
