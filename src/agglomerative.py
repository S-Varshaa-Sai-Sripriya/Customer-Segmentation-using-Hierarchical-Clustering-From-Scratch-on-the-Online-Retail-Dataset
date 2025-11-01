"""Custom agglomerative clustering implementation (placeholder)"""
import numpy as np
from .distance_metrics import pairwise_distances


class AgglomerativeClustering:
	"""Simple from-scratch agglomerative clustering.

	Supports 'single' (single-linkage) and 'average' linkages.
	Not optimized — intended for small datasets and testing.
	"""

	def __init__(self, n_clusters=2, linkage="single"):
		self.n_clusters = int(n_clusters)
		if linkage not in ("single", "average"):
			raise ValueError("linkage must be 'single' or 'average'")
		self.linkage = linkage
		self.labels_ = None

	def _cluster_distance(self, X, clusters, i, j, D):
		# clusters are lists/sets of indices
		a = clusters[i]
		b = clusters[j]
		# compute pairwise distances using precomputed D
		pairs = [(p, q) for p in a for q in b]
		vals = [D[p, q] for p, q in pairs]
		if self.linkage == "single":
			return float(np.min(vals))
		else:
			return float(np.mean(vals))

	def fit(self, X):
		X = np.asarray(X)
		n = X.shape[0]
		if n == 0:
			self.labels_ = np.array([])
			return self

		D = pairwise_distances(X, metric="euclidean")

		# start with each sample in its own cluster
		clusters = [set([i]) for i in range(n)]

		while len(clusters) > self.n_clusters:
			# find closest pair
			best = None
			best_d = np.inf
			for i in range(len(clusters)):
				for j in range(i + 1, len(clusters)):
					d = self._cluster_distance(X, clusters, i, j, D)
					if d < best_d:
						best_d = d
						best = (i, j)
			# merge
			i, j = best
			new_cluster = clusters[i].union(clusters[j])
			# replace clusters[i] with merged and remove clusters[j]
			clusters[i] = new_cluster
			clusters.pop(j)

		# assign labels
		labels = np.empty(n, dtype=int)
		for idx, c in enumerate(clusters):
			for sample in c:
				labels[sample] = idx

		self.labels_ = labels
		return self

	def fit_predict(self, X):
		self.fit(X)
		return self.labels_


__all__ = ["AgglomerativeClustering"]
