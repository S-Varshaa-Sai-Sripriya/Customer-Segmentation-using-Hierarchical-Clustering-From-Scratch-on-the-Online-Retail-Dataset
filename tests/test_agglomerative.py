import unittest
import numpy as np
from src.agglomerative import AgglomerativeClustering


class TestAgglomerative(unittest.TestCase):
    def test_two_clusters_single_link(self):
        X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]])
        model = AgglomerativeClustering(n_clusters=2, linkage="single")
        labels = model.fit_predict(X)
        # ensure there are exactly 2 unique labels and points split roughly
        self.assertEqual(len(set(labels)), 2)
        # first two should be in same cluster
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])


if __name__ == "__main__":
    unittest.main()
