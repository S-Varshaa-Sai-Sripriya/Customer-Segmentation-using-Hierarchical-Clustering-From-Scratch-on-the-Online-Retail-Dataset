import unittest
import numpy as np
from src.evaluation import silhouette_score, davies_bouldin_score


class TestEvaluation(unittest.TestCase):
    def test_metrics_small(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        labels = np.array([0, 0, 1, 1])
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        self.assertTrue(-1.0 <= sil <= 1.0)
        self.assertTrue(db >= 0.0)


if __name__ == '__main__':
    unittest.main()
