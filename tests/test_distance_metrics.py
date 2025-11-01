import unittest
import numpy as np
from src.distance_metrics import euclidean, manhattan, cosine, pairwise_distances


class TestDistanceMetrics(unittest.TestCase):
    def test_euclidean(self):
        a = [0, 0]
        b = [3, 4]
        self.assertAlmostEqual(euclidean(a, b), 5.0)

    def test_manhattan(self):
        a = [1, 2, 3]
        b = [4, 0, -1]
        self.assertEqual(manhattan(a, b), 9)

    def test_cosine(self):
        a = [1, 0]
        b = [0, 1]
        self.assertAlmostEqual(cosine(a, b), 1.0)

    def test_pairwise(self):
        X = np.array([[0, 0], [0, 1], [3, 4]])
        D = pairwise_distances(X, metric="euclidean")
        self.assertAlmostEqual(D[0, 2], 5.0)


if __name__ == "__main__":
    unittest.main()
