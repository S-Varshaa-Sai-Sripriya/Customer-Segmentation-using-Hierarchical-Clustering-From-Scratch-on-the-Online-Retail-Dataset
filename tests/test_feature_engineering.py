import unittest
import pandas as pd
from datetime import datetime
from src.feature_engineering import compute_rfm, scale_features


class TestFeatureEngineering(unittest.TestCase):
    def test_compute_rfm(self):
        data = {
            'CustomerID': [1, 1, 2, 2, 2],
            'InvoiceDate': [
                '2020-01-01', '2020-02-01', '2020-01-15', '2020-02-20', '2020-02-25'
            ],
            'Quantity': [1, 2, 1, 1, 3],
            'UnitPrice': [10, 10, 5, 5, 5]
        }
        df = pd.DataFrame(data)
        rfm = compute_rfm(df, snapshot_date=datetime(2020, 3, 1))
        # check customers present
        self.assertIn(1, rfm.index)
        self.assertIn(2, rfm.index)
        # frequency counts
        self.assertEqual(rfm.loc[1, 'Frequency'], 2)
        self.assertEqual(rfm.loc[2, 'Frequency'], 3)

    def test_scale(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 4.0, 6.0]})
        scaled = scale_features(df)
        # means ~ 0
        self.assertAlmostEqual(float(scaled['a'].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(scaled['b'].mean()), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
