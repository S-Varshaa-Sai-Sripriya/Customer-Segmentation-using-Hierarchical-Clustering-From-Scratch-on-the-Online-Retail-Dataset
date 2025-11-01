import unittest
import importlib


class ImportTest(unittest.TestCase):
    def test_import_src_modules(self):
        """Ensure the src modules can be imported (placeholder test)."""
        modules = [
            'src.agglomerative',
            'src.divisive',
            'src.distance_metrics',
            'src.feature_engineering',
            'src.evaluation',
            'src.visualization',
            'src.utils',
        ]
        for m in modules:
            with self.subTest(module=m):
                mod = importlib.import_module(m)
                self.assertIsNotNone(mod)


if __name__ == '__main__':
    unittest.main()
