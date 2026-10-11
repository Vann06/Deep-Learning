import math
import unittest

import torch

from src.evaluacion import horizontal_sharpness, pass_at_k


class EvaluationTests(unittest.TestCase):
    def test_horizontal_sharpness_uses_columns_not_rows(self):
        images = torch.tensor([[[[0.0, 0.25, 1.0], [0.0, 0.0, 0.0]]]])
        self.assertAlmostEqual(horizontal_sharpness(images).item(), 0.25)

    def test_pass_at_k_matches_combinations_and_boundary(self):
        for c in (0, 2, 5, 9, 14, 20):
            for k in range(1, 21):
                expected = 1 - math.comb(20 - c, k) / math.comb(20, k)
                self.assertAlmostEqual(pass_at_k(20, c, k), expected)
        self.assertLessEqual(pass_at_k(20, 2, 13), 0.9)
        self.assertGreater(pass_at_k(20, 2, 14), 0.9)


if __name__ == "__main__":
    unittest.main()
