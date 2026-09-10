import unittest

import numpy as np

from dehaze import DEFAULT_ALPHA_GRID, blend_outputs, profile_name


class BlendTests(unittest.TestCase):
    def test_wd0375_formula(self) -> None:
        a0 = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        expert = np.array([0.6, 0.1, 1.0], dtype=np.float32)
        actual = blend_outputs(a0, expert, 0.375)
        np.testing.assert_allclose(actual, a0 + 0.375 * (expert - a0))

    def test_blend_clamps_to_image_range(self) -> None:
        a0 = np.array([0.0, 1.0], dtype=np.float32)
        expert = np.array([-1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(
            blend_outputs(a0, expert, 1.0), [0.0, 1.0]
        )

    def test_profile_name_and_grid(self) -> None:
        self.assertEqual(profile_name("wd", 0.375), "WD0375")
        self.assertEqual(
            DEFAULT_ALPHA_GRID,
            (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0),
        )

    def test_rejects_invalid_alpha_and_shape(self) -> None:
        with self.assertRaises(ValueError):
            blend_outputs(np.zeros(1), np.zeros(1), 1.1)
        with self.assertRaises(ValueError):
            blend_outputs(np.zeros(1), np.zeros(2), 0.375)


if __name__ == "__main__":
    unittest.main()
