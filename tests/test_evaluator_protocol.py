"""Cloud-only regression checks for pairing and unresampled SOTS alignment."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_wdmamba import align_gt, audit_pairs, evaluate, metric_pair, padded_size, pair_paths


class EvaluationProtocolTests(unittest.TestCase):
    def test_sots_crop_preserves_exact_pixels(self):
        gt = torch.arange(3 * 48 * 64).reshape(1, 3, 48, 64).float()
        actual = align_gt(gt, (28, 44), border=10, resize=False)
        self.assertTrue(torch.equal(actual, gt[:, :, 10:38, 10:54]))
        with self.assertRaises(ValueError):
            align_gt(gt, (28, 44), border=0, resize=False)

    def test_haze4k_grid32_ssim_reference(self):
        torch.manual_seed(7)
        pred = torch.rand(1, 3, 400, 400)
        label = torch.rand(1, 3, 400, 400)
        self.assertEqual(padded_size((400, 400), 32), (416, 416))
        native = metric_pair(pred, label)[1]
        grid32 = metric_pair(pred, label, padded_size((400, 400), 32))[1]
        self.assertNotEqual(native, grid32)

    def test_pairing_census_and_geometry_before_smoke_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hazy, gt = root / "hazy", root / "gt"
            hazy.mkdir()
            gt.mkdir()
            Image.new("RGB", (44, 28)).save(hazy / "0001_0.8_0.2.jpg")
            Image.new("RGB", (64, 48)).save(gt / "0001.png")
            pairs = pair_paths(hazy, gt, expected_count=1)
            self.assertEqual(pairs[0][1].name, "0001.png")
            self.assertEqual(len(audit_pairs(pairs, border=10, resize=False)), 1)
            with self.assertRaises(ValueError):
                audit_pairs(pairs, border=0, resize=False)
            with self.assertRaises(RuntimeError):
                pair_paths(hazy, gt, expected_count=500)
            Image.new("RGB", (44, 28)).save(hazy / "9999_1.png")
            with self.assertRaises(RuntimeError):
                pair_paths(hazy, gt, limit=1)

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                pair_paths(Path(temp), Path(temp))

    def test_existing_run_is_never_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "status.txt").write_text("original")
            args = SimpleNamespace(gt_border=0, resize_gt=False, max_images=0,
                                   expected_count=0, print_freq=10, alphas=[0, 1], out_dir=root)
            with self.assertRaises(FileExistsError):
                evaluate(args)
            self.assertEqual((root / "status.txt").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
