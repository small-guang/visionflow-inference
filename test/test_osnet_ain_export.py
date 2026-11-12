import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import export_osnet_ain_onnx as exporter


class OsnetAinExporterTrustTests(unittest.TestCase):
    def test_pinned_official_source_and_license_exist(self):
        vendor = ROOT / "tools" / "vendor" / "torchreid"
        source = (vendor / "osnet_ain.py").read_text(encoding="utf-8")
        metadata = (vendor / "来源说明.md").read_text(encoding="utf-8")
        license_text = (vendor / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("def osnet_ain_x1_0", source)
        self.assertIn("f8cd150fdf77e8d9e1ed143b7f308c2c609ded50", metadata)
        self.assertIn("MIT License", license_text)

    def test_hash_mismatch_fails_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"untrusted")
            with mock.patch.object(exporter.torch, "load") as torch_load:
                with self.assertRaisesRegex(RuntimeError, "Checkpoint SHA256 mismatch"):
                    exporter.load_checkpoint(checkpoint, "0" * 64)
            torch_load.assert_not_called()


class TinyFeatureModel(nn.Module):
    def forward(self, image):
        pooled = image.mean(dim=(2, 3))
        return torch.cat([pooled, pooled], dim=1)


class AlmostUnitFeatureModel(nn.Module):
    def forward(self, image):
        feature = torch.zeros((image.shape[0], 2), dtype=torch.float32)
        feature[:, 0] = 1.000015
        return feature


class OsnetAinExporterBehaviorTests(unittest.TestCase):
    def test_normalized_wrapper_returns_unit_features(self):
        wrapper = exporter.NormalizedOSNet(TinyFeatureModel()).eval()
        output = wrapper(torch.ones((4, 3, 8, 4), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (4, 6))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(4))

    def test_export_locks_static_embedding_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            wrapper = exporter.NormalizedOSNet(
                TinyFeatureModel(), embedding_size=6
            ).eval()
            digest = exporter.export_release(wrapper, output, embedding_size=6)
            self.assertEqual(digest, exporter.file_sha256(output))
            self.assertEqual(set(output.parent.iterdir()), {output})
            model = exporter.onnx.load(str(output))
            self.assertEqual(
                exporter.graph_shape(model.graph.output[0]),
                ("batch", 6),
            )

    def test_classifier_shape_is_locked_to_msmt17_combineall(self):
        state = {
            "classifier.weight": torch.zeros((1041, 512)),
            "classifier.bias": torch.zeros(1041),
        }
        with self.assertRaisesRegex(RuntimeError, "4101"):
            exporter.validate_checkpoint_identity(state)

    def test_norm_deviation_above_absolute_tolerance_is_rejected(self):
        example = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "not L2-normalized"):
            exporter.validate_torch_output(
                AlmostUnitFeatureModel(),
                example,
                embedding_size=2,
            )

    def test_validation_failure_preserves_existing_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.onnx"
            output.write_bytes(b"previously validated")
            entries = set(output.parent.iterdir())
            with mock.patch.object(
                exporter,
                "validate_candidate",
                side_effect=RuntimeError("forced validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced validation failure"):
                    exporter.export_release(
                        TinyFeatureModel().eval(), output, embedding_size=6
                    )
            self.assertEqual(output.read_bytes(), b"previously validated")
            self.assertEqual(set(output.parent.iterdir()), entries)


if __name__ == "__main__":
    unittest.main(verbosity=2)
