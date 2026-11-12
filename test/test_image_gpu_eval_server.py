"""Contract checks for the isolated image evaluation server."""
import unittest
import tempfile
from pathlib import Path
import sys
import types
from unittest.mock import patch

from tools.image_gpu_eval_server import ImageGpuEvalModel, MODELS, VARIANTS, choose_profile, prepare_repository, validate_encoded_batch
import numpy as np


class EvalServerContractTests(unittest.TestCase):
    def test_neon_requires_explicit_model_and_variant_selection(self):
        self.assertEqual(VARIANTS, ("cpu_fused", "gpu_normalize", "gpu_resize"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "models"
            with self.assertRaisesRegex(ValueError, "tbps_clip_image.*gpu_resize_neon"):
                prepare_repository("unused", output, "unused", variants=["gpu_resize_neon"])
            self.assertFalse(output.exists())

    def test_neon_repository_contains_only_selected_clip_and_sbs_wrappers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, native, output = root / "source", root / "native", root / "models"
            for alias in ("clip_image", "sbs"):
                public, pre, engine, _, _, _ = MODELS[alias]
                for name in (public, pre, engine):
                    directory = source / "model_repository" / name
                    directory.mkdir(parents=True)
                    (directory / "config.pbtxt").write_text(
                        'default_model_filename: "model.plan"' if name == engine else f'name: "{name}"')
                plan = native / engine / "1" / "model.plan"
                plan.parent.mkdir(parents=True)
                plan.write_bytes(b"same-native-plan")
            # Wrapper selection does not depend on Windows symlink privileges.
            with patch.object(Path, "symlink_to", lambda link, target: link.write_bytes(Path(target).read_bytes())):
                prepare_repository(source, output, native, models=["clip_image", "sbs"],
                                   variants=["gpu_resize_neon"])
            self.assertEqual(sorted(p.name for p in output.glob("benchmark_*")),
                             ["benchmark_clip_image_gpu_resize_neon", "benchmark_sbs_gpu_resize_neon"])
            for alias in ("clip_image", "sbs"):
                wrapper = output / f"benchmark_{alias}_gpu_resize_neon"
                self.assertIn(f'dims: [{MODELS[alias][4]}]', (wrapper / "config.pbtxt").read_text())
                self.assertTrue((wrapper / "1" / "model.py").is_file())

    def test_runtime_passes_neon_rounding_only_for_new_variant(self):
        class StopAtPreprocessor(Exception):
            pass

        observed = []

        def preprocessor(alias, **kwargs):
            observed.append((alias, kwargs))
            raise StopAtPreprocessor

        device = unittest.mock.MagicMock()
        cp = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=lambda value: device,
                                                              Stream=lambda **kwargs: object()))
        modules = {"cupy": cp, "tensorrt": types.SimpleNamespace(),
                   "triton_python_backend_utils": types.SimpleNamespace(),
                   "image_gpu_candidate": types.SimpleNamespace(ImageGpuPreprocessor=preprocessor),
                   "yolo_orin_pipeline": types.SimpleNamespace(validate_tensorrt_version=lambda trt: None)}
        with patch.dict(sys.modules, modules), patch("tools.image_gpu_eval_server.load_cpu_preprocessor"):
            for alias, variant in (("clip_image", "gpu_resize_neon"), ("sbs", "gpu_resize_neon"),
                                   ("clip_image", "gpu_resize"), ("sbs", "gpu_normalize")):
                with self.assertRaises(StopAtPreprocessor):
                    ImageGpuEvalModel().initialize({"model_name": f"benchmark_{alias}_{variant}",
                                                   "model_instance_kind": "GPU", "model_repository": "unused"})
        self.assertEqual(observed[:2], [("clip_image", {"cp_module": cp, "opencv_rounding": "neon"}),
                                       ("sbs", {"cp_module": cp, "opencv_rounding": "neon"})])
        self.assertEqual(observed[2:], [("clip_image", {"cp_module": cp}), ("sbs", {"cp_module": cp})])

    def test_runtime_rejects_tbps_neon_before_importing_gpu_dependencies(self):
        with self.assertRaisesRegex(ValueError, "tbps_clip_image.*gpu_resize_neon"):
            ImageGpuEvalModel().initialize({"model_name": "benchmark_tbps_clip_image_gpu_resize_neon"})

    def test_profile_selection_requires_all_dimensions_in_bounds(self):
        profiles = [((1, 3, 224, 224), (2, 3, 224, 224), (4, 3, 224, 224))]
        self.assertEqual(choose_profile(profiles, (4, 3, 224, 224)), 0)
        for shape in ((5, 3, 224, 224), (1, 3, 384, 128), (1, 224, 224)):
            with self.assertRaises(ValueError):
                choose_profile(profiles, shape)

    def test_encoded_bytes_are_preserved_including_zero_suffix(self):
        payload = b"encoded\x00\x00"
        values = np.asarray([[payload]], dtype=object)
        self.assertEqual(validate_encoded_batch(values), [payload])

    def test_bad_batch_is_rejected_before_gpu_allocation(self):
        cases = [np.empty((0, 1), object), np.full((5, 1), b"x", object),
                 np.asarray([b"x"], object), np.asarray([[b""]], object),
                 np.asarray([["text"]], object), np.asarray([[b"x"]], dtype="S1")]
        for values in cases:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(ValueError):
                    validate_encoded_batch(values)

    def test_generated_ensembles_keep_empty_numeric_version_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, native, output = root / "source", root / "native", root / "models"
            for public, pre, engine, _, _, _ in MODELS.values():
                for name in (public, pre, engine):
                    directory = source / "model_repository" / name
                    directory.mkdir(parents=True)
                    text = 'default_model_filename: "model_throughput.plan"' if name == engine else 'name: "' + name + '"'
                    (directory / "config.pbtxt").write_text(text)
                plan = native / engine / "1" / "model_throughput.plan"
                plan.parent.mkdir(parents=True)
                plan.write_bytes(b"same-native-plan")
            with patch.object(Path, "symlink_to", lambda link, target: link.write_bytes(Path(target).read_bytes())):
                prepare_repository(source, output, native)
            for public, _, _, _, _, _ in MODELS.values():
                self.assertTrue((output / public / "1").is_dir())
            self.assertFalse(list(output.glob("*gpu_resize_neon*")))


if __name__ == "__main__":
    unittest.main()
