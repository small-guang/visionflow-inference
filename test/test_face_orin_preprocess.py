"""CPU contract checks; opt-in CUDA numerical parity uses the real operators."""
import contextlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("face_orin_preprocess_test", ROOT / "scripts/face_orin_preprocess.py")
GPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU)
MODELS = {"keypoint": "face_keypoint_2d106_pre", "qa": "face_qa_pre", "recognition": "face_rec_pre"}


class Tensor:
    def __init__(self, name, value):
        self.name, self.value = name, value
        self.device = False

    def as_numpy(self):
        return self.value

    @classmethod
    def from_dlpack(cls, name, value):
        result = cls(name, value)
        result.device = True
        return result


class Response:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


PB = types.SimpleNamespace(Tensor=Tensor, InferenceResponse=Response, TritonError=RuntimeError,
                           get_input_tensor_by_name=lambda request, name: request.get(name))


def load_model(mode):
    name = MODELS[mode]
    spec = importlib.util.spec_from_file_location(name + "_gpu_test", ROOT / "model_repository" / name / "1/model.py")
    module = importlib.util.module_from_spec(spec)
    replacements = {"triton_python_backend_utils": PB}
    if importlib.util.find_spec("skimage") is None:
        replacements["skimage"] = types.SimpleNamespace(transform=types.SimpleNamespace())
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


def sampled_warp(image, matrix, size):
    """Execute the integer coordinate contract on CPU, checked against OpenCV."""
    columns, rows = GPU.affine_tables(matrix[None], size)
    xx = (columns[0, None, :, 0] + rows[0, :, None, 0]) >> 5
    yy = (columns[0, None, :, 1] + rows[0, :, None, 1]) >> 5
    sx, sy, fx, fy = xx >> 5, yy >> 5, xx & 31, yy & 31
    values = np.zeros((size, size, 3), dtype=np.int32)
    for dy in range(2):
        for dx in range(2):
            x, y = sx + dx, sy + dy
            valid = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
            weight = (fx if dx else 32 - fx) * (fy if dy else 32 - fy)
            values[valid] += image[y[valid], x[valid]].astype(np.int32) * weight[valid, None]
    return ((values + 512) >> 10).astype(np.uint8)


def rounding_boundary_matrices():
    # Geometry of the 216x261 dataset face with bbox
    # [36, 44.1875, 179.8125, 217.625]. Synthetic pixels suffice to expose
    # the row-89 interpolation-phase error without storing a person's image.
    matrix = np.array([[0.738018018018018, 0., 16.363243243243247],
                       [0., 0.738018018018018, -0.6111711711711791]], np.float64)
    matrices = np.repeat(matrix[None], 4, axis=0)
    matrices[1, 1, 2] = np.nextafter(matrix[1, 2], np.inf)
    matrices[2, 1, 2] = np.nextafter(matrix[1, 2], -np.inf)
    matrices[3, 0, 2] = np.nextafter(matrix[0, 2], np.inf)
    return matrices


def rotated_boundary_matrices():
    # Rotated inverse translations at interpolation half-phases. Native
    # warpAffine inversion differs from invertAffineTransform's softdouble;
    # fixing only row multiply-add can expose errors previously cancelled out.
    return np.array([
        [[0.4909897971873827, 0.8872094810601749, -55.14140057028945],
         [-0.8872094810601749, 0.4909897971873827, 38.84135538044254]],
        [[2.0779073349165826, -1.013279489420344, -42.56977609320305],
         [1.013279489420344, 2.0779073349165826, -119.67892567239426]],
        [[2.066200403427864, 1.093803111205059, -126.4166971753791],
         [-1.093803111205059, 2.066200403427864, -11.927167183300016]],
        # Tiny rotations must not be discarded by an approximate diagonal
        # fast path: the translation is exactly at an interpolation boundary.
        [[1., 1e-16, -40.01513671875], [-1e-16, 1., -40.]],
    ], dtype=np.float64)


class FaceAffineContractTests(unittest.TestCase):
    def test_dataset_geometry_at_multiply_add_rounding_boundary_matches_opencv(self):
        image = np.random.default_rng(106).integers(0, 256, (261, 216, 3), dtype=np.uint8)
        for matrix in rounding_boundary_matrices():
            with self.subTest(matrix=matrix.tolist()):
                expected = cv2.warpAffine(image, matrix, (192, 192), borderValue=0.)
                np.testing.assert_array_equal(sampled_warp(image, matrix, 192), expected)

    def test_rotated_inverse_at_phase_boundaries_matches_opencv(self):
        image = np.random.default_rng(841).integers(0, 256, (137, 159, 3), dtype=np.uint8)
        for matrix in rotated_boundary_matrices():
            with self.subTest(matrix=matrix.tolist()):
                expected = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.)
                np.testing.assert_array_equal(sampled_warp(image, matrix, 112), expected)

    def test_integer_sampling_matches_opencv_rotations_borders_and_half_phases(self):
        rng = np.random.default_rng(812)
        image = rng.integers(0, 256, (137, 159, 3), dtype=np.uint8)
        matrices = [np.array([[1., 0., x], [0., 1., -x]]) for x in (0, 1 / 64, .5, -2.5, 111.)]
        for _ in range(20):
            matrix = cv2.getRotationMatrix2D((79.5, 68.5), rng.uniform(-80, 80), rng.uniform(.5, 2.5))
            matrix[:, 2] += rng.uniform(-45, 45, 2)
            matrices.append(matrix)
        for matrix in matrices:
            with self.subTest(matrix=matrix.tolist()):
                expected = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.)
                np.testing.assert_array_equal(sampled_warp(image, matrix, 112), expected)

    def test_default_and_explicit_cpu_do_not_import_cuda_even_for_gpu_instance(self):
        for mode in MODELS:
            module = load_model(mode)
            for parameters in ({}, {"preprocess_backend": {"string_value": "cpu"}}):
                instance = module.TritonPythonModel()
                with mock.patch.dict(sys.modules, {"face_orin_preprocess": None, "cupy": None}):
                    instance.initialize({"model_instance_kind": "GPU", "model_config": {"parameters": parameters}})
                self.assertIsNone(instance.gpu)

    def test_gpu_config_is_explicit_and_invalid_values_fail_loading(self):
        for mode, name in MODELS.items():
            module = load_model(mode)
            config = {"parameters": {"preprocess_backend": {"string_value": "other"}}}
            with self.assertRaisesRegex(ValueError, "preprocess_backend"):
                module.TritonPythonModel().initialize({"model_config": config})
            config["parameters"]["preprocess_backend"]["string_value"] = "orin_gpu"
            with self.assertRaisesRegex(RuntimeError, "KIND_GPU"):
                module.TritonPythonModel().initialize({"model_config": config, "model_instance_kind": "CPU"})
            constructor = mock.Mock(return_value=object())
            with mock.patch.dict(sys.modules, {"face_orin_preprocess": types.SimpleNamespace(FaceGpuPreprocessor=constructor)}), mock.patch.object(sys, "path", list(sys.path)):
                instance = module.TritonPythonModel()
                instance.initialize({"model_config": json.dumps(config), "model_instance_kind": "GPU", "model_instance_device_id": "0"})
                constructor.assert_called_once_with(0, mode)
            text = (ROOT / "model_repository" / name / "configs/orin_gpu.pbtxt").read_text()
            self.assertIn('key: "preprocess_backend" value { string_value: "orin_gpu" }', text)
            self.assertIn("kind: KIND_GPU", text)

    def test_qa_affine_preparation_does_not_touch_image_pixels(self):
        module = load_model("qa")
        image = np.zeros((112, 112, 3), np.uint8)
        landmark = module.FACE_QUALITY_REFERENCE_LANDMARKS.copy()
        with mock.patch.object(module.cv2, "warpAffine", side_effect=AssertionError("CPU warp forbidden")), mock.patch.object(module.cv2, "cvtColor", side_effect=AssertionError("CPU RGB forbidden")):
            matrix = module.prepare_affine(image, landmark)
        np.testing.assert_allclose(matrix, np.eye(2, 3), atol=1e-12)
        with self.assertRaisesRegex(module.InputValidationError, "RMS spread"):
            module.prepare_affine(image, np.zeros((5, 2), np.float32))


class DeviceView:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return DeviceView(self.value[index])

    def toDlpack(self):
        return self.value


class FaceGpuRequestTests(unittest.TestCase):
    def setUp(self):
        self.module = load_model("qa")
        self.gpu = GPU.FaceGpuPreprocessor.__new__(GPU.FaceGpuPreprocessor)
        self.gpu.mode = "qa"
        self.gpu.image_name, self.gpu.aux_name = "RAW_FACE_IMAGE", "LANDMARK"
        self.gpu.output_name = "QA_PREPROCESSED_IMAGE"
        self.gpu.device = contextlib.nullcontext()
        self.gpu.cp = types.SimpleNamespace(asnumpy=mock.Mock(side_effect=lambda value: value.value.copy()))
        self.calls = []

        def warp(images, matrices):
            self.calls.append((images.copy(), matrices.copy()))
            return DeviceView(images[:, 0, 0, 0].astype(np.float32).reshape(-1, 1, 1, 1))
        self.gpu.warp = warp

    def request(self, value, count=1, shape=(112, 112)):
        return {"RAW_FACE_IMAGE": Tensor("", np.full((count, *shape, 3), value, np.uint8)),
                "LANDMARK": Tensor("", np.repeat(self.module.FACE_QUALITY_REFERENCE_LANDMARKS[None], count, axis=0))}

    def execute(self, requests):
        return self.gpu.execute(requests, PB, self.module.prepare_affine, self.module._validate_batch,
                                self.module.InputValidationError, self.module.logger)

    def test_compatible_requests_batch_and_split_in_original_order(self):
        responses = self.execute([self.request(11), self.request(22, 2), self.request(33), self.request(44)])
        self.assertEqual([len(images) for images, _ in self.calls], [4, 1])
        self.assertEqual(self.gpu.cp.asnumpy.call_count, 2)
        for response, expected in zip(responses, ([11], [22, 22], [33], [44])):
            self.assertIsNone(response.error)
            self.assertFalse(response.output_tensors[0].device)
            self.assertTrue(response.output_tensors[0].value.flags.c_contiguous)
            np.testing.assert_array_equal(response.output_tensors[0].value.ravel(), expected)

    def test_host_bridge_failure_is_reported_for_its_group(self):
        self.gpu.cp.asnumpy.side_effect = RuntimeError("secret copy details")
        with self.assertLogs(self.module.logger, level="ERROR"):
            responses = self.execute([self.request(11), self.request(22)])
        self.assertEqual(len(self.calls), 1)
        for response in responses:
            self.assertRegex(str(response.error), "INTERNAL_ERROR: face_qa_pre failed; incident=[0-9a-f]{12}")
            self.assertNotIn("secret", str(response.error))

    def test_invalid_sample_rejects_only_its_request_before_device_work(self):
        invalid = self.request(22, 2)
        invalid["LANDMARK"].value[1, 2, 0] = np.nan
        responses = self.execute([self.request(11), invalid, self.request(33)])
        self.assertRegex(str(responses[1].error), "INVALID_INPUT: batch index 1: LANDMARK values must be finite")
        self.assertIsNone(responses[0].error)
        self.assertIsNone(responses[2].error)
        self.assertEqual(len(self.calls), 1)
        np.testing.assert_array_equal(self.calls[0][0][:, 0, 0, 0], [11, 33])

    def test_different_shapes_do_not_merge_and_gpu_failures_do_not_fallback(self):
        self.execute([self.request(11), self.request(22, shape=(128, 128))])
        self.assertEqual([images.shape[1:3] for images, _ in self.calls], [(112, 112), (128, 128)])
        self.gpu.warp = mock.Mock(side_effect=RuntimeError("secret driver details"))
        with self.assertLogs(self.module.logger, level="ERROR"):
            responses = self.execute([self.request(11)])
        self.assertRegex(str(responses[0].error), "INTERNAL_ERROR: face_qa_pre failed; incident=[0-9a-f]{12}")
        self.assertNotIn("secret", str(responses[0].error))

    def test_keypoint_keeps_fp64_affine_output_and_sample_order(self):
        self.gpu.mode = "keypoint"
        matrix = np.array([[1., 0., .123456789], [0., 1., -.987654321]], np.float64)
        responses = self.gpu.execute([self.request(5, 2)], PB, lambda image, aux: matrix.copy(),
                                    self.module._validate_batch, self.module.InputValidationError, self.module.logger)
        affine = responses[0].output_tensors[1]
        self.assertEqual(affine.name, "AFFINE_MATRIX")
        self.assertEqual(affine.value.dtype, np.float64)
        np.testing.assert_array_equal(affine.value, np.repeat(matrix[None], 2, axis=0))


@unittest.skipUnless(os.environ.get("TRITON_RUN_GPU_TESTS") == "1", "explicit CUDA validation only")
class FaceCudaNumericalParityTests(unittest.TestCase):
    def test_rounding_boundary_geometry_matches_opencv_at_batch_one_and_four(self):
        import cupy as cp
        images = np.random.default_rng(106).integers(0, 256, (4, 261, 216, 3), dtype=np.uint8)
        matrices = rounding_boundary_matrices()
        helper = GPU.FaceGpuPreprocessor(0, "keypoint")
        expected = np.stack([cv2.warpAffine(image, matrix, (192, 192), borderValue=0.)[:, :, ::-1].transpose(2, 0, 1)
                             for image, matrix in zip(images, matrices)]).astype(np.float32)
        for count in (1, 4):
            with self.subTest(batch=count):
                actual = cp.asnumpy(helper.warp(images[:count], matrices[:count]))
                np.testing.assert_array_equal(actual, expected[:count])

    def test_rotated_inverse_boundary_matches_all_modes_at_batch_one_and_four(self):
        import cupy as cp
        images = np.random.default_rng(841).integers(0, 256, (4, 137, 159, 3), dtype=np.uint8)
        matrices = rotated_boundary_matrices()
        for mode, size in (("keypoint", 192), ("qa", 112), ("recognition", 112)):
            helper = GPU.FaceGpuPreprocessor(0, mode)
            expected = np.stack([cv2.warpAffine(image, matrix, (size, size), borderValue=0.)[:, :, ::-1].transpose(2, 0, 1)
                                 for image, matrix in zip(images, matrices)]).astype(np.float32)
            if mode == "qa":
                expected = (expected / np.float32(255.) - np.float32(.5)) / np.float32(.5)
            elif mode == "recognition":
                expected = (expected - np.float32(127.5)) * np.float32(1. / 127.5)
            for count in (1, 4):
                with self.subTest(mode=mode, batch=count):
                    actual = cp.asnumpy(helper.warp(images[:count], matrices[:count]))
                    np.testing.assert_array_equal(actual, expected[:count])

    def test_real_cuda_affine_rotation_and_strided_input_match_opencv(self):
        import cupy as cp
        rng = np.random.default_rng(947)
        images = rng.integers(0, 256, (4, 149, 346, 3), dtype=np.uint8)[:, :, ::2]
        matrices = []
        for angle, scale, shift in ((25., .6, -35.5), (-47., 1.3, 22.25), (0., 1., 1 / 64), (81., 2.1, -10.75)):
            matrix = cv2.getRotationMatrix2D((86.5, 74.5), angle, scale)
            matrix[:, 2] += [shift, -shift]
            matrices.append(matrix)
        helper = GPU.FaceGpuPreprocessor(0, "keypoint")
        actual = cp.asnumpy(helper.warp(images, np.asarray(matrices)))
        expected = np.stack([cv2.warpAffine(image, matrix, (192, 192), borderValue=0.)[:, :, ::-1].transpose(2, 0, 1)
                             for image, matrix in zip(images, matrices)]).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_real_cuda_warp_matches_all_three_cpu_references(self):
        import cupy as cp
        rng = np.random.default_rng(621)
        images = rng.integers(0, 256, (4, 192, 224, 3), dtype=np.uint8)
        # Include hard edges, asymmetric colors, fractional translations and
        # crops extending outside the image. Batch 4 preserves per-face order.
        images[0] = 0
        images[0, :30, :50] = [13, 127, 255]
        images[1, ::2, ::2] = 255
        for mode in MODELS:
            module = load_model(mode)
            helper = GPU.FaceGpuPreprocessor(0, mode)
            if mode == "keypoint":
                auxiliary = np.array([[0., 0., 120., 150.], [-20.5, -10.25, 130.25, 170.5],
                                      [90., 50., 224., 192.], [40.5, 30.5, 180.5, 160.5]], np.float32)
                reference = lambda image, aux: module.preprocess(image, aux, input_size=192, input_mean=0., input_std=1.)[0]
            else:
                points = (module.FACE_QUALITY_REFERENCE_LANDMARKS if mode == "qa" else module.arcface_dst)
                auxiliary = np.stack([points * 1.25 + [15., 8.], points * .7 - [9., 8.],
                                      points * 1.4 + [55., 45.], points + [.5, -.5]]).astype(np.float32)
                reference = module.preprocess
            matrices = np.stack([module.prepare_affine(image, aux) for image, aux in zip(images, auxiliary)])
            for count in (1, 4):
                with self.subTest(mode=mode, batch=count):
                    actual = cp.asnumpy(helper.warp(images[:count], matrices[:count]))
                    expected = np.stack([reference(image, aux) for image, aux in zip(images[:count], auxiliary[:count])])
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.2e-7 if mode != "keypoint" else 0)


if __name__ == "__main__":
    unittest.main()
