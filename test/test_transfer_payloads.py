import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from test.test_batch_python_models import (
    FakeTensor,
    load_python_model,
    make_encoded_images,
    response_output,
    run_execute,
)
from test.test_yolo26_end2end_postprocess import (
    META_640,
    SEG_MODULE,
    initialized_model,
    install_fake_pb_utils,
    output_map,
    request,
    SEG_CLIENT_MODULE,
)


IMAGE_MODELS = (
    "sbs_r50_ibn_image_preprocess",
    "TBPS_clip_image_preprocess",
)


class PackedMaskTransferTests(unittest.TestCase):
    def setUp(self):
        install_fake_pb_utils(SEG_MODULE)
        self.model = initialized_model(SEG_MODULE)

    def test_big_endian_rows_preserve_asymmetric_masks_and_sample_alignment(self):
        # Identity sampling from a 160x160 prototype makes every bit independent
        # of a packing helper: 10110001 -> 177, 01001110 -> 78.
        pattern = np.tile([1, 0, 1, 1, 0, 0, 0, 1], 20)
        binary = np.tile(pattern, (160, 1)).astype(np.uint8)
        binary[1::2] = 1 - binary[1::2]
        proto = np.stack([binary * 2.0 - 1.0, 1.0 - binary * 2.0])
        proto = proto[:, None].astype(np.float32)
        raw = np.zeros((2, 1, 7), dtype=np.float32)
        raw[:, 0] = [0, 0, 640, 640, 0.9, 0, 1]
        response = self.model.execute([
            request(RAW_OUTPUT=raw, MASK_PROTO=proto,
                    LETTERBOX_META=np.stack([META_640, META_640]))
        ])[0]

        self.assertIsNone(response.error)
        outputs = output_map(response)
        self.assertEqual(set(outputs), {"DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"})
        packed = outputs["MASK_CROPS_PACKED"]
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(packed.shape, (2, 1, 160, 20))
        np.testing.assert_array_equal(packed[0, 0, 0], [177] * 20)
        np.testing.assert_array_equal(packed[0, 0, 1], [78] * 20)
        np.testing.assert_array_equal(packed[1, 0, 0], [78] * 20)
        np.testing.assert_array_equal(packed[1, 0, 1], [177] * 20)
        np.testing.assert_array_equal(
            np.unpackbits(packed[:, 0], axis=-1, count=160, bitorder="big"),
            np.stack([binary, 1 - binary]),
        )
        np.testing.assert_array_equal(outputs["NUM_DETECTIONS"], [[1], [1]])
        self.assertTrue(packed.flags.c_contiguous)
        self.assertEqual(packed.nbytes, 2 * 160 * 160 // 8)

    def test_only_packed_batch_is_allocated_and_empty_slots_remain_zero(self):
        raw = np.zeros((2, 2, 7), dtype=np.float32)
        raw[0, :, :] = [0, 0, 640, 640, 0.9, 0, 1]
        raw[0, 1, 6] = -1
        proto = np.ones((2, 1, 1, 1), dtype=np.float32)
        shapes = []
        allocate = SEG_MODULE._allocate_zeros

        def record(shape, dtype):
            shapes.append(tuple(shape))
            return allocate(shape, dtype)

        with mock.patch.object(SEG_MODULE, "_allocate_zeros", side_effect=record):
            response = self.model.execute([
                request(RAW_OUTPUT=raw, MASK_PROTO=proto,
                        LETTERBOX_META=np.stack([META_640, META_640]))
            ])[0]
        self.assertIsNone(response.error)
        outputs = output_map(response)
        self.assertIn("MASK_CROPS_PACKED", outputs)
        packed = outputs["MASK_CROPS_PACKED"]
        self.assertEqual(packed.shape, (2, 2, 160, 20))
        np.testing.assert_array_equal(packed[0, 0], np.full((160, 20), 255, dtype=np.uint8))
        self.assertFalse(packed[0, 1].any())
        self.assertFalse(packed[1].any())
        np.testing.assert_array_equal(outputs["NUM_DETECTIONS"], [[2], [0]])
        self.assertFalse(any(len(shape) > 2 and shape[-2:] == (160, 160) for shape in shapes), shapes)


class BytesImageTransferTests(unittest.TestCase):
    @staticmethod
    def infer(model, value):
        return run_execute(model, [{"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", value)}])[0]

    @staticmethod
    def reference(model_name, encoded):
        if model_name == "sbs_r50_ibn_image_preprocess":
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            image = cv2.resize(rgb, (128, 384), interpolation=cv2.INTER_CUBIC)
        else:
            with Image.open(io.BytesIO(encoded)) as decoded:
                image = decoded.convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))

    def test_unequal_complete_encoded_elements_match_original_preprocessing_exactly(self):
        first, second = make_encoded_images()
        values = [first.tobytes() + b"\x00\x00", second.tobytes(), first.tobytes(), second.tobytes() + b"\x00"]
        self.assertNotEqual(len(values[0]), len(values[1]))
        batch = np.array([[value] for value in values], dtype=object)
        for name in IMAGE_MODELS:
            with self.subTest(model=name):
                response = self.infer(load_python_model(name), batch)
                self.assertIsNone(response.error)
                actual = response_output(response, "image")
                np.testing.assert_array_equal(actual, np.stack([self.reference(name, value) for value in values]))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(batch[:, 0].tolist(), values)

    def test_decoder_receives_original_bytes_including_trailing_nuls(self):
        encoded = make_encoded_images()[0].tobytes() + b"\x00\x00\x00"
        batch = np.array([[encoded]], dtype=object)
        for name in IMAGE_MODELS:
            with self.subTest(model=name):
                model = load_python_model(name)
                seen = []
                if name == "sbs_r50_ibn_image_preprocess":
                    decode = cv2.imdecode

                    def observe(value, flags):
                        seen.append(value.tobytes())
                        return decode(value, flags)

                    with mock.patch.object(cv2, "imdecode", side_effect=observe):
                        response = self.infer(model, batch)
                else:
                    decode = Image.open

                    def observe(value, *args, **kwargs):
                        seen.append(value.getvalue())
                        return decode(value, *args, **kwargs)

                    with mock.patch.object(Image, "open", side_effect=observe):
                        response = self.infer(model, batch)
                self.assertIsNone(response.error)
                self.assertEqual(seen, [encoded])
                np.testing.assert_array_equal(response_output(response, "image")[0], self.reference(name, encoded))

    def test_invalid_second_element_is_indexed_and_requests_are_isolated(self):
        encoded = make_encoded_images()[0].tobytes()
        good = np.array([[encoded]], dtype=object)
        for name in IMAGE_MODELS:
            for invalid in (None, 7, "SECRET_PAYLOAD", bytearray(encoded), memoryview(encoded), b"", b"SECRET_PAYLOAD"):
                with self.subTest(model=name, invalid_type=type(invalid).__name__):
                    bad = np.empty((2, 1), dtype=object)
                    bad[:, 0] = [encoded, invalid]
                    responses = run_execute(load_python_model(name), [
                        {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", bad)},
                        {"IMAGE_BYTES": FakeTensor("IMAGE_BYTES", good)},
                    ])
                    self.assertIsNotNone(responses[0].error)
                    self.assertIn("batch index 1", str(responses[0].error))
                    self.assertNotIn("SECRET_PAYLOAD", str(responses[0].error))
                    self.assertIsNone(responses[1].error)

    def test_wrong_shape_and_non_object_arrays_are_rejected(self):
        encoded = make_encoded_images()[0].tobytes()
        invalid_values = (
            np.array([encoded], dtype=object),
            np.array([[encoded, encoded]], dtype=object),
            np.empty((0, 1), dtype=object),
            np.empty((1, 0), dtype=object),
            np.array([[encoded]], dtype="S"),
            np.zeros((1, 1), dtype=np.uint8),
        )
        for name in IMAGE_MODELS:
            model = load_python_model(name)
            for value in invalid_values:
                with self.subTest(model=name, shape=value.shape, dtype=value.dtype):
                    self.assertIsNotNone(self.infer(model, value).error)


class TransferExampleTests(unittest.TestCase):
    def test_sbs_input_is_one_complete_bytes_element(self):
        from test.test_sbs_r50_ibn_image import make_input

        encoded = make_encoded_images()[0].tobytes() + b"\x00\x00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            path.write_bytes(encoded)
            batch = make_input(path, 1)
        self.assertEqual(batch.dtype, np.object_)
        self.assertEqual(batch.shape, (1, 1))
        self.assertEqual(batch[0, 0], encoded)

    def test_segmentation_example_unpacks_only_selected_mask_prefix(self):
        packed = np.full((1, 2, 160, 20), 177, dtype=np.uint8)
        packed[0, 1] = 255
        detections = np.zeros((1, 2, 6), dtype=np.float32)
        try:
            selected, masks, count = SEG_CLIENT_MODULE.select_batch_outputs(
                detections, packed, np.array([[1]], dtype=np.int32)
            )
        except ValueError as exc:
            self.fail(f"valid packed mask response was rejected: {exc}")
        self.assertEqual(count, 1)
        self.assertEqual(selected.shape, (1, 6))
        self.assertEqual(masks.shape, (1, 160, 160))
        np.testing.assert_array_equal(masks[0, 0], [1, 0, 1, 1, 0, 0, 0, 1] * 20)


if __name__ == "__main__":
    unittest.main()
