import json
import warnings
from io import BytesIO

import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils


MAX_ENCODED_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_PIXELS = 8_388_608


class ImageLimitError(ValueError):
    pass


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "image")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )
        self.image_size = (128, 256)
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, encoded_image, batch_index):
        encoded = np.ascontiguousarray(encoded_image, dtype=np.uint8).reshape(-1)
        if encoded.size == 0:
            raise ValueError(f"image at batch index {batch_index} is empty")
        if encoded.size > MAX_ENCODED_IMAGE_BYTES:
            raise ImageLimitError(
                f"image at batch index {batch_index} exceeds encoded byte limit "
                f"of {MAX_ENCODED_IMAGE_BYTES}"
            )
        # UINT8 rows have no per-sample length; preserve Pillow's existing
        # trailing-data behavior and rely on clients to right-pad with zeros.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(encoded.tobytes())) as decoded:
                    width, height = decoded.size
                    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                        raise ImageLimitError(
                            f"image at batch index {batch_index} dimensions exceed "
                            f"limit {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}: "
                            f"{width}x{height}"
                        )
                    if width * height > MAX_IMAGE_PIXELS:
                        raise ImageLimitError(
                            f"image at batch index {batch_index} pixel count exceeds "
                            f"limit {MAX_IMAGE_PIXELS}: {width * height}"
                        )
                    image = decoded.convert("RGB")
        except ImageLimitError:
            raise
        except Exception as exc:
            raise ValueError(
                f"unable to decode image at batch index {batch_index}"
            ) from exc
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return np.ascontiguousarray(
            np.transpose(pixels, (2, 0, 1)), dtype=self.output_dtype
        )

    def _execute_request(self, request):
        input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if input_tensor is None:
            raise ValueError("missing required input 'IMAGE_BYTES'")
        input_value = input_tensor.as_numpy()
        if input_value.ndim != 2:
            raise ValueError(
                "input 'IMAGE_BYTES' must have shape [batch, encoded_bytes], "
                f"got {input_value.shape}"
            )
        batch_size = input_value.shape[0]
        if batch_size == 0:
            raise ValueError("input 'IMAGE_BYTES' batch is empty")
        if input_value.shape[1] == 0:
            raise ValueError("input 'IMAGE_BYTES' contains no encoded bytes")
        images = [
            self._preprocess(input_value[index], index)
            for index in range(batch_size)
        ]
        output = np.ascontiguousarray(np.stack(images, axis=0))
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("image", output)]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "OSNet-AIN x1.0 image preprocessing failed: "
                            f"{exc}"
                        )
                    )
                )
        return responses
