import json

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "image")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )
        self.image_size = (128, 384)  # OpenCV expects width, height.
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        backend = model_config.get("parameters", {}).get(
            "preprocess_backend", {}).get("string_value", "cpu")
        if backend not in ("cpu", "orin_gpu"):
            raise ValueError(f"unsupported preprocess_backend: {backend!r}")
        self._gpu_runtime = None
        if backend == "orin_gpu":
            if args.get("model_instance_kind") != "GPU":
                raise ValueError("orin_gpu preprocessing requires a KIND_GPU instance")
            if self.output_dtype != np.float32:
                raise ValueError("orin_gpu preprocessing requires FP32 image output")
            import sys
            sys.path.insert(0, "/usr/local/bin")
            from image_orin_preprocess import ImagePreprocessRuntime
            self._gpu_runtime = ImagePreprocessRuntime(
                "sbs", pb_utils, int(args["model_instance_device_id"]))

    def _preprocess(self, encoded_image, batch_index):
        if not isinstance(encoded_image, bytes):
            raise ValueError(f"image at batch index {batch_index} must be bytes")
        if not encoded_image:
            raise ValueError(f"image at batch index {batch_index} is empty")
        encoded = np.frombuffer(encoded_image, dtype=np.uint8)

        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(
                f"unable to decode image bytes at batch index {batch_index}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        pixels = image.astype(np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return np.ascontiguousarray(
            np.transpose(pixels, (2, 0, 1)), dtype=self.output_dtype
        )

    def _execute_request(self, request):
        input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if input_tensor is None:
            raise ValueError("missing required input 'IMAGE_BYTES'")

        input_value = input_tensor.as_numpy()
        if input_value.ndim != 2 or input_value.shape[1] != 1:
            raise ValueError(
                "input 'IMAGE_BYTES' must have shape [batch, 1], "
                f"got {input_value.shape}"
            )
        if input_value.dtype != np.object_:
            raise ValueError("input 'IMAGE_BYTES' must have dtype object (BYTES)")

        batch_size = input_value.shape[0]
        if batch_size == 0:
            raise ValueError("input 'IMAGE_BYTES' batch is empty")
        images = [
            self._preprocess(input_value[index, 0], index)
            for index in range(batch_size)
        ]
        output = np.ascontiguousarray(np.stack(images, axis=0))
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("image", output)]
        )

    def execute(self, requests):
        if getattr(self, "_gpu_runtime", None) is not None:
            return self._gpu_runtime.execute(requests)
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"SBS R50-IBN image preprocessing failed: {exc}"
                        )
                    )
                )
        return responses
