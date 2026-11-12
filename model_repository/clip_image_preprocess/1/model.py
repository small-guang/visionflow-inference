import numpy as np
import triton_python_backend_utils as pb_utils
import cv2
import io
import json
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TritonPythonModel:
    """
    clip 模型图像预处理模型
    """
    def initialize(self, args):
        """
        加载模型配置
        """
        self.model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "image")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
        self.target_size = (224, 224)
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        backend = self.model_config.get("parameters", {}).get(
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
                "clip_image", pb_utils, int(args["model_instance_device_id"]))

    def _preprocess(self, img_bytes, batch_index=0):
        """
        图像预处理
        """
        if isinstance(img_bytes, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(img_bytes, dtype=np.uint8)
        else:
            encoded = np.ascontiguousarray(img_bytes, dtype=np.uint8).reshape(-1)
        if encoded.size == 0:
            raise ValueError(f"image at batch index {batch_index} is empty")
        try:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("decoder returned no pixels")
            img = cv2.resize(
                img, self.target_size, interpolation=cv2.INTER_CUBIC
            )
            # Convert only the resized pixels; normalization still expects RGB.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            return np.ascontiguousarray(img, dtype=self.output_dtype)
        except Exception as exc:
            raise ValueError(
                f"unable to decode image at batch index {batch_index}"
            ) from exc

    def _execute_request(self, request):
        in_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if in_tensor is None:
            raise ValueError("missing required input 'IMAGE_BYTES'")
        values = in_tensor.as_numpy()
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(
                "input 'IMAGE_BYTES' must have shape [batch, 1], "
                f"got {values.shape}"
            )
        if values.shape[0] == 0:
            raise ValueError("input 'IMAGE_BYTES' batch is empty")

        output = np.ascontiguousarray(
            np.stack(
                [
                    self._preprocess(values[index, 0], index)
                    for index in range(values.shape[0])
                ],
                axis=0,
            )
        )
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("image", output)]
        )

    async def execute(self, requests):
        """
        处理推理请求
        """
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
                            f"CLIP image preprocessing failed: {exc}"
                        )
                    )
                )
        return responses

    def finalize(self):
        """模型释放回调"""
        logger.info('Cleaning up model instance...')
