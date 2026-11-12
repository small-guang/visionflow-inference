import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
import io
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """TBPS CLIP图像预处理"""
    
    def initialize(self, args):
        """初始化模型配置"""
        self.model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "image")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])
        
      
        self.image_resolution = (224, 224)
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_image(self, image_bytes, batch_index=0):
        """图像预处理"""
        if not isinstance(image_bytes, bytes):
            raise ValueError(f"image at batch index {batch_index} must be bytes")
        if not image_bytes:
            raise ValueError(f"image at batch index {batch_index} is empty")
        try:
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                image = decoded.convert('RGB')
            image = image.resize(self.image_resolution, Image.Resampling.BICUBIC)
            img_np = np.asarray(image, dtype=np.float32) / 255.0
            img_np = (img_np - self.image_mean) / self.image_std
            img_np = img_np.transpose((2, 0, 1))
            return np.ascontiguousarray(img_np, dtype=self.output_dtype)
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise ValueError(
                f"unable to decode image at batch index {batch_index}"
            ) from e

    def _execute_request(self, request):
        input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if input_tensor is None:
            raise ValueError("missing required input 'IMAGE_BYTES'")
        image_bytes = input_tensor.as_numpy()
        if image_bytes.ndim != 2 or image_bytes.shape[1] != 1:
            raise ValueError(
                "input 'IMAGE_BYTES' must have shape [batch, 1], "
                f"got {image_bytes.shape}"
            )
        if image_bytes.dtype != np.object_:
            raise ValueError("input 'IMAGE_BYTES' must have dtype object (BYTES)")
        if image_bytes.shape[0] == 0:
            raise ValueError("input 'IMAGE_BYTES' batch is empty")
        output = np.ascontiguousarray(
            np.stack(
                [
                    self._preprocess_image(image_bytes[index, 0], index)
                    for index in range(image_bytes.shape[0])
                ],
                axis=0,
            )
        )
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("image", output)]
        )

    async def execute(self, requests):
        """处理推理请求"""
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"TBPS CLIP image preprocessing failed: {exc}"
                        )
                    )
                )
            
        return responses

    def finalize(self):
        """模型释放回调"""
        logger.info("清理TBPS CLIP图像预处理模型")
