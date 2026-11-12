"""CLIP/SBS 的 Orin CUDA 预处理：CPU 解码，GPU cubic/RGB/normalize/CHW。

仅由 preprocess_backend=orin_gpu 启用。OpenCV 参考为 ARMv8 NEON
INTER_CUBIC 的 11-bit 系数、float32 FMA 与 ties-to-even 舍入。结果须在
目标 OpenCV/CUDA 镜像验证；不使用通用 bilinear 替换，也不静默回退 CPU。
"""

from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np


SPECS = {
    "clip_image": {
        "size": (224, 224),
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "error_prefix": "CLIP image preprocessing failed",
    },
    "sbs": {
        "size": (128, 384),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "error_prefix": "SBS R50-IBN image preprocessing failed",
    },
}


def _fma32(left, right, addend):
    """小型 cubic 权重表的 FP32 FMA，保留 double rounding 的残差。

    FP32 操作数的乘积可由 FP64 精确表示；TwoSum 恢复相加时被舍去的
    部分。若 FP64 和正好落在两个 FP32 值中点，按残差方向修正，避免
    直接 cast 的第二次 ties-to-even 舍入改变 ARM fmadd 的结果。仅用于
    此处有限、幅度很小的系数操作数，不提供通用溢出或 signed-zero 契约。
    """
    product = np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
    addend = np.asarray(addend, dtype=np.float64)
    total = product + addend
    addend_part = total - product
    residual = (product - (total - addend_part)) + (addend - addend_part)
    rounded = total.astype(np.float32)
    toward_upper = total > rounded.astype(np.float64)
    adjacent = np.nextafter(rounded, np.where(toward_upper, np.float32(np.inf), np.float32(-np.inf)))
    midpoint = (rounded.astype(np.float64) + adjacent.astype(np.float64)) * 0.5
    correct = ((total == midpoint) & (residual != 0)
               & ((residual > 0) == toward_upper))
    return np.where(correct, adjacent, rounded)


@lru_cache(maxsize=128)
def opencv_cubic_coefficients(source_size, target_size):
    """仅生成小型坐标/权重表；像素计算完全在 CUDA kernel 内。"""
    if source_size < 1 or target_size < 1:
        raise ValueError("resize dimensions must be positive")
    scale = 1.0 / (float(target_size) / source_size)
    position = ((np.arange(target_size, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    base = np.floor(position).astype(np.int32)
    x = position - base.astype(np.float32)
    one = np.float32(1)
    plus, other = x + one, one - x
    weights = np.empty((target_size, 4), dtype=np.float32)
    # OpenCV's ARM build contracts interpolateCubic to scalar fmadd. A sequence
    # of NumPy float32 multiplies/adds changes a few 11-bit coefficients by one.
    weights[:, 0] = _fma32(_fma32(_fma32(-0.75, plus, 3.75), plus, -6), plus, 3)
    weights[:, 1] = _fma32(_fma32(1.25, x, -2.25) * x, x, one)
    weights[:, 2] = _fma32(_fma32(1.25, other, -2.25) * other, other, one)
    weights[:, 3] = one - weights[:, 0] - weights[:, 1] - weights[:, 2]
    indices = np.clip(base[:, None] + np.arange(-1, 3, dtype=np.int32), 0, source_size - 1)
    return (np.ascontiguousarray(indices, dtype=np.int32),
            np.ascontiguousarray(np.rint(weights * np.float32(2048)), dtype=np.int32))


CUDA_SOURCE = r'''
extern "C" __global__ void image_cubic_bgr_batch(
    const unsigned long long* records, float* output, int width, int height,
    const float* mean, const float* stddev) {
    int sample = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels * 3) return;
    const unsigned long long* record = records + sample * 7;
    const unsigned char* src = (const unsigned char*)record[0];
    int source_width = (int)record[1], source_height = (int)record[2];
    int channel = i / pixels, bgr_channel = 2 - channel;
    int pixel = i - channel * pixels;
    int y = pixel / width, x = pixel - y * width;
    int quantized;
    if (source_width == width && source_height == height) {
        quantized = src[pixel * 3 + bgr_channel];
    } else {
        const int* xi = (const int*)record[3];
        const int* xw = (const int*)record[4];
        const int* yi = (const int*)record[5];
        const int* yw = (const int*)record[6];
        int horizontal[4];
        #pragma unroll
        for (int ky = 0; ky < 4; ky++) {
            int row = yi[y * 4 + ky];
            int value = 0;
            #pragma unroll
            for (int kx = 0; kx < 4; kx++) {
                int column = xi[x * 4 + kx];
                value += (int)src[((long long)row * source_width + column) * 3 + bgr_channel]
                    * xw[x * 4 + kx];
            }
            horizontal[ky] = value;
        }
        // OpenCV 4.10 ARMv8 VResizeCubicVec_32s8u: reversed FMA order.
        float value = __fmul_rn((float)horizontal[3],
            __fmul_rn((float)yw[y * 4 + 3], 2.384185791015625e-7f));
        value = __fmaf_rn((float)horizontal[2],
            __fmul_rn((float)yw[y * 4 + 2], 2.384185791015625e-7f), value);
        value = __fmaf_rn((float)horizontal[1],
            __fmul_rn((float)yw[y * 4 + 1], 2.384185791015625e-7f), value);
        value = __fmaf_rn((float)horizontal[0],
            __fmul_rn((float)yw[y * 4 + 0], 2.384185791015625e-7f), value);
        quantized = max(0, min(255, __float2int_rn(value)));
    }
    float unit = __fdiv_rn((float)quantized, 255.0f);
    output[(long long)sample * pixels * 3 + i] =
        __fdiv_rn(__fsub_rn(unit, mean[channel]), stddev[channel]);
}
'''


class GpuImagePreprocessor:
    """单模型实例、单自有 stream；返回已完成且拥有独立存储的 GPU NCHW。

    混合原图尺寸通过设备指针表合批，一次 kernel 写完整输出，不拼接原图。
    保留所有 CPU/GPU 临时对象到 stream 完成，避免异步拷贝提前释放。
    """

    def __init__(self, alias, device_id=0, cp_module=None):
        if alias not in SPECS:
            raise ValueError(f"unsupported image preprocessing model: {alias!r}")
        if cp_module is None:
            import cupy as cp_module
        self.alias, self.spec = alias, SPECS[alias]
        self.cp = cp_module
        self.device = cp_module.cuda.Device(int(device_id))
        self.mean = np.asarray(self.spec["mean"], dtype=np.float32)
        self.std = np.asarray(self.spec["std"], dtype=np.float32)
        self._tables = OrderedDict()
        with self.device:
            self.stream = cp_module.cuda.Stream(non_blocking=True)
            self.kernel = cp_module.RawKernel(
                CUDA_SOURCE, "image_cubic_bgr_batch", options=("--fmad=false",)
            )
            # Loading the model must fail before readiness if NVRTC cannot build.
            self.kernel.compile()
            with self.stream:
                self._mean = cp_module.asarray(self.mean)
                self._std = cp_module.asarray(self.std)
            self.stream.synchronize()

    def decode(self, encoded_image, batch_index=0):
        """沿用各模型 bytes 校验，保持 OpenCV 解码及 BGR 内存布局。"""
        if self.alias == "sbs" and not isinstance(encoded_image, bytes):
            raise ValueError(f"image at batch index {batch_index} must be bytes")
        if isinstance(encoded_image, (bytes, bytearray, memoryview)):
            encoded = np.frombuffer(encoded_image, dtype=np.uint8)
        else:
            encoded = np.ascontiguousarray(encoded_image, dtype=np.uint8).reshape(-1)
        if not encoded.size:
            raise ValueError(f"image at batch index {batch_index} is empty")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"unable to decode image bytes at batch index {batch_index}")
        return np.ascontiguousarray(image)

    def _coefficients(self, shape):
        height, width = shape
        key = height, width
        if key in self._tables:
            self._tables.move_to_end(key)
            return self._tables[key]
        target_width, target_height = self.spec["size"]
        host = (*opencv_cubic_coefficients(width, target_width),
                *opencv_cubic_coefficients(height, target_height))
        device = tuple(self.cp.asarray(value) for value in host)
        # Keep host buffers as well: a newly built table may still be copying.
        self._tables[key] = host, device
        if len(self._tables) > 64:
            self._tables.popitem(last=False)
        return host, device

    def preprocess_batch(self, images):
        if not images or len(images) > 4:
            raise ValueError("GPU image preprocessing batch must contain 1 to 4 images")
        host_images = []
        for image in images:
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3
                    or min(image.shape[:2]) < 1):
                raise ValueError("decoded image must be nonempty uint8 BGR [height,width,3]")
            host_images.append(np.ascontiguousarray(image))
        cp = self.cp
        width, height = self.spec["size"]
        sources, tables = [], []
        with self.device:
            try:
                with self.stream:
                    output = cp.empty((len(images), 3, height, width), dtype=cp.float32)
                    records = np.zeros((len(images), 7), dtype=np.uint64)
                    for index, image in enumerate(host_images):
                        source = cp.asarray(image)
                        sources.append(source)
                        records[index, :3] = source.data.ptr, image.shape[1], image.shape[0]
                        if image.shape[:2] != (height, width):
                            table = self._coefficients(image.shape[:2])
                            tables.append(table)
                            records[index, 3:] = [value.data.ptr for value in table[1]]
                    device_records = cp.asarray(records)
                    self.kernel(((3 * width * height + 255) // 256, len(images)),
                                (256,), (device_records, output, np.int32(width),
                                         np.int32(height), self._mean, self._std))
            finally:
                # All host/device locals remain alive through this synchronization.
                # The device result also remains usable by a future in-process runner.
                self.stream.synchronize()
        return output

    def to_host(self, output):
        """Orin 跨进程输出桥接；只复制已缩小的 FP32 NCHW，无 CPU 像素计算。

        目标 Jetson 运行环境不支持 Python stub -> native backend 所需的 CUDA
        IPC。桥接是固定传输路径，不是 CUDA 失败后执行 CPU 算子的 fallback。
        """
        with self.device:
            return self.cp.asnumpy(output)


class ImagePreprocessRuntime:
    """Triton 适配器：每请求校验/解码隔离，成功请求合并至最多 B4。"""

    def __init__(self, alias, pb_utils, device_id=0, processor=None):
        self.alias, self.pb = alias, pb_utils
        self.processor = processor or GpuImagePreprocessor(alias, device_id)

    def _decode_request(self, request):
        tensor = self.pb.get_input_tensor_by_name(request, "IMAGE_BYTES")
        if tensor is None:
            raise ValueError("missing required input 'IMAGE_BYTES'")
        values = tensor.as_numpy()
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"input 'IMAGE_BYTES' must have shape [batch, 1], got {values.shape}")
        if self.alias == "sbs" and values.dtype != np.object_:
            raise ValueError("input 'IMAGE_BYTES' must have dtype object (BYTES)")
        if not 1 <= values.shape[0] <= 4:
            raise ValueError("input 'IMAGE_BYTES' batch size must be between 1 and 4")
        return [self.processor.decode(values[index, 0], index)
                for index in range(values.shape[0])]

    def _error(self, exc):
        return self.pb.InferenceResponse(error=self.pb.TritonError(
            f"{SPECS[self.alias]['error_prefix']}: {exc}"
        ))

    def _run_group(self, group, responses):
        try:
            images = [image for _, decoded in group for image in decoded]
            output = self.processor.preprocess_batch(images)
            # Keep CUDA computation, but avoid unsupported Orin CUDA IPC between
            # the Python stub and Triton's native inference backend process.
            host_output = self.processor.to_host(output)
            offset = 0
            for index, decoded in group:
                size = len(decoded)
                # One D2H per group; each contiguous view retains the host owner.
                view = host_output[offset:offset + size]
                tensor = self.pb.Tensor("image", view)
                responses[index] = self.pb.InferenceResponse(output_tensors=[tensor])
                offset += size
        except Exception as exc:
            for index, _ in group:
                responses[index] = self._error(exc)

    def execute(self, requests):
        responses = [None] * len(requests)
        group, samples = [], 0
        for index, request in enumerate(requests):
            try:
                decoded = self._decode_request(request)
            except Exception as exc:
                responses[index] = self._error(exc)
                continue
            if samples + len(decoded) > 4:
                self._run_group(group, responses)
                group, samples = [], 0
            group.append((index, decoded))
            samples += len(decoded)
        if group:
            self._run_group(group, responses)
        return responses
