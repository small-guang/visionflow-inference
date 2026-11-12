"""TBPS Pillow-compatible GPU resize, normalization, and CHW conversion.

Pillow still decodes encoded bytes to RGB. The GPU uses the measured separable
antialiased BICUBIC algorithm with 22-bit signed coefficients and uint8 clipping
after each pass. FP32 division/subtraction order matches the original CPU path.
The caller owns the current CUDA stream and synchronization; failures propagate.
"""

from functools import lru_cache
import io
import math

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


@lru_cache(maxsize=128)
def _pillow_coefficients(source_size, target_size):
    """Pillow's full-image BICUBIC coefficients, including downsample support."""
    scale = float(np.float32(source_size)) / target_size
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    kernel_size = math.ceil(support) * 2 + 1
    bounds = np.zeros((target_size, 2), dtype=np.int32)
    coefficients = np.zeros((target_size, kernel_size), dtype=np.int32)
    inverse_filter_scale = 1.0 / filter_scale
    for output_index in range(target_size):
        center = (output_index + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        last = min(source_size, int(center + support + 0.5))
        values = []
        total = 0.0
        for source_index in range(first, last):
            distance = abs((source_index - center + 0.5) * inverse_filter_scale)
            if distance < 1.0:
                value = ((1.5 * distance - 2.5) * distance * distance) + 1.0
            elif distance < 2.0:
                value = (((distance - 5.0) * distance + 8.0) * distance - 4.0) * -0.5
            else:
                value = 0.0
            values.append(value)
            total += value
        bounds[output_index] = first, last - first
        for tap, value in enumerate(values):
            normalized = value / total if total != 0 else value
            scaled = normalized * (1 << 22)
            coefficients[output_index, tap] = int(scaled + (0.5 if scaled >= 0 else -0.5))
    return bounds, coefficients


_CUDA_SOURCE = r'''
__device__ __forceinline__ float normalize_pixel(int pixel, int channel,
    const float* mean, const float* stddev) {
    float unit = __fdiv_rn((float)pixel, 255.0f);
    return __fdiv_rn(__fsub_rn(unit, mean[channel]), stddev[channel]);
}

extern "C" __global__ void normalize_rgb(const unsigned char* src,
    float* dst, int pixels, const float* mean, const float* stddev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels * 3) return;
    int channel = i / pixels;
    int pixel = i - channel * pixels;
    dst[i] = normalize_pixel(src[pixel * 3 + channel], channel, mean, stddev);
}

extern "C" __global__ void pillow_horizontal(const unsigned char* src,
    unsigned char* dst, int source_width, int width, int source_height,
    const int* bounds, const int* weights, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= source_height * width * 3) return;
    int channel = i % 3;
    int pixel = i / 3;
    int y = pixel / width, x = pixel - y * width;
    int first = bounds[x * 2], count = bounds[x * 2 + 1];
    int value = 1 << 21;
    for (int k = 0; k < count; k++)
        value += (int)src[(y * source_width + first + k) * 3 + channel]
            * weights[x * kernel_size + k];
    dst[i] = (unsigned char)max(0, min(255, value >> 22));
}

extern "C" __global__ void pillow_vertical_normalize(const unsigned char* src,
    float* dst, int width, int height, const int* bounds, const int* weights,
    int kernel_size, const float* mean, const float* stddev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels * 3) return;
    int channel = i / pixels;
    int pixel = i - channel * pixels;
    int y = pixel / width, x = pixel - y * width;
    int first = bounds[y * 2], count = bounds[y * 2 + 1];
    int value = 1 << 21;
    for (int k = 0; k < count; k++)
        value += (int)src[((first + k) * width + x) * 3 + channel]
            * weights[y * kernel_size + k];
    int quantized = max(0, min(255, value >> 22));
    dst[i] = normalize_pixel(quantized, channel, mean, stddev);
}
'''

class TbpsGpuPreprocessor:
    """Enqueue TBPS preprocessing on the caller's current CUDA stream.

    Do not call concurrently from multiple threads. Host RGB, uploaded tables,
    device intermediates, and outputs remain retained until a recorded completion
    event reports done. Cache eviction therefore cannot free in-flight inputs.
    No method synchronizes the device or falls back to CPU image arithmetic.
    """

    def __init__(self, cp_module=None):
        self._cp = cp_module
        self._kernels = {}
        self._constants_cache = {}
        self._coefficient_cache = {}
        self._pending = []
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self._pillow_version = tuple(int(part) for part in PILLOW_VERSION.split('.')[:2])

    def _vertical_first(self, source_height, source_width):
        # Each pass clips to uint8, so its order is part of Pillow pixel parity.
        # Pillow 13 moved this decision into libImaging/Resample.c:
        # https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c
        if self._pillow_version >= (13, 0):
            return source_height > 224 and source_height - 224 > 2 * (source_width - 224)
        # 12.2/12.3 Image.resize only changes order for extremely tall images:
        # https://github.com/python-pillow/Pillow/blob/12.2.0/src/PIL/Image.py
        return (self._pillow_version >= (12, 2)
                and source_height > source_width * 100 and source_height > 224)

    def decode(self, payload, batch_index=0):
        if not isinstance(payload, bytes):
            raise ValueError(f'image at batch index {batch_index} must be bytes')
        if not payload:
            raise ValueError(f'image at batch index {batch_index} is empty')
        try:
            with Image.open(io.BytesIO(payload)) as decoded:
                return np.ascontiguousarray(np.asarray(decoded.convert('RGB')))
        except Exception as exc:
            raise ValueError(f'unable to decode image at batch index {batch_index}') from exc

    @staticmethod
    def _validate_rgb(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError('decoded image must be a NumPy uint8 RGB array')
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('decoded image must have nonempty shape [height, width, 3]')
        return np.ascontiguousarray(rgb)

    def _cuda(self):
        if self._cp is None:
            import cupy
            self._cp = cupy
        return self._cp

    def _kernel(self, name):
        if name not in self._kernels:
            self._kernels[name] = self._cuda().RawKernel(_CUDA_SOURCE, name, options=('--fmad=false',))
        return self._kernels[name]

    def _constants(self):
        cp = self._cuda()
        device = cp.cuda.runtime.getDevice()
        if device not in self._constants_cache:
            mean, std = cp.asarray(self._mean), cp.asarray(self._std)
            ready = cp.cuda.Event(disable_timing=True)
            ready.record()
            self._constants_cache[device] = mean, std, ready
        mean, std, ready = self._constants_cache[device]
        cp.cuda.get_current_stream().wait_event(ready)
        return mean, std

    def _coefficients(self, source_height, source_width):
        cp = self._cuda()
        key = cp.cuda.runtime.getDevice(), source_height, source_width
        if key not in self._coefficient_cache:
            xb, xw = _pillow_coefficients(source_width, 224)
            yb, yw = _pillow_coefficients(source_height, 224)
            arrays = tuple(cp.asarray(value) for value in (xb, xw, yb, yw))
            ready = cp.cuda.Event(disable_timing=True)
            ready.record()
            if len(self._coefficient_cache) >= 128:
                self._coefficient_cache.pop(next(iter(self._coefficient_cache)))
            self._coefficient_cache[key] = arrays, xw.shape[1], yw.shape[1], ready
        arrays, x_size, y_size, ready = self._coefficient_cache[key]
        cp.cuda.get_current_stream().wait_event(ready)
        return (*arrays, x_size, y_size)

    def _retain_until_complete(self, resources):
        done = self._cuda().cuda.Event(disable_timing=True)
        done.record()
        self._pending.append((done, resources))

    def preprocess(self, rgb):
        rgb = self._validate_rgb(rgb)
        cp = self._cuda()
        self._pending = [(done, values) for done, values in self._pending if not done.done]
        resources = [rgb]
        try:
            source_height, source_width = rgb.shape[:2]
            source = cp.asarray(rgb)
            resources.append(source)
            output = cp.empty((3, 224, 224), dtype=cp.float32)
            resources.append(output)
            mean, std = self._constants()
            resources.extend((mean, std))
            pixels = 224 * 224
            grid = ((pixels * 3 + 255) // 256,)
            if (source_width, source_height) == (224, 224):
                self._kernel('normalize_rgb')(grid, (256,), (source, output, np.int32(pixels), mean, std))
                return output
            xb, xw, yb, yw, x_size, y_size = self._coefficients(source_height, source_width)
            resources.extend((xb, xw, yb, yw))
            if self._vertical_first(source_height, source_width):
                transposed = cp.ascontiguousarray(source.transpose(1, 0, 2))
                resources.append(transposed)
                vertical_t = cp.empty((source_width, 224, 3), dtype=cp.uint8)
                resources.append(vertical_t)
                v_grid = ((source_width * 224 * 3 + 255) // 256,)
                self._kernel('pillow_horizontal')(v_grid, (256,), (transposed, vertical_t,
                    np.int32(source_height), np.int32(224), np.int32(source_width),
                    yb, yw, np.int32(y_size)))
                vertical = cp.ascontiguousarray(vertical_t.transpose(1, 0, 2))
                resources.append(vertical)
                resized = cp.empty((224, 224, 3), dtype=cp.uint8)
                resources.append(resized)
                self._kernel('pillow_horizontal')(grid, (256,), (vertical, resized,
                    np.int32(source_width), np.int32(224), np.int32(224),
                    xb, xw, np.int32(x_size)))
                self._kernel('normalize_rgb')(grid, (256,), (resized, output,
                    np.int32(pixels), mean, std))
                return output
            if source_width != 224:
                temporary = cp.empty((source_height, 224, 3), dtype=cp.uint8)
                resources.append(temporary)
                h_grid = ((source_height * 224 * 3 + 255) // 256,)
                self._kernel('pillow_horizontal')(h_grid, (256,), (source, temporary,
                    np.int32(source_width), np.int32(224), np.int32(source_height),
                    xb, xw, np.int32(x_size)))
            else:
                temporary = source
            if source_height != 224:
                self._kernel('pillow_vertical_normalize')(grid, (256,), (temporary,
                    output, np.int32(224), np.int32(224), yb, yw,
                    np.int32(y_size), mean, std))
            else:
                self._kernel('normalize_rgb')(grid, (256,), (temporary, output,
                    np.int32(pixels), mean, std))
            return output
        finally:
            # Also retain enqueued work if compilation or a later launch fails.
            self._retain_until_complete(resources)
