"""Standalone CuPy candidates for image preprocessing experiments only.

Nothing imports this module from the serving model repository. Decoding retains
the deployed decoder. ``normalize`` retains the original CPU resize as well;
``resize`` transfers decoded RGB and performs resize and normalization on CUDA.

Pillow uses its separable antialias filter, 22-bit signed coefficients, and uint8
rounding/clipping after *each* pass. OpenCV uses its A=-0.75, 11-bit coefficient
fixed-point cubic algorithm. OpenCV CPU SIMD/HAL implementations can differ at
rounding boundaries, so its GPU resize candidate is NOT claimed bit-exact:
measure and report parity on the target OpenCV build and real images.
The opt-in ``opencv_rounding='neon'`` candidate reproduces ARMv8 NEON float32
vertical FMA order and ties-to-even rounding; validate it against target OpenCV.

Algorithm references:
https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp
"""

from functools import lru_cache
import io
import math

import cv2
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


SPECS = {
    "clip_image": {
        "size": (224, 224), "decoder": "opencv", "resize": "opencv_cubic_fixed_point",
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "engine_model": "clip_image_trt", "input": "image",
        "output": "unnorm_image_features", "public_model": "clip_ensemble_trt",
        "public_output": "unnorm_image_features",
    },
    "tbps_clip_image": {
        "size": (224, 224), "decoder": "pillow", "resize": "pillow_bicubic_antialias_22bit",
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "engine_model": "TBPS_clip_image_trt", "input": "image",
        "output": "image_feat", "public_model": "TBPS_clip_image_ensemble",
        "public_output": "image_feat",
    },
    "sbs": {
        "size": (128, 384), "decoder": "opencv", "resize": "opencv_cubic_fixed_point",
        "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
        "engine_model": "sbs_r50_ibn_onnx", "input": "image",
        "output": "embedding", "public_model": "sbs_r50_ibn_image_ensemble",
        "public_output": "embedding",
    },
}


@lru_cache(maxsize=128)
def _opencv_coefficients(source_size, target_size):
    """Host coefficient preparation only; image arithmetic executes on CUDA."""
    # OpenCV computes a double coordinate, then casts to float before floor.
    scale = 1.0 / (float(target_size) / source_size)
    position = ((np.arange(target_size, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    base = np.floor(position).astype(np.int32)
    x = position - base.astype(np.float32)
    a = np.float32(-0.75)
    one = np.float32(1)
    plus = x + one
    other = one - x
    weights = np.empty((target_size, 4), dtype=np.float32)
    weights[:, 0] = ((a * plus - 5 * a) * plus + 8 * a) * plus - 4 * a
    weights[:, 1] = ((a + 2) * x - (a + 3)) * x * x + one
    weights[:, 2] = ((a + 2) * other - (a + 3)) * other * other + one
    weights[:, 3] = one - weights[:, 0] - weights[:, 1] - weights[:, 2]
    indices = np.clip(base[:, None] + np.arange(-1, 3, dtype=np.int32), 0, source_size - 1)
    coefficients = np.rint(weights * np.float32(2048)).astype(np.int32)
    return np.ascontiguousarray(indices, dtype=np.int32), coefficients


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

extern "C" __global__ void opencv_cubic(const unsigned char* src,
    float* dst, int source_width, int width, int height,
    const int* x_indices, const int* x_weights,
    const int* y_indices, const int* y_weights,
    const float* mean, const float* stddev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels * 3) return;
    int channel = i / pixels;
    int pixel = i - channel * pixels;
    int y = pixel / width, x = pixel - y * width;
    long long value = 0;
    #pragma unroll
    for (int ky = 0; ky < 4; ky++) {
        int row = y_indices[y * 4 + ky];
        int horizontal = 0;
        #pragma unroll
        for (int kx = 0; kx < 4; kx++) {
            int column = x_indices[x * 4 + kx];
            horizontal += (int)src[(row * source_width + column) * 3 + channel]
                * x_weights[x * 4 + kx];
        }
        value += (long long)horizontal * y_weights[y * 4 + ky];
    }
    int quantized = (int)((value + (1 << 21)) >> 22);
    quantized = max(0, min(255, quantized));
    dst[i] = normalize_pixel(quantized, channel, mean, stddev);
}

// OpenCV 4.10 VResizeCubicVec_32s8u + ARMv8 intrin_neon v_muladd/v_round.
// Target row widths (224*3 and 128*3) have no scalar SIMD tail.
extern "C" __global__ void opencv_cubic_neon(const unsigned char* src,
    float* dst, int source_width, int width, int height,
    const int* x_indices, const int* x_weights,
    const int* y_indices, const int* y_weights,
    const float* mean, const float* stddev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (i >= pixels * 3) return;
    int channel = i / pixels;
    int pixel = i - channel * pixels;
    int y = pixel / width, x = pixel - y * width;
    int horizontal[4];
    #pragma unroll
    for (int ky = 0; ky < 4; ky++) {
        int row = y_indices[y * 4 + ky];
        int value = 0;
        #pragma unroll
        for (int kx = 0; kx < 4; kx++) {
            int column = x_indices[x * 4 + kx];
            value += (int)src[(row * source_width + column) * 3 + channel]
                * x_weights[x * 4 + kx];
        }
        horizontal[ky] = value;
    }
    float value = __fmul_rn((float)horizontal[3],
        __fmul_rn((float)y_weights[y * 4 + 3], 2.384185791015625e-7f));
    value = __fmaf_rn((float)horizontal[2],
        __fmul_rn((float)y_weights[y * 4 + 2], 2.384185791015625e-7f), value);
    value = __fmaf_rn((float)horizontal[1],
        __fmul_rn((float)y_weights[y * 4 + 1], 2.384185791015625e-7f), value);
    value = __fmaf_rn((float)horizontal[0],
        __fmul_rn((float)y_weights[y * 4 + 0], 2.384185791015625e-7f), value);
    int quantized = max(0, min(255, __float2int_rn(value)));
    dst[i] = normalize_pixel(quantized, channel, mean, stddev);
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


class ImageGpuPreprocessor:
    """Per-model benchmark state. Calls enqueue on CuPy's current stream.

    Use one instance from one CUDA stream, without concurrent calls. The caller
    owns synchronization and measurement. CUDA failures propagate; there is no
    automatic CPU fallback. First calls include kernel compilation and coefficient
    setup, so benchmark warmup must exercise each image shape.
    """

    def __init__(self, alias, cp_module=None, opencv_rounding="scalar"):
        if alias not in SPECS:
            raise ValueError(f"unknown image benchmark model: {alias!r}")
        if opencv_rounding not in ("scalar", "neon"):
            raise ValueError("OpenCV benchmark rounding must be 'scalar' or 'neon'")
        self.opencv_rounding = opencv_rounding
        self.alias = alias
        self.spec = SPECS[alias]
        self.mean = np.asarray(self.spec["mean"], dtype=np.float32)
        self.std = np.asarray(self.spec["std"], dtype=np.float32)
        self._cp = cp_module
        self._kernels = {}
        self._device_state = {}
        self._coefficient_cache = {}

    @staticmethod
    def _rgb(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("decoded image must be a NumPy uint8 RGB array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("decoded image must have nonempty shape [height, width, 3]")
        return np.ascontiguousarray(rgb)

    def decode(self, encoded_bytes):
        if not isinstance(encoded_bytes, bytes) or not encoded_bytes:
            raise ValueError("encoded image must be nonempty bytes")
        try:
            if self.spec["decoder"] == "pillow":
                with Image.open(io.BytesIO(encoded_bytes)) as decoded:
                    rgb = np.asarray(decoded.convert("RGB"))
            else:
                bgr = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.size == 0:
                    raise ValueError("decoder returned no pixels")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(rgb)
        except Exception as exc:
            raise ValueError("unable to decode image bytes") from exc

    def cpu_resize(self, rgb):
        rgb = self._rgb(rgb)
        if self.spec["decoder"] == "pillow":
            return np.asarray(Image.fromarray(rgb).resize(self.spec["size"], Image.Resampling.BICUBIC))
        return cv2.resize(rgb, self.spec["size"], interpolation=cv2.INTER_CUBIC)

    def cpu_tensor(self, rgb):
        pixels = self.cpu_resize(rgb).astype(np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)

    def _cuda(self):
        if self._cp is None:
            import cupy
            self._cp = cupy
        return self._cp

    def _kernel(self, name):
        if name not in self._kernels:
            self._kernels[name] = self._cuda().RawKernel(_CUDA_SOURCE, name, options=("--fmad=false",))
        return self._kernels[name]

    def _constants(self):
        cp = self._cuda()
        device = cp.cuda.runtime.getDevice()
        if device not in self._device_state:
            mean, std = cp.asarray(self.mean), cp.asarray(self.std)
            ready = cp.cuda.Event(disable_timing=True)
            ready.record()
            self._device_state[device] = mean, std, ready
        mean, std, ready = self._device_state[device]
        cp.cuda.get_current_stream().wait_event(ready)
        return mean, std

    def _coefficients(self, source_shape):
        cp = self._cuda()
        height, width = source_shape
        target_width, target_height = self.spec["size"]
        key = cp.cuda.runtime.getDevice(), height, width
        if key not in self._coefficient_cache:
            prepare = _pillow_coefficients if self.spec["decoder"] == "pillow" else _opencv_coefficients
            xi, xw = prepare(width, target_width)
            yi, yw = prepare(height, target_height)
            arrays = tuple(cp.asarray(value) for value in (xi, xw, yi, yw))
            ready = cp.cuda.Event(disable_timing=True)
            ready.record()
            # Keep cached device tables bounded for long-running image sweeps.
            if len(self._coefficient_cache) >= 64:
                self._coefficient_cache.pop(next(iter(self._coefficient_cache)))
            self._coefficient_cache[key] = arrays, xw.shape[1], yw.shape[1], ready
        arrays, x_size, y_size, ready = self._coefficient_cache[key]
        cp.cuda.get_current_stream().wait_event(ready)
        return (*arrays, x_size, y_size)

    def gpu_tensor(self, rgb, mode="resize"):
        if mode not in ("normalize", "resize"):
            raise ValueError("GPU benchmark mode must be 'normalize' or 'resize'")
        rgb = self._rgb(rgb)
        cp = self._cuda()
        width, height = self.spec["size"]
        pixels = width * height
        output = cp.empty((3, height, width), dtype=cp.float32)
        mean, std = self._constants()
        grid = ((pixels * 3 + 255) // 256,)
        if mode == "normalize":
            resized = cp.asarray(self.cpu_resize(rgb))
            self._kernel("normalize_rgb")(grid, (256,), (resized, output, np.int32(pixels), mean, std))
            return output
        source_height, source_width = rgb.shape[:2]
        source = cp.asarray(rgb)
        if (source_width, source_height) == (width, height):
            self._kernel("normalize_rgb")(grid, (256,), (source, output, np.int32(pixels), mean, std))
            return output
        xi, xw, yi, yw, x_size, y_size = self._coefficients(rgb.shape[:2])
        if self.spec["decoder"] == "opencv":
            kernel = "opencv_cubic_neon" if self.opencv_rounding == "neon" else "opencv_cubic"
            self._kernel(kernel)(grid, (256,), (source, output,
                np.int32(source_width), np.int32(width), np.int32(height),
                xi, xw, yi, yw, mean, std))
        else:
            # Pillow >=12.2 avoids a huge intermediate for very tall images by
            # resampling vertically first. Retain its order of uint8 clipping.
            pillow_version = tuple(int(part) for part in PILLOW_VERSION.split(".")[:2])
            if pillow_version >= (12, 2) and source_height > source_width * 100 and height < source_height:
                transposed = cp.ascontiguousarray(source.transpose(1, 0, 2))
                vertical_t = cp.empty((source_width, height, 3), dtype=cp.uint8)
                v_grid = ((source_width * height * 3 + 255) // 256,)
                self._kernel("pillow_horizontal")(v_grid, (256,), (transposed, vertical_t,
                    np.int32(source_height), np.int32(height), np.int32(source_width),
                    yi, yw, np.int32(y_size)))
                vertical = cp.ascontiguousarray(vertical_t.transpose(1, 0, 2))
                resized = cp.empty((height, width, 3), dtype=cp.uint8)
                self._kernel("pillow_horizontal")(grid, (256,), (vertical, resized,
                    np.int32(source_width), np.int32(width), np.int32(height),
                    xi, xw, np.int32(x_size)))
                self._kernel("normalize_rgb")(grid, (256,), (resized, output,
                    np.int32(pixels), mean, std))
                return output
            # Pillow skips an unchanged axis; do not re-quantize it.
            if source_width != width:
                temporary = cp.empty((source_height, width, 3), dtype=cp.uint8)
                h_grid = ((source_height * width * 3 + 255) // 256,)
                self._kernel("pillow_horizontal")(h_grid, (256,), (source, temporary,
                    np.int32(source_width), np.int32(width), np.int32(source_height),
                    xi, xw, np.int32(x_size)))
            else:
                temporary = source
            if source_height != height:
                self._kernel("pillow_vertical_normalize")(grid, (256,), (temporary,
                    output, np.int32(width), np.int32(height), yi, yw,
                    np.int32(y_size), mean, std))
            else:
                self._kernel("normalize_rgb")(grid, (256,), (temporary, output,
                    np.int32(pixels), mean, std))
        return output


def parity_metrics(reference, candidate):
    """Host-array comparison; caller explicitly transfers CUDA output first."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("parity arrays must have identical nonempty shape")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("parity arrays must contain only finite values")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
            "different_elements": int(np.count_nonzero(difference)),
            "elements": int(difference.size)}
