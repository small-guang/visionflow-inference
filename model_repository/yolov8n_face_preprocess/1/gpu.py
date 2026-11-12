"""Fused CUDA letterbox with OpenCV's uint8 INTER_LINEAR arithmetic.

This file is shipped with each YOLO model so selectively packaged services do
not depend on another model's directory. Pixel buffers stay on the GPU; only
small geometry/interpolation tables and LETTERBOX_META are built on the CPU.
"""

from collections import OrderedDict

import cupy as cp
import numpy as np


class GpuInputValidationError(ValueError):
    pass


_KERNEL = r'''
extern "C" __global__ void letterbox(
    const unsigned char* src, float* dst, const int* xt, const int* yt,
    int batches, int target_h, int target_w, int resized_h, int resized_w,
    int top, int left, int padding, long long sb, long long sh,
    long long sw, long long sc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int plane = target_h * target_w;
    if (i >= batches * 3 * plane) return;
    int x = i % target_w - left;
    int y = (i / target_w) % target_h - top;
    int channel = 2 - (i / plane) % 3;
    int batch = i / (3 * plane);
    int pixel = padding;
    if (x >= 0 && x < resized_w && y >= 0 && y < resized_h) {
        const int* a = xt + x * 4;
        const int* b = yt + y * 4;
        long long offset = batch * sb + channel * sc;
        int row0 = src[offset + b[0] * sh + a[0] * sw] * a[2]
                 + src[offset + b[0] * sh + a[1] * sw] * a[3];
        int row1 = src[offset + b[1] * sh + a[0] * sw] * a[2]
                 + src[offset + b[1] * sh + a[1] * sw] * a[3];
        // OpenCV's 8-bit linear path truncates horizontal intermediates
        // before vertical fixed-point multiplication and final rounding.
        pixel = (((b[2] * (row0 >> 4)) >> 16)
               + ((b[3] * (row1 >> 4)) >> 16) + 2) >> 2;
    }
    dst[i] = __fdiv_rn((float)pixel, 255.0f);
}
'''


def _interpolation_table(source_size, target_size, *, horizontal):
    # OpenCV forms coordinates in double, then stores them in float before
    # extracting the fraction. Computing these short tables on the CPU avoids
    # per-pixel double arithmetic on Jetson and gives the same coefficients.
    coordinates = ((np.arange(target_size, dtype=np.float64) + 0.5)
                   * (source_size / target_size) - 0.5).astype(np.float32)
    index = np.floor(coordinates).astype(np.int32)
    fraction = coordinates - index.astype(np.float32)
    if horizontal:
        boundary = (index < 0) | (index >= source_size - 1)
        fraction[boundary] = 0
        index = np.clip(index, 0, source_size - 1)
    first = np.clip(index, 0, source_size - 1)
    second = np.clip(index + 1, 0, source_size - 1)
    weight0 = np.rint((np.float32(1) - fraction) * np.float32(2048)).astype(np.int32)
    weight1 = np.rint(fraction * np.float32(2048)).astype(np.int32)
    return np.stack((first, second, weight0, weight1), axis=1)


class GpuPreprocessor:
    def __init__(self, device_id, *, host_outputs=False):
        self.device = cp.cuda.Device(int(device_id))
        self.host_outputs = bool(host_outputs)
        self.to_host = cp.asnumpy
        self._tables = OrderedDict()
        with self.device:
            self._kernel = cp.RawKernel(_KERNEL, "letterbox")
            # Fail model loading clearly if the GPU/compiler is unavailable.
            self._kernel.compile()

    def from_tensor(self, tensor):
        if tensor.is_cpu():
            return tensor.as_numpy()
        with self.device:
            return cp.from_dlpack(tensor.to_dlpack())

    def preprocess(self, images, *, meta_width=5, imgsz=640, padding_value=114):
        if not isinstance(images, (np.ndarray, cp.ndarray)):
            raise GpuInputValidationError("RAW_IMAGE must be an array")
        if images.ndim != 4 or images.shape[3] != 3:
            raise GpuInputValidationError("RAW_IMAGE batch must have shape [B,H,W,3]")
        if images.dtype != np.uint8:
            raise GpuInputValidationError("RAW_IMAGE must have dtype uint8")
        batches, height, width, _ = images.shape
        if batches == 0:
            raise GpuInputValidationError("RAW_IMAGE batch must be non-empty")
        if height <= 0 or width <= 0:
            raise GpuInputValidationError("batch index 0: RAW_IMAGE height and width must be positive")
        if height > 8192 or width > 8192:
            raise GpuInputValidationError("batch index 0: RAW_IMAGE exceeds max side 8192")
        if height * width > 8_388_608:
            raise GpuInputValidationError("batch index 0: RAW_IMAGE exceeds max pixels 8388608")
        ratio = min(imgsz / height, imgsz / width)
        resized_w, resized_h = round(width * ratio), round(height * ratio)
        if resized_w < 1 or resized_h < 1:
            raise GpuInputValidationError("batch index 0: RAW_IMAGE aspect ratio produces a resize dimension smaller than 1 pixel")
        buckets = (192, 384, 480, 640)
        target_h = next(size for size in buckets if size >= resized_h)
        target_w = next(size for size in buckets if size >= resized_w)
        top = round((target_h - resized_h) / 2 - 0.1)
        left = round((target_w - resized_w) / 2 - 0.1)
        meta = [height, width, ratio, left, top]
        if meta_width == 7:
            meta.extend((target_h, target_w))
        metas = np.tile(np.asarray(meta, dtype=np.float32), (batches, 1))
        with self.device:
            if isinstance(images, np.ndarray):
                images = cp.asarray(images, order="C")
            elif images.device.id != self.device.id:
                images = cp.asarray(images)
            key = (height, width, resized_h, resized_w)
            tables = self._tables.pop(key, None)
            if tables is None:
                tables = (
                    cp.asarray(_interpolation_table(width, resized_w, horizontal=True)),
                    cp.asarray(_interpolation_table(height, resized_h, horizontal=False)),
                )
            self._tables[key] = tables
            if len(self._tables) > 16:
                self._tables.popitem(last=False)
            output = cp.empty((batches, 3, target_h, target_w), dtype=cp.float32)
            size = output.size
            self._kernel(((size + 255) // 256,), (256,), (
                images, output, *tables, np.int32(batches), np.int32(target_h),
                np.int32(target_w), np.int32(resized_h), np.int32(resized_w),
                np.int32(top), np.int32(left), np.int32(padding_value),
                *(np.int64(stride) for stride in images.strides),
            ))
            return output, metas
