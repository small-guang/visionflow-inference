"""Explicit Orin face preprocessing; CPU geometry, CUDA pixels, host bridge.

The uint8 warp uses OpenCV INTER_LINEAR's 10-bit affine coordinates and
5-bit interpolation phases. Small CPU coordinate tables avoid per-pixel
double arithmetic on Orin. No image pixels are warped or normalized on CPU.
Triton responses cross a host bridge because this Jetson runtime cannot share
CUDA buffers between the Python stub and the server through CUDA IPC.
"""
import uuid

import cv2
import numpy as np


MODES = {
    "keypoint": (192, "RAW_IMAGE", "BBOX", "PREPROCESSED_IMAGE", 0),
    "qa": (112, "RAW_FACE_IMAGE", "LANDMARK", "QA_PREPROCESSED_IMAGE", 1),
    "recognition": (112, "raw_image", "landmark", "processed_input", 2),
}
MAX_BATCH_SIZE = 4
MAX_PACK_BYTES = 4 * 8_388_608 * 3


WARP_SOURCE = r'''
extern "C" __global__ void face_warp(
    const unsigned char* src, const int* columns, const int* rows, float* out,
    int batches, int height, int width, int size, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = size * size;
    if (i >= batches * 3 * plane) return;
    int x = i % size;
    int y = (i / size) % size;
    int c = 2 - (i / plane) % 3;
    int batch = i / (3 * plane);
    const int* col = columns + (batch * size + x) * 2;
    const int* row = rows + (batch * size + y) * 2;
    int fixed_x = (col[0] + row[0]) >> 5;
    int fixed_y = (col[1] + row[1]) >> 5;
    int sx = fixed_x >> 5;
    int sy = fixed_y >> 5;
    int fx = fixed_x & 31;
    int fy = fixed_y & 31;
    int value = 0;
    long long base = (long long)batch * height * width * 3;
    for (int dy = 0; dy < 2; ++dy) {
        int yy = sy + dy;
        if (yy < 0 || yy >= height) continue;
        int wy = dy ? fy : 32 - fy;
        for (int dx = 0; dx < 2; ++dx) {
            int xx = sx + dx;
            if (xx < 0 || xx >= width) continue;
            int wx = dx ? fx : 32 - fx;
            value += src[base + ((long long)yy * width + xx) * 3 + c] * wx * wy;
        }
    }
    float pixel = (float)((value + 512) >> 10);
    if (mode == 1) {
        // QA's NumPy reference divides before subtraction. Do not rewrite
        // this as (pixel - 127.5) / 127.5: its FP32 rounding differs.
        pixel = __fdiv_rn(__fsub_rn(__fdiv_rn(pixel, 255.f), .5f), .5f);
    } else if (mode == 2) {
        pixel = __fmul_rn(__fsub_rn(pixel, 127.5f), (float)(1.0 / 127.5));
    }
    // The keypoint graph owns its normalization: preserve RGB 0..255.
    out[i] = pixel;
}
'''


def affine_tables(matrices, size):
    """Build only O(batch * output side) CPU geometry, preserving FP64 affine."""
    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 3):
        raise ValueError("affine matrices must have shape [B,2,3]")
    if matrices.dtype != np.float64 or not np.isfinite(matrices).all():
        raise ValueError("affine matrices must be finite float64")
    coordinates = np.arange(size, dtype=np.float64)
    columns, rows = [], []
    for matrix in matrices:
        # warpAffine inverts with native double arithmetic. OpenCV's separate
        # invertAffineTransform uses softdouble, whose unfused translation
        # arithmetic can select another phase for rotated faces. Keep the
        # small inverse and translation in OpenCV's native double operators.
        _, inverse_linear = cv2.invert(matrix[:, :2])
        # One input channel avoids transform's diagonal shortcut, which
        # discards off-diagonal terms <= DBL_EPSILON even at phase boundaries.
        translation_terms = np.column_stack((
            -inverse_linear[:, 0], -inverse_linear[:, 1] * matrix[1, 2]))
        inverse_translation = cv2.transform(
            matrix[:1, 2].reshape(1, 1, 1), translation_terms)[0, 0]
        inverse = np.column_stack((inverse_linear, inverse_translation))
        column = np.stack((inverse[0, 0] * coordinates,
                           inverse[1, 0] * coordinates), axis=1)
        # Use OpenCV's native FP64 multiply-add for the O(size) geometry.
        # NumPy's separate multiply/add rounds twice; on Orin this can cross
        # a half-integer before cvRound and select a different 1/32-pixel
        # interpolation phase from warpAffine. No image pixels run on CPU.
        row = cv2.transform(coordinates.reshape(-1, 1, 1), inverse[:, 1:])[:, 0, :]
        # cvRound on the supported Linux/Orin path uses ties-to-even.
        columns.append(np.rint(column * 1024.0).astype(np.int32))
        rows.append(np.rint(row * 1024.0).astype(np.int32) + 16)
    return np.ascontiguousarray(columns), np.ascontiguousarray(rows)


class FaceGpuPreprocessor:
    def __init__(self, device_id, mode):
        if mode not in MODES:
            raise ValueError(f"unsupported face preprocessing mode: {mode}")
        import cupy as cp
        self.cp = cp
        self.mode = mode
        self.size, self.image_name, self.aux_name, self.output_name, self.operation = MODES[mode]
        self.device = cp.cuda.Device(int(device_id))
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.kernel = cp.RawKernel(WARP_SOURCE, "face_warp", options=("--fmad=false",))
            self.kernel.compile()

    def warp(self, images, matrices):
        """One batched fused warp. Returns owned contiguous device FP32 NCHW."""
        if (not isinstance(images, np.ndarray) or images.dtype != np.uint8
                or images.ndim != 4 or images.shape[3] != 3
                or not 1 <= images.shape[0] <= MAX_BATCH_SIZE
                or min(images.shape[1:3]) < 1):
            raise ValueError("face warp requires a non-empty uint8 [B,H,W,3] batch with B <= 4")
        columns, rows = affine_tables(matrices, self.size)
        if len(columns) != images.shape[0]:
            raise ValueError("image and affine matrix batch sizes must match")
        with self.device, self.stream:
            # Keep host/device inputs alive until work completes, including
            # exceptions. The caller chooses when to materialize host outputs.
            try:
                source = self.cp.asarray(images, order="C")
                device_columns = self.cp.asarray(columns)
                device_rows = self.cp.asarray(rows)
                output = self.cp.empty((images.shape[0], 3, self.size, self.size), dtype=self.cp.float32)
                self.kernel(((output.size + 255) // 256,), (256,), (
                    source, device_columns, device_rows, output,
                    np.int32(images.shape[0]), np.int32(images.shape[1]),
                    np.int32(images.shape[2]), np.int32(self.size), np.int32(self.operation),
                ))
                return output
            finally:
                self.stream.synchronize()

    def execute(self, requests, pb, prepare_affine, validate_batch, invalid, logger):
        """Validate independently, combine compatible requests, preserve order."""
        responses = [None] * len(requests)
        groups = []
        pending = {}
        for request_index, request in enumerate(requests):
            batch_index = None
            try:
                image_tensor = pb.get_input_tensor_by_name(request, self.image_name)
                aux_tensor = pb.get_input_tensor_by_name(request, self.aux_name)
                if image_tensor is None:
                    raise invalid(f"missing required input {self.image_name}")
                if aux_tensor is None:
                    raise invalid(f"missing required input {self.aux_name}")
                images, aux = image_tensor.as_numpy(), aux_tensor.as_numpy()
                validate_batch(images, aux)
                if images.shape[0] > MAX_BATCH_SIZE:
                    raise invalid(f"input batch exceeds maximum {MAX_BATCH_SIZE}")
                matrices = []
                for batch_index, (image, sample_aux) in enumerate(zip(images, aux)):
                    try:
                        matrices.append(prepare_affine(image, sample_aux))
                    except invalid as exc:
                        raise invalid(f"batch index {batch_index}: {exc}") from exc
                matrices = np.ascontiguousarray(matrices, dtype=np.float64)
                key = images.shape[1:]
                group = pending.get(key)
                if (group is None or sum(item[1].shape[0] for item in group) + images.shape[0] > MAX_BATCH_SIZE
                        or sum(item[1].nbytes for item in group) + images.nbytes > MAX_PACK_BYTES):
                    group = []
                    groups.append(group)
                    pending[key] = group
                group.append((request_index, images, matrices))
            except invalid as exc:
                responses[request_index] = pb.InferenceResponse(output_tensors=[], error=pb.TritonError(f"INVALID_INPUT: {exc}"))
            except Exception:
                responses[request_index] = self._internal_error(pb, logger, request_index, batch_index)

        for group in groups:
            try:
                images = group[0][1] if len(group) == 1 else np.concatenate([item[1] for item in group])
                matrices = group[0][2] if len(group) == 1 else np.concatenate([item[2] for item in group])
                values = self.warp(images, matrices)
                # Materialize once per fused batch. The GPU pixel algorithm is
                # unchanged; only the Python-stub/server transport uses CPU
                # memory because CUDA IPC is unsupported by this Orin runtime.
                with self.device:
                    host_values = self.cp.asnumpy(values)
                offset = 0
                for request_index, images, request_matrices in group:
                    stop = offset + images.shape[0]
                    tensor = pb.Tensor(self.output_name, host_values[offset:stop])
                    tensors = [tensor]
                    if self.mode == "keypoint":
                        tensors.append(pb.Tensor("AFFINE_MATRIX", request_matrices))
                    responses[request_index] = pb.InferenceResponse(output_tensors=tensors)
                    offset = stop
            except Exception:
                # GPU failure is observable; never silently switch algorithms.
                for request_index, _, _ in group:
                    responses[request_index] = self._internal_error(pb, logger, request_index, None)
        return responses

    def _internal_error(self, pb, logger, request_index, batch_index):
        incident = uuid.uuid4().hex[:12]
        logger.exception("%s internal GPU failure request_index=%d batch_index=%s incident=%s",
                         logger.name, request_index, batch_index, incident)
        return pb.InferenceResponse(output_tensors=[], error=pb.TritonError(
            f"INTERNAL_ERROR: {logger.name} failed; incident={incident}"))
