"""CuPy segmentation selection, bounded mask GEMM and packed CUDA sampling."""
import uuid

import cupy as cp
import numpy as np


SELECT_SOURCE = r'''
extern "C" __global__ void segment_select(
    const float* raw, const float* geometry, int n, int channels, int slots,
    int proto_h, int proto_w, float threshold, float* detections,
    float* coefficients, float* proto_boxes, int* counts) {
    const float negative_infinity = __int_as_float((int)0xff800000u);
    __shared__ float scores[1024];
    int batch = blockIdx.x;
    int stride = 6 + channels;
    const float* input = raw + batch * n * stride;
    const float* m = geometry + batch * 7;
    float* out = detections + batch * slots * 6;
    float* coeff = coefficients + batch * slots * channels;
    float* boxes = proto_boxes + batch * slots * 4;
    for (int i = threadIdx.x; i < slots * 6; i += blockDim.x) out[i] = 0.f;
    if (threadIdx.x == 0) counts[batch] = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float* p = input + i * stride;
        float x1 = fminf(fmaxf((p[0]-m[3])/m[2], 0.f), m[1]);
        float y1 = fminf(fmaxf((p[1]-m[4])/m[2], 0.f), m[0]);
        float x2 = fminf(fmaxf((p[2]-m[3])/m[2], 0.f), m[1]);
        float y2 = fminf(fmaxf((p[3]-m[4])/m[2], 0.f), m[0]);
        bool valid = p[4] >= threshold && rintf(p[5]) == 0.f
            && x2 > x1 && y2 > y1 && rintf(x2) > rintf(x1) && rintf(y2) > rintf(y1);
        scores[i] = valid ? p[4] : negative_infinity;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float score = scores[i];
        if (score == negative_infinity) continue;
        int rank = 0;
        for (int j = 0; j < n; ++j)
            rank += scores[j] > score || (scores[j] == score && j < i);
        if (rank >= slots) continue;
        atomicAdd(counts + batch, 1);
        const float* p = input + i * stride;
        float* d = out + rank * 6;
        for (int axis = 0; axis < 4; ++axis) {
            float padding = axis % 2 ? m[4] : m[3];
            float bound = axis % 2 ? m[0] : m[1];
            float coordinate = fminf(fmaxf((p[axis]-padding)/m[2], 0.f), bound);
            d[axis] = coordinate;
            float scale = axis % 2 ? (float)proto_h/m[5] : (float)proto_w/m[6];
            boxes[rank*4+axis] = (coordinate*m[2]+padding)*scale;
        }
        d[4] = score;
        d[5] = rintf(p[5]);
        for (int c = 0; c < channels; ++c) coeff[rank*channels+c] = p[6+c];
    }
}
'''


MASK_SOURCE = r'''
extern "C" __global__ void packed_crops(
    const float* logits, const float* boxes, int count, int h, int w,
    int numpy2, unsigned char* output) {
    int byte_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (byte_index >= count * 160 * 20) return;
    int instance = byte_index / (160 * 20);
    int row = (byte_index / 20) % 160;
    int start_x = (byte_index % 20) * 8;
    const float* b = boxes + instance * 4;
    float mx, my, tx, ty;
    if (numpy2) {
        mx = (b[2]-b[0]) / 160.f;
        my = (b[3]-b[1]) / 160.f;
        tx = (b[0] + 0.5f * mx) - 0.5f;
        ty = (b[1] + 0.5f * my) - 0.5f;
    } else {
        double sx = (double)(b[2]-b[0]) / 160.0;
        double sy = (double)(b[3]-b[1]) / 160.0;
        mx = (float)sx;
        my = (float)sy;
        tx = (float)((double)b[0] + 0.5 * sx - 0.5);
        ty = (float)((double)b[1] + 0.5 * sy - 0.5);
    }
    // OpenCV warpAffine uses 10-bit deltas and a 5-bit interpolation table.
    int y_fixed = (__double2int_rn(((double)my * row + (double)ty) * 1024.0) + 16) >> 5;
    int y0 = y_fixed >> 5;
    float fy = (float)(y_fixed & 31) * (1.f / 32.f);
    int ya = min(max(y0, 0), h-1);
    int yb = min(max(y0+1, 0), h-1);
    const float* src = logits + instance * h * w;
    unsigned char packed = 0;
    for (int bit = 0; bit < 8; ++bit) {
        int x = start_x + bit;
        int x_fixed = (__double2int_rn((double)tx * 1024.0) + 16
                     + __double2int_rn((double)mx * x * 1024.0)) >> 5;
        int x0 = x_fixed >> 5;
        float fx = (float)(x_fixed & 31) * (1.f / 32.f);
        int xa = min(max(x0, 0), w-1);
        int xb = min(max(x0+1, 0), w-1);
        float w00 = (1.f-fy) * (1.f-fx);
        float w01 = (1.f-fy) * fx;
        float w10 = fy * (1.f-fx);
        float w11 = fy * fx;
        float value = src[ya*w+xa]*w00 + src[ya*w+xb]*w01
                    + src[yb*w+xa]*w10 + src[yb*w+xb]*w11;
        if (value > 0.f) packed |= (unsigned char)(1 << (7-bit));
    }
    output[byte_index] = packed;
}
'''


class SampleError(RuntimeError):
    def __init__(self, batch_index):
        self.batch_index = batch_index
        super().__init__("unexpected GPU postprocessing failure")


class GpuPostprocessor:
    def __init__(self, device_id, model, api, host_outputs=False):
        self.device = cp.cuda.Device(device_id)
        self.host_outputs = bool(host_outputs)
        self.model = model
        self.api = api
        self.error = api["InputValidationError"]
        self.select_kernel = cp.RawKernel(SELECT_SOURCE, "segment_select", options=("--fmad=false",))
        self.mask_kernel = cp.RawKernel(MASK_SOURCE, "packed_crops", options=("--fmad=false",))
        self.numpy2 = int(np.lib.NumpyVersion(np.__version__) >= "2.0.0")
        with self.device:
            cp.zeros(1)

    def _input(self, request, name, pb_utils, host=False):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise self.error(f"missing required input {name}")
        if tensor.is_cpu():
            return tensor.as_numpy()
        value = cp.from_dlpack(tensor.to_dlpack())
        return cp.asnumpy(value) if host else value

    def process(self, raw, meta, proto):
        with self.device:
            return self._process(raw, meta, proto)

    def _preflight(self, raw, proto, meta):
        if raw.ndim != 3:
            raise self.error("RAW_OUTPUT batch must have shape [B,N,6 + mask_channels]")
        if raw.dtype != cp.float32:
            raise self.error("RAW_OUTPUT must have dtype float32")
        batch_size = raw.shape[0]
        if batch_size == 0:
            raise self.error("input batch must be non-empty")
        if batch_size > self.model.max_batch_size:
            raise self.error(f"batch size {batch_size} exceeds configured maximum {self.model.max_batch_size}")
        if raw.shape[1] > self.api["MAX_PREDICTIONS"]:
            raise self.error(f"RAW_OUTPUT exceeds max predictions {self.api['MAX_PREDICTIONS']}")
        if raw.size > self.api["MAX_RAW_OUTPUT_VALUES"]:
            raise self.error(f"RAW_OUTPUT total values exceed {self.api['MAX_RAW_OUTPUT_VALUES']}")
        if proto.ndim != 4:
            raise self.error("MASK_PROTO batch must have shape [B,C,H,W]")
        if proto.dtype != cp.float32:
            raise self.error("MASK_PROTO must have dtype float32")
        if proto.size > self.api["MAX_MASK_PROTO_VALUES"]:
            raise self.error(f"MASK_PROTO total values exceed {self.api['MAX_MASK_PROTO_VALUES']}")
        if meta.ndim != 2 or meta.shape[1] != 7:
            raise self.error("LETTERBOX_META batch must have shape [B,7]")
        if meta.dtype != np.float32:
            raise self.error("LETTERBOX_META must have dtype float32")
        if proto.shape[0] != batch_size or meta.shape[0] != batch_size:
            raise self.error("RAW_OUTPUT, MASK_PROTO, and LETTERBOX_META batch size must match")
        geometries = []
        channels, height, width = proto.shape[1:]
        for index in range(batch_size):
            try:
                geometry = self.api["_validate_meta"](meta[index])
                if channels < 1 or channels > self.api["MAX_MASK_CHANNELS"]:
                    raise self.error(f"MASK_PROTO channels must be between 1 and {self.api['MAX_MASK_CHANNELS']}")
                if height < 1 or width < 1:
                    raise self.error("MASK_PROTO spatial dimensions must be positive")
                if height > self.api["MAX_PROTO_SIDE"] or width > self.api["MAX_PROTO_SIDE"]:
                    raise self.error(f"MASK_PROTO exceeds max side {self.api['MAX_PROTO_SIDE']}")
                proto_module = np if isinstance(proto, np.ndarray) else cp
                if not bool(proto_module.isfinite(proto[index]).all()):
                    raise self.error("MASK_PROTO values must be finite")
                if raw.shape[2] != 6 + channels:
                    raise self.error("YOLO segment output must use the end-to-end layout "
                                     f"[N, 6 + mask_channels] with width {6 + channels}, got {raw[index].shape}")
                raw_module = np if isinstance(raw, np.ndarray) else cp
                if not bool(raw_module.isfinite(raw[index]).all()):
                    raise self.error("RAW_OUTPUT values must be finite")
                if geometries and (geometry.original_h, geometry.original_w) != (
                    geometries[0].original_h, geometries[0].original_w
                ):
                    raise self.error("all samples in a request must have the same original height and width")
                geometries.append(geometry)
            except self.error as exc:
                raise self.error(f"batch index {index}: {exc}") from exc
        return geometries

    def _select(self, raw, geometry):
        valid_person = (raw[:, 4] >= self.model.conf_threshold) & (cp.rint(raw[:, 5]) == 0)
        selected = raw[valid_person]
        boxes = selected[:, :4].copy()
        boxes[:, [0, 2]] = cp.clip((boxes[:, [0, 2]] - geometry.left) / geometry.ratio, 0, geometry.original_w)
        boxes[:, [1, 3]] = cp.clip((boxes[:, [1, 3]] - geometry.top) / geometry.ratio, 0, geometry.original_h)
        # Python round(float32) uses ties-to-even; rint has the same rule.
        pixels = cp.rint(boxes)
        valid = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) &
                 (pixels[:, 2] > pixels[:, 0]) & (pixels[:, 3] > pixels[:, 1]))
        boxes, selected = boxes[valid], selected[valid]
        keep = cp.lexsort(cp.stack((cp.arange(selected.shape[0], dtype=cp.float32), -selected[:, 4])))[:self.model.max_detections]
        detections = cp.empty((keep.size, 6), dtype=cp.float32)
        detections[:, :4] = boxes[keep]
        detections[:, 4] = selected[keep, 4]
        detections[:, 5] = cp.rint(selected[keep, 5])
        return detections, cp.ascontiguousarray(selected[keep, 6:])

    def _process(self, raw, meta, proto):
        geometries = self._preflight(raw, proto, meta)
        # Check all request tensors before materializing even a zero-stride host view.
        raw = cp.ascontiguousarray(cp.asarray(raw))
        proto = cp.ascontiguousarray(cp.asarray(proto))
        selections = []
        batch_size = len(geometries)
        fused_boxes = None
        fused_counts = None
        if raw.shape[1] <= 1024:
            try:
                limit = self.model.max_detections
                _, channels, proto_h, proto_w = proto.shape
                selected = cp.empty((batch_size, limit, 6), dtype=cp.float32)
                coefficients = cp.empty((batch_size, limit, channels), dtype=cp.float32)
                fused_boxes = cp.empty((batch_size, limit, 4), dtype=cp.float32)
                fused_counts = cp.empty((batch_size, 1), dtype=cp.int32)
                canonical = np.array([
                    [g.original_h, g.original_w, g.ratio, g.left, g.top, g.input_h, g.input_w]
                    for g in geometries
                ], dtype=np.float32)
                self.select_kernel((batch_size,), (256,), (
                    raw, cp.asarray(canonical), np.int32(raw.shape[1]), np.int32(channels),
                    np.int32(limit), np.int32(proto_h), np.int32(proto_w),
                    np.float32(self.model.conf_threshold), selected, coefficients,
                    fused_boxes, fused_counts,
                ))
                # One batch-sized read, solely for the public variable-slot shape.
                host_counts = cp.asnumpy(fused_counts).ravel()
                selections = [(selected[index, :int(count)], coefficients[index, :int(count)])
                              for index, count in enumerate(host_counts)]
            except Exception as exc:
                raise SampleError(0) from exc
        else:
            for index, geometry in enumerate(geometries):
                try:
                    selections.append(self._select(raw[index], geometry))
                except Exception as exc:
                    raise SampleError(index) from exc
        slots = max(1, max(item[0].shape[0] for item in selections))
        if batch_size * slots * 160 * 160 > self.api["MAX_MASK_OUTPUT_VALUES"]:
            raise self.error("MASK_CROPS_PACKED logical output values exceed budget "
                             f"{self.api['MAX_MASK_OUTPUT_VALUES']}")
        detections = cp.zeros((batch_size, slots, 6), dtype=cp.float32)
        crops = cp.zeros((batch_size, slots, 160, 20), dtype=cp.uint8)
        counts = fused_counts if fused_counts is not None else cp.zeros((batch_size, 1), dtype=cp.int32)
        _, channels, proto_h, proto_w = proto.shape
        for index, (sample_detections, coefficients) in enumerate(selections):
            try:
                count = sample_detections.shape[0]
                if count:
                    detections[index, :count] = sample_detections
                    geometry = geometries[index]
                    if fused_boxes is not None:
                        boxes = fused_boxes[index, :count]
                    else:
                        boxes = sample_detections[:, :4].copy()
                        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * geometry.ratio + geometry.left) * (proto_w / geometry.input_w)
                        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * geometry.ratio + geometry.top) * (proto_h / geometry.input_h)
                    # Bound the scratch allocation even for allowed large proto shapes.
                    chunk_size = max(1, min(count, (16 * 1024 * 1024) // (4 * proto_h * proto_w)))
                    flattened = cp.ascontiguousarray(proto[index]).reshape(channels, -1)
                    for start in range(0, count, chunk_size):
                        stop = min(count, start + chunk_size)
                        logits = cp.ascontiguousarray(coefficients[start:stop] @ flattened)
                        chunk_boxes = cp.ascontiguousarray(boxes[start:stop])
                        elements = (stop - start) * 160 * 20
                        self.mask_kernel(((elements + 255) // 256,), (256,), (
                            logits, chunk_boxes, np.int32(stop-start), np.int32(proto_h), np.int32(proto_w),
                            np.int32(self.numpy2), crops[index, start:stop],
                        ))
                if fused_counts is None:
                    counts[index, 0] = count
            except Exception as exc:
                raise SampleError(index) from exc
        return detections, crops, counts

    def execute(self, request, pb_utils, request_index=0):
        with self.device:
            raw = self._input(request, "RAW_OUTPUT", pb_utils)
            proto = self._input(request, "MASK_PROTO", pb_utils)
            meta = self._input(request, "LETTERBOX_META", pb_utils, host=True)
            try:
                values = self._process(raw, meta, proto)
            except SampleError as exc:
                incident = uuid.uuid4().hex[:12]
                self.api["logger"].exception(
                    "yolo26s_seg_postprocess internal failure request_index=%d batch_index=%d incident=%s",
                    request_index, exc.batch_index, incident,
                )
                return pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(
                    f"INTERNAL_ERROR: yolo26s_seg_postprocess failed; incident={incident}"))
            cp.cuda.get_current_stream().synchronize()
            if self.host_outputs:
                return pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor(name, cp.asnumpy(value))
                    for name, value in zip(("DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"), values)
                ])
            return pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack(name, cp.ascontiguousarray(value).toDlpack())
                for name, value in zip(("DETECTIONS", "MASK_CROPS_PACKED", "NUM_DETECTIONS"), values)
            ])
