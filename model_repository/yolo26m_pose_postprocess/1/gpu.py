"""CuPy pose postprocessing; imported only by a GPU model instance."""
import uuid

import cupy as cp
import numpy as np


POSE_SOURCE = r'''
// YOLO end-to-end plans normally emit 300 rows. A shared score table replaces
// boolean compaction, lexsort and their host size synchronizations.
extern "C" __global__ void pose_select(
    const float* raw, const float* meta, int n, int slots, float threshold,
    float* detections, float* points, int* counts) {
    const float negative_infinity = __int_as_float((int)0xff800000u);
    __shared__ float scores[1024];
    int batch = blockIdx.x;
    const float* input = raw + batch * n * 57;
    const float* m = meta + batch * 7;
    float* out = detections + batch * slots * 6;
    float* keypoints = points + batch * slots * 51;
    for (int i = threadIdx.x; i < slots * 6; i += blockDim.x) out[i] = 0.f;
    for (int i = threadIdx.x; i < slots * 51; i += blockDim.x) keypoints[i] = 0.f;
    if (threadIdx.x == 0) counts[batch] = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float* p = input + i * 57;
        float x1 = fminf(fmaxf((p[0]-m[3])/m[2], 0.f), m[1]);
        float y1 = fminf(fmaxf((p[1]-m[4])/m[2], 0.f), m[0]);
        float x2 = fminf(fmaxf((p[2]-m[3])/m[2], 0.f), m[1]);
        float y2 = fminf(fmaxf((p[3]-m[4])/m[2], 0.f), m[0]);
        scores[i] = p[4] >= threshold && x2 > x1 && y2 > y1 ? p[4] : negative_infinity;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float score = scores[i];
        if (score == negative_infinity) continue;
        int rank = 0;
        // Original index is the stable tie breaker after filtering.
        for (int j = 0; j < n; ++j)
            rank += scores[j] > score || (scores[j] == score && j < i);
        if (rank >= slots) continue;
        atomicAdd(counts + batch, 1);
        const float* p = input + i * 57;
        float* d = out + rank * 6;
        for (int axis = 0; axis < 4; ++axis) {
            float padding = axis % 2 ? m[4] : m[3];
            float bound = axis % 2 ? m[0] : m[1];
            d[axis] = fminf(fmaxf((p[axis]-padding)/m[2], 0.f), bound);
        }
        d[4] = score;
        d[5] = rintf(p[5]);
        for (int k = 0; k < 17; ++k) {
            keypoints[rank*51+k*3] = fminf(fmaxf((p[6+k*3]-m[3])/m[2], 0.f), m[1]);
            keypoints[rank*51+k*3+1] = fminf(fmaxf((p[7+k*3]-m[4])/m[2], 0.f), m[0]);
            keypoints[rank*51+k*3+2] = p[8+k*3];
        }
    }
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
        self.select_kernel = cp.RawKernel(POSE_SOURCE, "pose_select", options=("--fmad=false",))
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

    def process(self, raw, meta):
        with self.device:
            return self._process(raw, meta)

    def _process(self, raw, meta):
        if raw.ndim != 3 or raw.shape[2] != 57:
            raise self.error("RAW_OUTPUT batch must have shape [B,N,57]")
        if raw.dtype != cp.float32:
            raise self.error("RAW_OUTPUT must have dtype float32")
        if raw.shape[0] == 0:
            raise self.error("input batch must be non-empty")
        if meta.ndim != 2 or meta.shape[1] != 7:
            raise self.error("LETTERBOX_META batch must have shape [B,7]")
        if meta.dtype != np.float32:
            raise self.error("LETTERBOX_META must have dtype float32")
        if raw.shape[0] != meta.shape[0]:
            raise self.error("RAW_OUTPUT and LETTERBOX_META batch size must match")
        # Validate every sample before allocating the output batch.
        for index in range(raw.shape[0]):
            try:
                self.api["_validate_meta"](meta[index])
                if raw.shape[1] > self.api["MAX_PREDICTIONS"]:
                    raise self.error(f"RAW_OUTPUT exceeds max predictions {self.api['MAX_PREDICTIONS']}")
                array_module = np if isinstance(raw, np.ndarray) else cp
                if not bool(array_module.isfinite(raw[index]).all()):
                    raise self.error("RAW_OUTPUT values must be finite")
            except self.error as exc:
                raise self.error(f"batch index {index}: {exc}") from exc
        # CPU tensors remain views until their logical sizes and values are checked.
        raw = cp.ascontiguousarray(cp.asarray(raw))
        batch_size = raw.shape[0]
        slots = self.model.max_detections
        if raw.shape[1] <= 1024:
            detections = cp.empty((batch_size, slots, 6), dtype=cp.float32)
            keypoints = cp.empty((batch_size, slots, 17, 3), dtype=cp.float32)
            counts = cp.empty((batch_size, 1), dtype=cp.int32)
            try:
                self.select_kernel((batch_size,), (256,), (
                    raw, cp.ascontiguousarray(cp.asarray(meta)), np.int32(raw.shape[1]),
                    np.int32(slots), np.float32(self.model.conf_threshold),
                    detections, keypoints, counts,
                ))
            except Exception as exc:
                raise SampleError(0) from exc
            return detections, keypoints, counts
        detections = cp.zeros((batch_size, slots, 6), dtype=cp.float32)
        keypoints = cp.zeros((batch_size, slots, 17, 3), dtype=cp.float32)
        counts = cp.zeros((batch_size, 1), dtype=cp.int32)
        for index in range(batch_size):
            try:
                pred = raw[index]
                selected = pred[pred[:, 4] >= self.model.conf_threshold]
                if not selected.shape[0]:
                    continue
                boxes = selected[:, :4].copy()
                points = selected[:, 6:].reshape(-1, 17, 3).copy()
                height, width, ratio, left, top = meta[index, :5]
                boxes[:, [0, 2]] = cp.clip((boxes[:, [0, 2]] - left) / ratio, 0, width)
                boxes[:, [1, 3]] = cp.clip((boxes[:, [1, 3]] - top) / ratio, 0, height)
                points[:, :, 0] = cp.clip((points[:, :, 0] - left) / ratio, 0, width)
                points[:, :, 1] = cp.clip((points[:, :, 1] - top) / ratio, 0, height)
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, points, selected = boxes[valid], points[valid], selected[valid]
                keep = cp.lexsort(cp.stack((cp.arange(selected.shape[0], dtype=cp.float32), -selected[:, 4])))[:slots]
                count = keep.size
                detections[index, :count, :4] = boxes[keep]
                detections[index, :count, 4] = selected[keep, 4]
                detections[index, :count, 5] = cp.rint(selected[keep, 5])
                keypoints[index, :count] = points[keep]
                counts[index, 0] = count
            except Exception as exc:
                raise SampleError(index) from exc
        return detections, keypoints, counts

    def execute(self, request, pb_utils, request_index=0):
        with self.device:
            raw = self._input(request, "RAW_OUTPUT", pb_utils)
            meta = self._input(request, "LETTERBOX_META", pb_utils, host=True)
            try:
                values = self._process(raw, meta)
            except SampleError as exc:
                incident = uuid.uuid4().hex[:12]
                self.api["logger"].exception(
                    "yolo26m_pose_postprocess internal failure request_index=%d batch_index=%d incident=%s",
                    request_index, exc.batch_index, incident,
                )
                return pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(
                    f"INTERNAL_ERROR: yolo26m_pose_postprocess failed; incident={incident}"))
            cp.cuda.get_current_stream().synchronize()
            if self.host_outputs:
                return pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor(name, cp.asnumpy(value))
                    for name, value in zip(("DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"), values)
                ])
            return pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack(name, cp.ascontiguousarray(value).toDlpack())
                for name, value in zip(("DETECTIONS", "KEYPOINTS", "NUM_DETECTIONS"), values)
            ])
