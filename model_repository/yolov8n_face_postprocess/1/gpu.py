"""CuPy face box filtering and greedy CUDA NMS."""
import uuid

import cupy as cp
import numpy as np


NMS_SOURCE = r'''
extern "C" __global__ void prepare_faces(
    const float* raw, const float* meta, int n, int batch_size, float threshold,
    float* boxes, float* scores) {
    const float negative_infinity = __int_as_float((int)0xff800000u);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * n) return;
    int batch = index / n;
    int row = index % n;
    const float* p = raw + batch * 5 * n + row;
    const float* m = meta + batch * 5;
    float* b = boxes + index * 4;
    b[0] = fminf(fmaxf(((p[0]-p[2*n]/2.f)-m[3])/m[2], 0.f), m[1]);
    b[1] = fminf(fmaxf(((p[n]-p[3*n]/2.f)-m[4])/m[2], 0.f), m[0]);
    b[2] = fminf(fmaxf(((p[0]+p[2*n]/2.f)-m[3])/m[2], 0.f), m[1]);
    b[3] = fminf(fmaxf(((p[n]+p[3*n]/2.f)-m[4])/m[2], 0.f), m[0]);
    scores[index] = p[4*n] >= threshold && b[2] > b[0] && b[3] > b[1] ? p[4*n] : negative_infinity;
}

extern "C" __global__ void greedy_nms(
    const float* all_boxes, const float* all_scores, const long long* all_order,
    const int* lengths, int stride, float threshold,
    int limit, unsigned char* all_suppressed, float* all_output, int* counts) {
    __shared__ int picked;
    __shared__ int kept;
    int batch = blockIdx.x;
    int n = lengths[batch];
    const float* boxes = all_boxes + batch * stride * 4;
    const float* scores = all_scores + batch * stride;
    const long long* order = all_order + batch * stride;
    unsigned char* suppressed = all_suppressed + batch * stride;
    float* output = all_output + batch * limit * 6;
    if (threadIdx.x == 0) kept = 0;
    for (int j = threadIdx.x; j < limit * 6; j += blockDim.x) output[j] = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) suppressed[j] = 0;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        if (threadIdx.x == 0) {
            picked = !suppressed[i] && kept < limit;
            if (picked) {
                const float* box = boxes + order[i] * 4;
                float* detection = output + kept++ * 6;
                for (int k = 0; k < 4; ++k) detection[k] = box[k];
                detection[4] = scores[order[i]];
            }
        }
        __syncthreads();
        if (kept >= limit) break;
        if (picked) {
            const float* a = boxes + order[i] * 4;
            const float area_a = fmaxf(0.f, a[2]-a[0]) * fmaxf(0.f, a[3]-a[1]);
            for (int j = i + 1 + threadIdx.x; j < n; j += blockDim.x) {
                const float* b = boxes + order[j] * 4;
                const float area_b = fmaxf(0.f, b[2]-b[0]) * fmaxf(0.f, b[3]-b[1]);
                const float intersection = fmaxf(0.f, fminf(a[2],b[2])-fmaxf(a[0],b[0]))
                                         * fmaxf(0.f, fminf(a[3],b[3])-fmaxf(a[1],b[1]));
                const float iou = intersection / fmaxf(area_a + area_b - intersection, 1e-7f);
                if (iou > threshold) suppressed[j] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[batch] = kept;
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
        self.prepare = cp.RawKernel(NMS_SOURCE, "prepare_faces", options=("--fmad=false",))
        self.nms = cp.RawKernel(NMS_SOURCE, "greedy_nms", options=("--fmad=false",))
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
        if raw.ndim != 3 or raw.shape[1] != 5 or raw.shape[2] == 5:
            raise self.error("RAW_OUTPUT batch must have unambiguous shape [B,5,N]")
        if raw.dtype != cp.float32:
            raise self.error("RAW_OUTPUT must have dtype float32")
        if raw.shape[0] == 0:
            raise self.error("input batch must be non-empty")
        if meta.ndim != 2 or meta.shape[1] != 5:
            raise self.error("LETTERBOX_META batch must have shape [B,5]")
        if meta.dtype != np.float32:
            raise self.error("LETTERBOX_META must have dtype float32")
        if raw.shape[0] != meta.shape[0]:
            raise self.error("RAW_OUTPUT and LETTERBOX_META batch size must match")
        for index in range(raw.shape[0]):
            try:
                self.api["_validate_meta"](meta[index])
                if raw.shape[2] > self.api["MAX_PREDICTIONS"]:
                    raise self.error(f"RAW_OUTPUT exceeds max predictions {self.api['MAX_PREDICTIONS']}")
                if raw[index].size > self.api["MAX_RAW_VALUES"]:
                    raise self.error(f"RAW_OUTPUT exceeds max total values {self.api['MAX_RAW_VALUES']}")
                array_module = np if isinstance(raw, np.ndarray) else cp
                if not bool(array_module.isfinite(raw[index]).all()):
                    raise self.error("RAW_OUTPUT values must be finite")
            except self.error as exc:
                raise self.error(f"batch index {index}: {exc}") from exc
        # CPU tensors remain views until their logical sizes and values are checked.
        raw = cp.ascontiguousarray(cp.asarray(raw))
        batch_size, slots = raw.shape[0], self.model.max_detections
        predictions = raw.shape[2]
        detections = cp.empty((batch_size, slots, 6), dtype=cp.float32)
        counts = cp.empty((batch_size, 1), dtype=cp.int32)
        try:
            boxes = cp.empty((batch_size, predictions, 4), dtype=cp.float32)
            scores = cp.empty((batch_size, predictions), dtype=cp.float32)
            self.prepare((max(1, (batch_size * predictions + 255) // 256),), (256,), (
                raw, cp.ascontiguousarray(cp.asarray(meta)), np.int32(predictions), np.int32(batch_size),
                np.float32(self.model.conf_threshold), boxes, scores,
            ))
            # Preserve NumPy's original quicksort/reverse tie ordering. Only a
            # score table crosses to the CPU, once for the complete request.
            host_scores = cp.asnumpy(scores)
            order = np.zeros((batch_size, predictions), dtype=np.int64)
            lengths = np.empty((batch_size,), dtype=np.int32)
            for index in range(batch_size):
                valid = np.flatnonzero(np.isfinite(host_scores[index]))
                lengths[index] = valid.size
                order[index, :valid.size] = valid[host_scores[index, valid].argsort()[::-1]]
            suppressed = cp.empty((batch_size, predictions), dtype=cp.uint8)
            self.nms((batch_size,), (256,), (
                boxes, scores, cp.asarray(order), cp.asarray(lengths), np.int32(predictions),
                np.float32(self.model.iou_threshold), np.int32(slots), suppressed,
                detections, counts,
            ))
        except Exception as exc:
            raise SampleError(0) from exc
        return detections, counts

    def execute(self, request, pb_utils, request_index=0):
        with self.device:
            raw = self._input(request, "RAW_OUTPUT", pb_utils)
            meta = self._input(request, "LETTERBOX_META", pb_utils, host=True)
            try:
                values = self._process(raw, meta)
            except SampleError as exc:
                incident = uuid.uuid4().hex[:12]
                self.api["logger"].exception(
                    "yolov8n_face_postprocess internal failure request_index=%d batch_index=%d incident=%s",
                    request_index, exc.batch_index, incident,
                )
                return pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(
                    f"INTERNAL_ERROR: yolov8n_face_postprocess failed; incident={incident}"))
            cp.cuda.get_current_stream().synchronize()
            if self.host_outputs:
                return pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor(name, cp.asnumpy(value))
                    for name, value in zip(("DETECTIONS", "NUM_DETECTIONS"), values)
                ])
            return pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack(name, cp.ascontiguousarray(value).toDlpack())
                for name, value in zip(("DETECTIONS", "NUM_DETECTIONS"), values)
            ])
