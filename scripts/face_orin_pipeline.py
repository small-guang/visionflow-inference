"""Orin 人脸仿射 CUDA 与 TensorRT 同进程执行，仅回传最终模型结果。"""
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from time import perf_counter
import uuid

import numpy as np

from orin_pipeline_metrics import NullMetrics, PipelineMetrics


logger = logging.getLogger('face_orin_pipeline')
SPECS = {
    'face_keypoint_2d106_ensemble': {
        'mode': 'keypoint', 'pre': 'face_keypoint_2d106_pre', 'post': 'face_keypoint_2d106_post',
        'native': 'face_keypoint_2d106', 'input': 'data', 'output': 'fc1',
        'size': 192, 'raw_shape': (212,), 'public_output': 'LANDMARKS_106', 'public_shape': (106, 2),
        'image': 'RAW_IMAGE', 'aux': 'BBOX', 'aux_shape': (4,), 'api_version': '1',
    },
    'face_qa_ensemble': {
        'mode': 'qa', 'pre': 'face_qa_pre', 'post': 'face_qa_postprocess',
        'native': 'face_qa', 'input': 'input', 'output': 'quality',
        'size': 112, 'raw_shape': (10,), 'public_output': 'quality_score', 'public_shape': (1,),
        'image': 'RAW_FACE_IMAGE', 'aux': 'LANDMARK', 'aux_shape': (5, 2), 'api_version': '2',
    },
    'face_recognition_pipeline': {
        'mode': 'recognition', 'pre': 'face_rec_pre', 'post': None,
        'native': 'w600k_r50_trt', 'input': 'input.1', 'output': '683',
        'size': 112, 'raw_shape': (512,), 'public_output': 'EMBEDDING', 'public_shape': (512,),
        'image': 'RAW_IMAGE', 'aux': 'LANDMARK', 'aux_shape': (5, 2), 'api_version': '1',
    },
}
MAX_BATCH_SIZE = 4
MAX_PACK_BYTES = 4 * 8_388_608 * 3


def load_helper(root, name, version):
    path = root / name / version / 'model.py'
    module_name = 'orin_fused_' + name + '_' + version
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import face helper: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_config(args):
    name = args.get('model_name')
    if name not in SPECS:
        raise ValueError(f'unsupported fused face model: {name!r}')
    config = args.get('model_config', {})
    if isinstance(config, str):
        config = json.loads(config)
    spec = SPECS[name]
    parameters = config.get('parameters', {})
    if (args.get('model_instance_kind') != 'GPU' or config.get('max_batch_size') != 4
            or parameters.get('pipeline_backend', {}).get('string_value') != 'orin_gpu'
            or parameters.get('api_contract_version', {}).get('string_value') != spec['api_version']):
        raise ValueError('fused face requires KIND_GPU, B4, orin_gpu and the original API version')
    expected = {
        'input': {spec['image']: ('TYPE_UINT8', (-1, -1, 3)),
                  spec['aux']: ('TYPE_FP32', spec['aux_shape'])},
        'output': {spec['public_output']: ('TYPE_FP32', spec['public_shape'])},
    }
    for key, wanted in expected.items():
        rows = config.get(key, [])
        actual = {row.get('name'): (row.get('data_type'), tuple(row.get('dims', []))) for row in rows}
        if len(rows) != len(wanted) or actual != wanted:
            raise ValueError(f'fused face {key} violates the public contract')
    return spec


class FaceOrinPipeline:
    def initialize(self, args):
        self.spec = validate_config(args)
        self.name, self.mode = args['model_name'], self.spec['mode']
        version = str(args.get('model_version', '1'))
        if not re.fullmatch(r'[0-9]+', version):
            raise ValueError('model_version must be numeric')
        root = Path(args['model_repository'])
        if root.name == self.name:
            root = root.parent
        if not (root / self.name).is_dir():
            raise ValueError(f'model repository does not contain {self.name}')
        import cupy as cp
        import tensorrt as trt
        import triton_python_backend_utils as pb
        from face_orin_preprocess import FaceGpuPreprocessor
        from orin_trt_runtime import create_runner

        self.pb, self.cp, self.metrics = pb, cp, NullMetrics()
        self.pre = load_helper(root, self.spec['pre'], version)
        self.post = load_helper(root, self.spec['post'], version) if self.spec['post'] else None
        self.validate_batch = self.pre._validate_batch if self.mode == 'qa' else self.pre._validate_batch_inputs
        self.device = cp.cuda.Device(int(args.get('model_instance_device_id', 0)))
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.gpu = FaceGpuPreprocessor(self.device.id, self.mode)
            self.runner = create_runner(root, self.spec['native'], version, cp, trt,
                device_id=self.device.id, input_name=self.spec['input'],
                input_shape=(3, self.spec['size'], self.spec['size']),
                output_shapes={self.spec['output']: self.spec['raw_shape']})
        self.metrics = PipelineMetrics(pb, args, self.mode)

    def _error(self, index, kind, detail=None):
        self.metrics.error(kind)
        if kind == 'invalid':
            message = f'INVALID_INPUT: {detail}'
        elif kind == 'cancelled':
            return self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(
                'CANCELLED: request cancelled', self.pb.TritonError.CANCELLED))
        else:
            incident = uuid.uuid4().hex[:12]
            logger.exception('%s internal failure request_index=%d incident=%s', self.name, index, incident)
            message = f'INTERNAL_ERROR: {self.name} failed; incident={incident}'
        return self.pb.InferenceResponse(output_tensors=[], error=self.pb.TritonError(message))

    def _set_cancelled(self, index, request, responses):
        # 已终结的请求不再次查询或计数，也不覆盖之前的取消/错误响应。
        if responses[index] is not None:
            return True
        if not getattr(request, 'is_cancelled', lambda: False)():
            return False
        responses[index] = self._error(index, 'cancelled')
        return True

    def _prepare(self, request):
        invalid = self.pre.InputValidationError
        arrays = []
        for name in (self.spec['image'], self.spec['aux']):
            tensor = self.pb.get_input_tensor_by_name(request, name)
            if tensor is None:
                raise invalid(f'missing required input {name}')
            if not tensor.is_cpu():
                raise invalid(f'{name} requires CPU transport')
            arrays.append(tensor.as_numpy())
        images, auxiliary = arrays
        self.validate_batch(images, auxiliary)
        if images.shape[0] > MAX_BATCH_SIZE:
            raise invalid(f'input batch exceeds maximum {MAX_BATCH_SIZE}')
        matrices = []
        for index, (image, item) in enumerate(zip(images, auxiliary)):
            try:
                matrices.append(self.pre.prepare_affine(image, item))
            except invalid as exc:
                raise invalid(f'batch index {index}: {exc}') from exc
        return images, np.ascontiguousarray(matrices, dtype=np.float64)

    def _postprocess(self, values, matrices):
        if self.mode == 'keypoint':
            return np.stack([self.post.postprocess(row, matrix, input_size=192)
                             for row, matrix in zip(values, matrices)])
        if self.mode == 'qa':
            self.post._validate_quality(values)
            return np.ascontiguousarray((values.mean(axis=1, dtype=np.float32)
                                        * np.float32(10.0)).reshape(-1, 1))
        if not np.isfinite(values).all():
            raise RuntimeError('TensorRT face embedding must be finite')
        return values

    def _run_group(self, group, requests, responses):
        group = [item for item in group
                 if not self._set_cancelled(item[0], requests[item[0]], responses)]
        if not group:
            return
        same_shape = all(item[1].shape[1:] == group[0][1].shape[1:] for item in group)
        batch_size = sum(len(item[1]) for item in group)
        input_bytes = sum(item[1].nbytes for item in group)
        images = matrices = tensor = outputs = values = None
        tensors = []
        with self.device:
            try:
                started = perf_counter()
                try:
                    if same_shape:
                        images = group[0][1] if len(group) == 1 else np.concatenate([item[1] for item in group])
                        matrices = group[0][2] if len(group) == 1 else np.concatenate([item[2] for item in group])
                        tensor = self.gpu.warp(images, matrices)
                    else:
                        # Each request retains its original H/W and FP64 affine.
                        # Warp returns a completed device tensor with fixed shape.
                        # Append individually so a later warp failure cannot free
                        # an earlier result before this group's finally barrier.
                        for _, request_images, request_matrices in group:
                            tensors.append(self.gpu.warp(request_images, request_matrices))
                        with self.stream:
                            tensor = self.cp.concatenate(tensors, axis=0)
                finally:
                    self.metrics.stage('warp', perf_counter() - started)
                self.metrics.copy('input_h2d', input_bytes)
                started = perf_counter()
                try:
                    with self.stream:
                        outputs = self.runner.infer(tensor, self.stream.ptr)
                        self.metrics.batch(batch_size, len(group))
                        values = outputs[self.spec['output']].get(stream=self.stream, blocking=True)
                    self.metrics.copy('output_d2h', values.nbytes)
                finally:
                    self.metrics.stage('trt_download', perf_counter() - started)
            finally:
                # Keep raw hosts, FP64 matrices, every warp result, concatenated
                # device input and output bindings alive through native completion.
                self.stream.synchronize()
        offset = 0
        for index, request_images, request_matrices in group:
            stop = offset + len(request_images)
            request_values = values[offset:stop]
            offset = stop
            try:
                if self._set_cancelled(index, requests[index], responses):
                    continue
                started = perf_counter()
                try:
                    value = self._postprocess(request_values, request_matrices)
                finally:
                    self.metrics.stage('postprocess', perf_counter() - started)
                if not self._set_cancelled(index, requests[index], responses):
                    responses[index] = self.pb.InferenceResponse(output_tensors=[
                        self.pb.Tensor(self.spec['public_output'], value)])
            except Exception:
                if responses[index] is None:
                    responses[index] = self._error(index, 'internal')

    def execute(self, requests):
        self.metrics.requests(len(requests))
        responses = [None] * len(requests)
        groups, group = [], []
        group_samples = group_bytes = 0
        for index, request in enumerate(requests):
            try:
                if self._set_cancelled(index, request, responses):
                    continue
                started = perf_counter()
                try:
                    images, matrices = self._prepare(request)
                finally:
                    self.metrics.stage('prepare', perf_counter() - started)
                if self._set_cancelled(index, request, responses):
                    continue
                if images.nbytes > MAX_PACK_BYTES:
                    raise self.pre.InputValidationError('input request exceeds image byte budget')
                if group and (group_samples + len(images) > MAX_BATCH_SIZE
                              or group_bytes + images.nbytes > MAX_PACK_BYTES):
                    groups.append(group)
                    group = []
                    group_samples = group_bytes = 0
                group.append((index, images, matrices))
                group_samples += len(images)
                group_bytes += images.nbytes
            except self.pre.InputValidationError as exc:
                responses[index] = self._error(index, 'invalid', str(exc))
            except Exception:
                responses[index] = self._error(index, 'internal')
        if group:
            groups.append(group)
        for group in groups:
            try:
                self._run_group(group, requests, responses)
            except Exception:
                for index, _, _ in group:
                    if responses[index] is None:
                        responses[index] = self._error(index, 'internal')
        return responses

    def finalize(self):
        if hasattr(self, 'stream'):
            with self.device:
                self.stream.synchronize()
                self.runner = None
                self.gpu = None
        if hasattr(self, 'metrics'):
            self.metrics.close()
