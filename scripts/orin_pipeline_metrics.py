"""融合链路的有界计数器；阶段耗时是主机 wall time，不是 CUDA kernel time。"""
import os


STAGES = ('decode', 'prepare', 'preprocess', 'warp', 'trt_download', 'postprocess')
DIRECTIONS = ('input_h2d', 'output_d2h')


class NullMetrics:
    def stage(self, name, seconds): pass
    def requests(self, count): pass
    def batch(self, images, requests): pass
    def error(self, kind, count=1): pass
    def copy(self, direction, size): pass
    def close(self): pass


class PipelineMetrics:
    def __init__(self, pb, args, mode):
        labels = {'model': args['model_name'], 'version': str(args.get('model_version', '1')),
                  'instance': args.get('model_instance_name') or str(os.getpid()), 'mode': mode}
        self.families, self.metrics = {}, {}

        def counter(key, name, description, extra=None):
            if name not in self.families:
                self.families[name] = pb.MetricFamily(
                    name=name, description=description, kind=pb.MetricFamily.COUNTER)
            self.metrics[key] = self.families[name].Metric(labels={**labels, **(extra or {})})

        counter('requests', 'orin_pipeline_requests_total', 'Requests received by the fused Python model')
        counter('cross_groups', 'orin_pipeline_cross_request_groups_total',
                'TensorRT execution groups containing more than one request')
        for size in range(1, 5):
            counter(('batch', size), 'orin_pipeline_trt_batches_total',
                    'Successfully enqueued TensorRT batches', {'batch_size': str(size)})
            counter(('group_requests', size), 'orin_pipeline_group_requests_total',
                    'Requests included in successfully enqueued TensorRT batches', {'batch_size': str(size)})
        for stage in STAGES:
            counter(('seconds', stage), 'orin_pipeline_stage_seconds_total',
                    'Host wall seconds per stage, including synchronization where required', {'stage': stage})
            counter(('calls', stage), 'orin_pipeline_stage_calls_total',
                    'Measured stage calls; stages may be per request or execution group', {'stage': stage})
        for kind in ('invalid', 'internal', 'cancelled'):
            counter(('errors', kind), 'orin_pipeline_errors_total', 'Failed requests', {'kind': kind})
        for direction in DIRECTIONS:
            counter(('copy', direction), 'orin_pipeline_copy_bytes_total',
                    'Logical raw-image H2D or final-model-output D2H bytes; excludes metadata and driver traffic',
                    {'direction': direction})

    def stage(self, name, seconds):
        self.metrics[('seconds', name)].increment(seconds)
        self.metrics[('calls', name)].increment(1)

    def requests(self, count):
        self.metrics['requests'].increment(count)

    def batch(self, images, requests):
        self.metrics[('batch', images)].increment(1)
        self.metrics[('group_requests', images)].increment(requests)
        if requests > 1:
            self.metrics['cross_groups'].increment(1)

    def error(self, kind, count=1):
        self.metrics[('errors', kind)].increment(count)

    def copy(self, direction, size):
        self.metrics[('copy', direction)].increment(size)

    def close(self):
        # Python backend requires Metric objects to die before their family.
        self.metrics.clear()
        self.families.clear()
