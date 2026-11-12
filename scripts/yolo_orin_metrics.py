"""Bounded-cardinality Python backend counters, independent of CUDA."""
import os


class NullMetrics:
    def trt_call(self, batch_size): pass
    def trt_failure(self): pass
    def group_failure(self): pass
    def copy(self, direction, size, seconds=0): pass
    def close(self): pass


class PipelineMetrics:
    def __init__(self, pb, args):
        labels = {"model": args["model_name"], "version": str(args.get("model_version", "1")),
                  "instance": args.get("model_instance_name") or str(os.getpid())}
        self.families, self.metrics = {}, {}

        def counter(key, name, description, extra=None):
            family = self.families.get(name)
            if family is None:
                family = pb.MetricFamily(name=name, description=description, kind=pb.MetricFamily.COUNTER)
                self.families[name] = family
            metric = family.Metric(labels={**labels, **(extra or {})})
            self.metrics[key] = metric

        for batch in range(1, 5):
            counter(("trt", batch), "yolo_orin_trt_calls_total", "Actual TensorRT enqueue attempts",
                    {"batch_size": str(batch)})
        counter("trt_failure", "yolo_orin_trt_failures_total", "Failed TensorRT enqueue attempts")
        counter("group_failure", "yolo_orin_group_failures_total", "Failed validated execution groups")
        for direction in ("host_pack", "host_split", "input_h2d", "output_d2h"):
            counter(("bytes", direction), "yolo_orin_copy_bytes_total",
                    "Logical image/output tensor bytes copied; excludes metadata and driver traffic",
                    {"direction": direction})
        for direction in ("host_pack", "host_split"):
            counter(("seconds", direction), "yolo_orin_host_copy_seconds_total",
                    "Host packing or response splitting wall time", {"direction": direction})

    def trt_call(self, batch_size):
        self.metrics[("trt", batch_size)].increment(1)

    def trt_failure(self):
        self.metrics["trt_failure"].increment(1)

    def group_failure(self):
        self.metrics["group_failure"].increment(1)

    def copy(self, direction, size, seconds=0):
        self.metrics[("bytes", direction)].increment(size)
        if direction in ("host_pack", "host_split"):
            self.metrics[("seconds", direction)].increment(seconds)

    def close(self):
        # Triton requires every Metric to die before its owning MetricFamily.
        self.metrics.clear()
        self.families.clear()
