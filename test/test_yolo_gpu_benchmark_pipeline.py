"""Benchmark metadata must support both ensembles and in-process pipelines."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


class PipelineBenchmarkMetadataTests(unittest.TestCase):
    def load(self, ready):
        path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_yolo_gpu.py"
        spec = importlib.util.spec_from_file_location("pipeline_benchmark", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        clients = []

        class Client:
            def __init__(self, **kwargs):
                self.config_names = []
                clients.append(self)

            def is_server_ready(self):
                return True

            def is_model_ready(self, name):
                return name in ready

            def get_server_metadata(self):
                return {"name": "triton"}

            def get_model_config(self, name):
                if name not in ready:
                    raise RuntimeError("model is not loaded: " + name)
                self.config_names.append(name)
                return {"name": name}

            def close(self):
                pass

        http = types.ModuleType("tritonclient.http")
        http.InferenceServerClient = Client
        package = types.ModuleType("tritonclient")
        package.http = http
        return module, {"tritonclient": package, "tritonclient.http": http}, clients

    def test_pipeline_collects_loaded_public_config_without_loading_unused_children(self):
        public = "yolo26m_pose_ensemble"
        module, modules, clients = self.load({public})
        with mock.patch.dict(sys.modules, modules):
            runner = module.HttpInference(("localhost:18000",), [public])
        self.assertEqual(clients[0].config_names, [public])
        self.assertEqual(set(runner.model_configurations["localhost:18000"]), {public})

    def test_legacy_ensemble_still_records_all_four_loaded_configs(self):
        public = "yolo26m_pose_ensemble"
        names = {"yolo26m_pose_" + suffix for suffix in
                 ("ensemble", "preprocess", "onnx", "postprocess")}
        module, modules, clients = self.load(names)
        with mock.patch.dict(sys.modules, modules):
            runner = module.HttpInference(("localhost:18010",), [public])
        self.assertEqual(set(clients[0].config_names), names)
        self.assertEqual(set(runner.model_configurations["localhost:18010"]), names)

    def test_missing_public_model_still_fails(self):
        module, modules, _ = self.load(set())
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "model is not ready"):
                module.HttpInference(("localhost:18000",), ["yolo26m_pose_ensemble"])


if __name__ == "__main__":
    unittest.main()
