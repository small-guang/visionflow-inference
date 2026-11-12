"""Static contracts for the Jetson AGX Orin ARM64 deployment."""

import ast
import importlib.util
import json
import re
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def load_http_client_module(module_name, relative_path):
    tritonclient = types.ModuleType("tritonclient")
    http = types.ModuleType("tritonclient.http")
    tritonclient.http = http
    previous_client = sys.modules.get("tritonclient")
    previous_http = sys.modules.get("tritonclient.http")
    sys.modules["tritonclient"] = tritonclient
    sys.modules["tritonclient.http"] = http
    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_client is None:
            sys.modules.pop("tritonclient", None)
        else:
            sys.modules["tritonclient"] = previous_client
        if previous_http is None:
            sys.modules.pop("tritonclient.http", None)
        else:
            sys.modules["tritonclient.http"] = previous_http


POSE_CLIENT_MODULE = load_http_client_module(
    "jetson_yolo26m_pose_image_client", "test/test_yolo26m_pose_image.py"
)
SEG_CLIENT_MODULE = load_http_client_module(
    "jetson_yolo26s_seg_image_client", "test/test_yolo26s_seg_image.py"
)


def service_block(compose_text, service_name):
    match = re.search(
        rf"(?ms)^  {re.escape(service_name)}:\s*\n"
        rf"(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\s*\n|\Z)",
        compose_text,
    )
    if match is None:
        raise AssertionError(f"Missing Compose service: {service_name}")
    return match.group("body")


class JetsonComposeContractTests(unittest.TestCase):
    def test_all_services_use_pinned_private_arm64_images(self):
        compose = read_text("docker-compose.jetson.yaml")
        active_services = set(
            re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", compose)
        )
        self.assertEqual(active_services, {"triton-server"})

        self.assertNotIn(":latest", compose)
        triton = service_block(compose, "triton-server")
        self.assertIn(
            "image: registry.cn-hangzhou.aliyuncs.com/eapil/"
            "tritonserver:26.04-py3-arm64-runtime",
            triton,
        )
        self.assertIn("platform: linux/arm64", triton)

        for fragment in (
            "#  prometheus:",
            "#    image: registry.cn-hangzhou.aliyuncs.com/eapil/"
            "prometheus:v2.53.4-arm64",
            "#  grafana:",
            "#    image: registry.cn-hangzhou.aliyuncs.com/eapil/"
            "grafana:11.6.0-arm64",
        ):
            self.assertIn(fragment, compose)

    def test_triton_service_preserves_nvidia_runtime_and_ports(self):
        compose = read_text("docker-compose.jetson.yaml")
        triton = service_block(compose, "triton-server")

        for expected in (
            "runtime: nvidia",
            "ipc: host",
            '"8000:8000"',
            '"8001:8001"',
            '"8003:8002"',
            "./model_repository:/models",
            "./logs:/var/log/triton",
            "NVIDIA_VISIBLE_DEVICES: all",
            "NVIDIA_DRIVER_CAPABILITIES: compute,utility",
            "TRITON_MODEL_SET",
            "${TRITON_MODEL_SET-all}",
            "start_triton.py",
            "--manifest /opt/triton/model_sets.json",
            "--model-repository /models",
            "--log-wrapper /usr/local/bin/run_triton_with_rotating_logs.py",
            "--model-repository=/models",
        ):
            self.assertIn(expected, triton)
        self.assertNotIn("${PWD}", triton)
        for build_fragment in (
            "build:",
            "context: .",
            "dockerfile: Dockerfile.triton.jetson",
        ):
            self.assertIn(build_fragment, triton)
        for forbidden in (
            "${TRITON_MODEL_SET:-all}",
            "TRITON_RUNTIME_PROFILE",
            "${TRITON_RUNTIME_PROFILE:-default}",
        ):
            self.assertNotIn(forbidden, compose)
        for forbidden in (
            "--model-control-mode",
            "--load-model",
            "--model-config-name",
        ):
            self.assertNotIn(forbidden, compose)

    def test_optional_monitoring_template_uses_service_dns(self):
        config = read_text("prometheus.jetson.yml")
        self.assertIn("triton-server:8002", config)
        self.assertNotIn("127.0.0.1:8003", config)
        compose = read_text("docker-compose.jetson.yaml")
        for fragment in (
            "#  prometheus:",
            "#    image: registry.cn-hangzhou.aliyuncs.com/eapil/"
            "prometheus:v2.53.4-arm64",
            "#      - ./prometheus.jetson.yml:/etc/prometheus/prometheus.yml:ro",
            "#  grafana:",
            "#    image: registry.cn-hangzhou.aliyuncs.com/eapil/"
            "grafana:11.6.0-arm64",
        ):
            self.assertIn(fragment, compose)

    def test_jetson_dockerfile_uses_private_base_and_existing_dependencies(self):
        dockerfile = read_text("Dockerfile.triton.jetson")
        self.assertIn(
            "registry.cn-hangzhou.aliyuncs.com/eapil/"
            "tritonserver:26.04-py3-arm64",
            dockerfile,
        )
        self.assertNotIn("nvcr.io/", dockerfile)
        for dependency in (
            "regex",
            "ftfy",
            "pillow",
            "numpy==1.26.4",
            "opencv-python-headless==4.10.0.84",
            "scikit-image==0.25.2",
        ):
            self.assertIn(dependency, dockerfile)
        self.assertIn(
            "COPY scripts/run_triton_with_rotating_logs.py ",
            dockerfile,
        )
        self.assertIn(
            "COPY scripts/start_triton.py /usr/local/bin/start_triton.py",
            dockerfile,
        )
        self.assertIn(
            "COPY profiles/model_sets.json /opt/triton/model_sets.json",
            dockerfile,
        )
        self.assertIn("mkdir -p /var/log/triton", dockerfile)

    def test_dockerignore_limits_the_build_context(self):
        ignored = read_text(".dockerignore").splitlines()
        self.assertEqual(
            ignored,
            [
                "**",
                "!Dockerfile.triton",
                "!Dockerfile.triton.jetson",
                "!scripts/",
                "!scripts/run_triton_with_rotating_logs.py",
                "!scripts/start_triton.py",
                "!profiles/",
                "!profiles/model_sets.json",
            ],
        )


class PublishScriptContractTests(unittest.TestCase):
    def test_publish_script_pins_sources_targets_and_arm64_validation(self):
        script = read_text("scripts/publish_jetson_arm64_images.ps1")
        required_images = (
            "nvcr.io/nvidia/tritonserver:26.04-py3",
            "quay.io/prometheus/prometheus:v2.53.4",
            "docker.m.daocloud.io/grafana/grafana:11.6.0",
            "registry.cn-hangzhou.aliyuncs.com/eapil/"
            "tritonserver:26.04-py3-arm64",
            "registry.cn-hangzhou.aliyuncs.com/eapil/"
            "tritonserver:26.04-py3-arm64-runtime",
            "registry.cn-hangzhou.aliyuncs.com/eapil/"
            "prometheus:v2.53.4-arm64",
            "registry.cn-hangzhou.aliyuncs.com/eapil/"
            "grafana:11.6.0-arm64",
        )
        for image in required_images:
            self.assertIn(image, script)

        for expected_arm64_digest in (
            "sha256:aa5fca20416939f243d587d56557d15f9d2bb81de30c6a4464707225250320fb",
            "sha256:d4aee9f05c4d8f294fe01de1a3d3db15ba01fa14c9c3cd19c3c0a2780118cba6",
            "sha256:c5bb8837eb70910907d8df3a11d0c489bd9e0f761cb67261e8d7d6af0724ddf3",
        ):
            self.assertIn(expected_arm64_digest, script)
        for required_fragment in (
            "Get-RemoteImageInfo",
            "Assert-ContainsArm64",
            "Assert-Arm64Only",
            "ConfigDigest",
            "$MaxAttempts = 3",
            "Start-Sleep -Seconds",
            '"pull", "--platform", "linux/arm64"',
            '"tag", $Source, $Target',
            '"push", $Target',
            '"buildx", "build"',
            '"--platform", "linux/arm64"',
            '"--push"',
            "Dockerfile.triton.jetson",
        ):
            self.assertIn(required_fragment, script)

        self.assertNotRegex(script, re.compile(r"docker\s+login", re.IGNORECASE))
        self.assertNotRegex(
            script,
            re.compile(r"(password|passwd)\s*=", re.IGNORECASE),
        )
        self.assertNotIn(
            "$targetInfo.Arm64Digest -ne $sourceInfo.Arm64Digest",
            script,
        )


class JetsonDocumentationContractTests(unittest.TestCase):
    def test_operator_guide_covers_lifecycle_and_real_gpu_smoke_test(self):
        guide = read_text("docs/Jetson部署指南.md")
        for required_fragment in (
            "Jetson AGX Orin",
            "JetPack 7.2",
            "docker compose -f docker-compose.jetson.yaml up -d --build",
            "docker compose -f docker-compose.jetson.yaml pull",
            "docker compose -f docker-compose.jetson.yaml up -d",
            "docker compose -f docker-compose.jetson.yaml down",
            "/v2/health/ready",
            "triton-server:8002",
            "clip_text_trt",
            '"data": [0] * 52',
            'assert output["datatype"] == "FP32"',
            'math.isfinite(value) for value in output["data"]',
            "clip_text_trt native TensorRT GPU smoke test passed",
            "tools/build_tensorrt_plans.py",
            "--precision fp32",
            "11 份匹配 Jetson 环境的 `model_throughput.plan`",
            "TensorRT 10.16.1.11",
            "TensorRT 10.16.2.10",
            "NVIDIA_TRITON_SERVER_VERSION",
            "TRITON_SERVER_VERSION",
            "CUDA_VERSION",
            "Prometheus 和 Grafana 默认未启用",
        ):
            self.assertIn(required_fragment, guide)
        self.assertNotIn("tritonserver --version", guide)
        self.assertNotIn("上述 CLIP 请求只验证 ONNX Runtime 链路", guide)

    def test_operator_guide_covers_upgrade_and_rollback(self):
        guide = read_text("docs/Jetson部署指南.md")
        for required_fragment in (
            "## 升级与回滚",
            "docker compose -f docker-compose.jetson.yaml pull",
            "docker compose -f docker-compose.jetson.yaml up -d",
            "docker compose -f docker-compose.jetson.yaml down",
            "RepoDigests",
            "model-artifacts.sha256",
            "model-artifacts.tar",
            "--force-recreate",
            "set -euo pipefail",
            "archive-check.XXXXXX",
            "restore.XXXXXX",
            "archive_bytes=$(du -sb model_repository",
            "available_bytes=$(df -PB1 \"$rollback_root\"",
            "required_bytes=$((archive_bytes * 2 + 1073741824))",
            "insufficient rollback space",
            "stat -c '%d'",
            "trap cleanup_archive_check EXIT",
            "trap cleanup_rollback_stage EXIT",
            "trap 'exit 130' INT",
            "trap 'exit 143' TERM",
            "preserve_rollback_stage=1",
            "preserving verified rollback staging",
        ):
            self.assertIn(required_fragment, guide)
        self.assertEqual(guide.count("trap 'exit 130' INT"), 2)
        self.assertEqual(guide.count("trap 'exit 143' TERM"), 2)
        self.assertNotIn("trap cleanup_archive_check EXIT INT TERM", guide)
        self.assertNotIn("trap cleanup_rollback_stage EXIT INT TERM", guide)
        self.assertNotIn("模型目录约为", guide)
        self.assertNotIn("建议回滚盘至少预留", guide)
        prepare_script = guide.split("```bash\nset -euo pipefail", 1)[1].split(
            "```", 1
        )[0]
        self.assertLess(
            prepare_script.index("stat -c '%d'"),
            prepare_script.index("tar -cpf"),
        )
        rollback_script = guide.split("升级后依次执行", 1)[1].split(
            "日常停止、恢复", 1
        )[0]
        self.assertLess(
            rollback_script.index("failed_dir=$(mktemp"),
            rollback_script.index(
                "docker compose -f docker-compose.jetson.yaml down"
            ),
        )
        self.assertLess(
            rollback_script.index("preserve_rollback_stage=1"),
            rollback_script.index(
                "docker compose -f docker-compose.jetson.yaml down"
            ),
        )
        success_markers = (
            'docker compose --project-directory "$repo_root" '
            '-f "$rollback_compose" ps',
            "preserve_rollback_stage=0",
            "cleanup_rollback_stage",
            "rollback_stage=''",
            "trap - EXIT",
            "trap - INT TERM",
        )
        for marker in success_markers:
            self.assertIn(marker, rollback_script)
        success_positions = [
            rollback_script.rindex(marker) for marker in success_markers
        ]
        self.assertEqual(success_positions, sorted(success_positions))

    def test_public_docs_do_not_restore_retired_plan_preservation_workflow(self):
        documents = (
            "README.md",
            "docs/模型接口与运维.md",
            "docs/模型下载与转换.md",
            "docs/模型来源与制品身份.md",
            "docs/Jetson部署指南.md",
        )
        # Native TensorRT plans are supported; the retired workflow is not.
        forbidden_tokens = (
            "preserved_plan",
            "tensorrt_candidates",
        )

        for document in documents:
            content = read_text(document).lower()
            for token in forbidden_tokens:
                with self.subTest(document=document, token=token):
                    self.assertNotIn(token.lower(), content)


class BatchInferenceDocumentationContractTests(unittest.TestCase):
    PUBLIC_CONTRACT_ROWS = (
        "| `yolov8n_face_ensemble` | `RAW_IMAGE [B,H,W,3]` | `DETECTIONS [B,100,6]`, `NUM_DETECTIONS [B,1]` |",
        "| `face_keypoint_2d106_ensemble` | `RAW_IMAGE [B,H,W,3]`, `BBOX [B,4]` | `LANDMARKS_106 [B,106,2]` |",
        "| `face_qa_ensemble` | `RAW_FACE_IMAGE [B,H,W,3]`, `LANDMARK [B,5,2]` | `quality_score [B,1]` |",
        "| `face_recognition_pipeline` | `RAW_IMAGE [B,H,W,3]`, `LANDMARK [B,5,2]` | `EMBEDDING [B,512]` |",
        "| `adaface_recognition_pipeline` | `RAW_IMAGE [B,H,W,3]`, `LANDMARK [B,5,2]` | `EMBEDDING [B,512]` |",
        "| `sbs_r50_ibn_image_ensemble` | `IMAGE_BYTES [B,1]` | `embedding [B,2048]` |",
        "| `osnet_ain_x1_0_image_ensemble` | `IMAGE_BYTES [B,L]` | `embedding [B,512]` |",
        "| `clip_ensemble_trt` | `IMAGE_BYTES [B,1]` | `unnorm_image_features [B,512]` |",
        "| `clip_text_ensemble` | `TEXT [B,1]` | `unnorm_text_features [B,512]` |",
        "| `TBPS_clip_image_ensemble` | `IMAGE_BYTES [B,1]` | `image_feat [B,512]` |",
        "| `TBPS_clip_text_ensemble` | `TEXT_IN [B,1]` | `text_feat [B,512]` |",
        "| `yolo26m_pose_ensemble` | `RAW_IMAGE [B,H,W,3]` | `DETECTIONS [B,100,6]`, `KEYPOINTS [B,100,17,3]`, `NUM_DETECTIONS [B,1]` |",
        "| `yolo26s_seg_ensemble` | `RAW_IMAGE [B,H,W,3]` | `DETECTIONS [B,M,6]`, `MASK_CROPS_PACKED [B,M,160,20]`, `NUM_DETECTIONS [B,1]` |",
    )
    README = "README.md"
    USAGE_GUIDE = "docs/模型接口与运维.md"
    PROVENANCE = "docs/模型来源与制品身份.md"
    JETSON_GUIDE = "docs/Jetson部署指南.md"
    DOWNLOAD_GUIDE = "docs/模型下载与转换.md"
    PUBLIC_DOCS = (README, USAGE_GUIDE, DOWNLOAD_GUIDE, PROVENANCE, JETSON_GUIDE)
    PROCESS_FRAGMENTS = (
        "历史 localhost 实例",
        "历史旧实例",
        "当前工作树",
        "dirty worktree",
        "本任务环境中的 `.git` 只读",
        "当前会话仍返回 permission denied",
        "当前验收摘要",
        "实际运行该命令，结果为",
        "tests，0 failure",
        "tests 通过；仅",
    )

    def test_provenance_document_has_a_docs_path_and_no_tools_alias(self):
        legacy_path = "tools" + "/model_source_report.md"
        with self.subTest(path=self.PROVENANCE):
            self.assertTrue((ROOT / self.PROVENANCE).is_file(), self.PROVENANCE)
        with self.subTest(path=legacy_path):
            self.assertFalse((ROOT / legacy_path).exists(), legacy_path)

        for relative_path in self.PUBLIC_DOCS:
            with self.subTest(document=relative_path):
                self.assertNotIn(legacy_path, read_text(relative_path))

    def test_readme_is_a_quick_entry_with_links_to_owned_documents(self):
        readme = read_text(self.README)
        self.assertLessEqual(len(readme.splitlines()), 320)
        for fragment in (
            "## 快速开始",
            "docker compose up -d --build",
            "docker compose -f docker-compose.jetson.yaml up -d --build",
            "/v2/health/ready",
            "/v2/repository/index",
            self.USAGE_GUIDE,
            self.PROVENANCE,
            self.JETSON_GUIDE,
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, readme)

    def test_usage_guide_is_the_only_complete_public_contract(self):
        usage = read_text(self.USAGE_GUIDE)
        for row in self.PUBLIC_CONTRACT_ROWS:
            with self.subTest(document=self.USAGE_GUIDE, row=row):
                self.assertEqual(
                    usage.count(row),
                    1,
                )

        for relative_path in (self.README, self.PROVENANCE, self.JETSON_GUIDE):
            document = read_text(relative_path)
            with self.subTest(document=relative_path):
                self.assertEqual(
                    sum(document.count(row) for row in self.PUBLIC_CONTRACT_ROWS),
                    0,
                )

    def test_documented_offline_gate_keeps_both_plan_builder_regressions(self):
        usage = read_text(self.USAGE_GUIDE)
        match = re.search(
            r"(?ms)^modules = (\(.*?^\))\ncompleted = subprocess.run\(",
            usage,
        )
        self.assertIsNotNone(match, "missing executable offline gate module list")
        modules = ast.literal_eval(match.group(1))
        self.assertEqual(len(modules), 13)
        self.assertEqual(len(set(modules)), len(modules))
        for required in (
            "test.test_yolo_shape_buckets",
            "test.test_tensorrt_plans",
            "test.test_yolo_tensorrt_plans",
            "test.test_jetson_deployment",
            "test.test_model_pipeline",
        ):
            self.assertIn(required, modules)
        for module in modules:
            with self.subTest(module=module):
                self.assertTrue(
                    (ROOT / Path(*module.split("."))).with_suffix(".py").is_file(),
                    module,
                )

    def test_provenance_links_catalog_sources_and_identity_records(self):
        provenance = read_text(self.PROVENANCE)
        catalog = json.loads(read_text("tools/model_pipeline/catalog.json"))
        for source_id, source in catalog["sources"].items():
            with self.subTest(source=source_id):
                self.assertIn(source_id, provenance)
                self.assertIn(source["filename"], provenance)
                for url in source.get("urls", []):
                    self.assertIn(url, provenance)
                for url in source.get("archive", {}).get("urls", []):
                    self.assertIn(url, provenance)
        for path in (
            "../tools/model_pipeline/catalog.json",
            "../tools/model_pipeline/catalog.json",
            "../tools/vendor/adaface/LICENSE",
            "../tools/vendor/torchreid/LICENSE",
            "../tools/vendor/sixdrepnet/LICENSE",
        ):
            with self.subTest(path=path):
                self.assertIn(path, provenance)
                self.assertTrue((ROOT / "docs" / path).exists())

    def test_download_guide_documents_source_only_candidate_pipeline(self):
        downloads = read_text(self.DOWNLOAD_GUIDE)
        self.assertIn(
            "[模型来源与制品身份](模型来源与制品身份.md)", downloads
        )
        self.assertNotRegex(downloads, re.compile(r"(?i)\b[A-Z]:[\\/]"))
        self.assertNotRegex(downloads, re.compile(r"(?i)\b[0-9a-f]{64}\b"))
        commands = [
            line.strip()
            for line in downloads.splitlines()
            if line.startswith("python -m tools.model_pipeline ")
        ]
        for action in ("list", "acquire", "onnx", "plan"):
            self.assertTrue(
                any(line.startswith(f"python -m tools.model_pipeline {action}") for line in commands),
                action,
            )
        for command in commands:
            if " onnx " in command or " plan " in command:
                with self.subTest(command=command):
                    self.assertIn("--output artifacts/", command)
                    self.assertNotIn("--model-repository", command)
            if " onnx " in command:
                self.assertIn("--no-reference", command)
        for batch in (1, 4, 8):
            with self.subTest(batch=batch):
                self.assertTrue(
                    any(
                        " plan " in line and f"--max-batch {batch} " in line
                        for line in commands
                    )
                )
        for lock in (
            "tools/model_pipeline/requirements-modern.txt",
            "tools/requirements-cn-clip-export.txt",
            "tools/model_pipeline/requirements-sbs-win.lock",
        ):
            with self.subTest(lock=lock):
                self.assertIn(lock, downloads)
                self.assertTrue((ROOT / lock).is_file())

    def test_public_docs_do_not_include_task_process_narrative(self):
        dated_process_narrative = re.compile(
            r"20\d{2}-\d{2}-\d{2}.{0,80}(运行态复验|当前验收|当前部署)"
        )
        for relative_path in self.PUBLIC_DOCS:
            document = " ".join(read_text(relative_path).split())
            for fragment in self.PROCESS_FRAGMENTS:
                with self.subTest(document=relative_path, fragment=fragment):
                    self.assertNotIn(fragment, document)
            with self.subTest(document=relative_path, pattern=dated_process_narrative.pattern):
                self.assertIsNone(dated_process_narrative.search(document))

    def test_readme_and_usage_document_one_b1_to_b4_runtime(self):
        for relative_path in (self.README, self.USAGE_GUIDE):
            document = " ".join(read_text(relative_path).split())
            with self.subTest(document=relative_path):
                self.assertNotIn("当前在线 max_batch_size 为 1", document)
                self.assertNotIn("TRITON_RUNTIME_PROFILE", document)
                self.assertIn("max_batch_size: 4", document)
                self.assertIn("model_throughput.plan", document)

    def test_offline_gate_renders_amd64_and_jetson_for_all_and_subset(self):
        usage = read_text(self.USAGE_GUIDE)
        gate = usage.split("### 10.3 ", 1)[1].split("### 10.4 ", 1)[0]
        self.assertNotIn("TRITON_RUNTIME_PROFILE", gate)
        self.assertEqual(gate.count("docker compose config > /dev/null"), 2)
        self.assertEqual(
            gate.count(
                "docker compose -f docker-compose.jetson.yaml config > /dev/null"
            ),
            2,
        )
        self.assertIn("export TRITON_MODEL_SET=all", gate)
        self.assertIn("export TRITON_MODEL_SET=yolov8n_face,face_qa", gate)

    def test_usage_examples_do_not_send_legacy_unbatched_shapes(self):
        usage = read_text("docs/模型接口与运维.md")
        forbidden_fragments = (
            'InferInput("RAW_IMAGE", list(image.shape), "UINT8")',
            'InferInput("BBOX", [4], "FP32")',
            '"RAW_FACE_IMAGE", list(face.shape), "UINT8"',
            'InferInput("LANDMARK", [5, 2], "FP32")',
            "payload = np.empty((1,), dtype=object)",
            'InferInput("IMAGE_BYTES", [1], "BYTES")',
            "batch size must be between 1 and 4",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, usage)

        self.assertGreaterEqual(
            usage.count("response batch size must be 1"),
            10,
        )

    def test_detailed_face_and_clip_image_contracts_include_explicit_b1(self):
        usage = read_text("docs/模型接口与运维.md")
        face_section = usage.split("### 6.1 `yolov8n_face_ensemble`", 1)[1].split(
            "### 6.2 `face_keypoint_2d106_ensemble`", 1
        )[0]
        for contract in (
            "| `RAW_IMAGE` UINT8 | `[B,H,W,3]` BGR |",
            "| `DETECTIONS` FP32 | `[B,100,6]` |",
            "| `NUM_DETECTIONS` INT32 | `[B,1]` |",
            "valid = detections_batch[0, :count]",
        ):
            self.assertIn(contract, face_section)

        clip_section = usage.split("### 6.8 `clip_ensemble_trt`", 1)[1].split(
            "### 6.9 `clip_text_ensemble`", 1
        )[0]
        self.assertIn("完整请求 shape 是 `[1,1]`", clip_section)
        self.assertIn('InferInput("IMAGE_BYTES", [1, 1], "BYTES")', clip_section)

    def test_detailed_segmentation_contract_uses_dynamic_mask_crops(self):
        usage = read_text("docs/模型接口与运维.md")
        seg_section = usage.split("### 6.13 `yolo26s_seg_ensemble`", 1)[1].split(
            "## 7. 活动模型与公开入口映射", 1
        )[0]
        for contract in (
            "`DETECTIONS [B,M,6]`",
            "`MASK_CROPS_PACKED [B,M,160,20]`",
            "`NUM_DETECTIONS [B,1]`",
            "`M = max(1, max(NUM_DETECTIONS))`",
            'InferRequestedOutput("MASK_CROPS_PACKED")',
            'response.as_numpy("MASK_CROPS_PACKED")',
            "cv2.INTER_NEAREST",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, seg_section)
        for stale_contract in (
            'InferRequestedOutput("MASKS")',
            'response.as_numpy("MASKS")',
        ):
            with self.subTest(stale_contract=stale_contract):
                self.assertNotIn(stale_contract, seg_section)

    def test_readme_does_not_advertise_online_reid_batch_repetition(self):
        readme = read_text("README.md")
        self.assertNotIn("`--batch-size` 会重复同一张图片", readme)

    def test_adaface_smoke_documents_batched_response_then_selected_row(self):
        usage = read_text("docs/模型接口与运维.md")
        adaface_row = next(
            line for line in usage.splitlines() if "| AdaFace 人脸链路 |" in line
        )
        self.assertIn("响应 `[1,512]`", adaface_row)
        self.assertIn("选择 row 0 后为 `[512]`", adaface_row)

class SmokeClientBatchContractTests(unittest.TestCase):
    def test_single_image_reid_smoke_clients_reject_cli_batch_sizes_above_one(self):
        for relative_path in (
            "test/test_sbs_r50_ibn_image.py",
            "test/test_osnet_ain_x1_0_image.py",
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / relative_path),
                    "--image",
                    "unused.jpg",
                    "--batch-size",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            with self.subTest(client=relative_path):
                self.assertEqual(completed.returncode, 2)
                self.assertIn("--batch-size", completed.stderr)
                self.assertIn("choose from 1", completed.stderr)

    def test_reid_clients_validate_b1_response_before_selecting_row_zero(self):
        for relative_path in (
            "test/test_sbs_r50_ibn_image.py",
            "test/test_osnet_ain_x1_0_image.py",
        ):
            source = read_text(relative_path)
            with self.subTest(client=relative_path):
                self.assertIn("expected_shape = (1, EMBEDDING_SIZE)", source)
                self.assertIn("return embedding[0]", source)

    def test_raw_image_clients_send_explicit_b1_and_select_b1_response(self):
        expected_fragments = {
            "test/test_yolo26m_pose_image.py": (
                "image_batch = np.ascontiguousarray(image[None, ...])",
                "select_online_outputs(",
                "detections.shape != (1, 100, 6)",
            ),
            "test/test_yolo26s_seg_image.py": (
                "image_batch = np.ascontiguousarray(image[None, ...])",
                "select_online_outputs(",
                'raise ValueError("DETECTIONS must have shape [1,M,6] with 1 <= M <= 100")',
            ),
        }
        for relative_path, fragments in expected_fragments.items():
            source = read_text(relative_path)
            with self.subTest(client=relative_path):
                for fragment in fragments:
                    self.assertIn(fragment, source)

    def test_segmentation_client_slices_selected_row_by_its_detection_count(self):
        source = read_text("test/test_yolo26s_seg_image.py")
        self.assertIn("count = int(counts[batch_index, 0])", source)
        self.assertIn('mask_crops[batch_index, :count], axis=-1, count=160, bitorder="big"', source)


class YoloOnlineResponseBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.detections = np.zeros((1, 100, 6), dtype=np.float32)
        self.seg_detections = np.zeros((1, 1, 6), dtype=np.float32)
        self.counts = np.array([[1]], dtype=np.int32)

    def test_pose_rejects_non_public_detection_capacity(self):
        detections = np.zeros((1, 99, 6), dtype=np.float32)
        keypoints = np.zeros((1, 99, 17, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, r"\[1,100,6\]"):
            POSE_CLIENT_MODULE.select_online_outputs(
                detections, keypoints, self.counts
            )

    def test_seg_accepts_dynamic_detection_capacity(self):
        detections = np.zeros((1, 2, 6), dtype=np.float32)
        mask_crops = np.zeros((1, 2, 160, 20), dtype=np.uint8)
        counts = np.array([[2]], dtype=np.int32)

        selected_detections, selected_crops, count = (
            SEG_CLIENT_MODULE.select_online_outputs(
                detections,
                mask_crops,
                counts,
                expect_masks=True,
                expected_image_shape=(12, 20),
            )
        )

        self.assertEqual(selected_detections.shape, (2, 6))
        self.assertEqual(selected_crops.shape, (2, 160, 160))
        self.assertEqual(count, 2)

    def test_seg_requires_masks_by_default(self):
        with self.assertRaisesRegex(ValueError, "MASK_CROPS_PACKED"):
            SEG_CLIENT_MODULE.select_online_outputs(
                self.seg_detections,
                None,
                self.counts,
                expect_masks=True,
                expected_image_shape=(12, 20),
            )

    def test_seg_boxes_only_allows_missing_masks(self):
        selected_detections, selected_masks, count = (
            SEG_CLIENT_MODULE.select_online_outputs(
                self.seg_detections,
                None,
                self.counts,
                expect_masks=False,
                expected_image_shape=(12, 20),
            )
        )

        self.assertEqual(selected_detections.shape, (1, 6))
        self.assertIsNone(selected_masks)
        self.assertEqual(count, 1)

    def test_seg_rejects_mask_crops_with_wrong_shape(self):
        mask_crops = np.zeros((1, 1, 159, 160), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "160,20"):
            SEG_CLIENT_MODULE.select_online_outputs(
                self.seg_detections,
                mask_crops,
                self.counts,
                expect_masks=True,
                expected_image_shape=(12, 20),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
