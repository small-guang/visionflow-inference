# Jetson AGX Orin ARM64 Triton 部署

[返回文档导航](../README.md#文档导航)

本手册适用于 Jetson AGX Orin、L4T R39.2、JetPack 7.2、CUDA 13.2、主机 TensorRT 10.16.2.10。Jetson 使用 `docker-compose.jetson.yaml`；现有 AMD64 主机继续使用 `docker-compose.yaml`。

## 阅读目录

| 阶段 | 章节 |
| --- | --- |
| 准备环境与模型 | [镜像](#镜像) · [前置检查](#前置检查) · [TensorRT plan 准备](#原生-tensorrt-plan-准备) |
| 启动与配置 | [启动](#启动) · [Batch 与 YOLO 尺寸分桶](#batch-与-yolo-shape-bucket) · [架构与版本检查](#架构与版本检查) |
| 验证部署 | [服务健康检查](#服务健康检查) · [分阶段发布检查](#分阶段发布检查) · [GPU 实际推理检查](#onnx-与原生-tensorrt-gpu-实际推理检查) |
| 日常运维 | [升级与回滚](#升级与回滚) · [故障定位](#故障定位) |

## 镜像

- 默认启用的 Triton：`registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime`
- 可选 Prometheus：`registry.cn-hangzhou.aliyuncs.com/eapil/prometheus:v2.53.4-arm64`
- 可选 Grafana：`registry.cn-hangzhou.aliyuncs.com/eapil/grafana:11.6.0-arm64`

`docker-compose.jetson.yaml` 中 Prometheus 和 Grafana 默认未启用，两个服务目前是注释模板。只有明确启用对应服务后，才会拉取可选镜像并开放 `9090`、`3003`；默认启动只包含 Triton。

Triton 26.04 容器声明 CUDA 13.2.1 和 TensorRT 10.16.1.11；主机包为 TensorRT 10.16.2.10。二者不是同一个补丁版本，因此验收时必须记录容器实际加载的库，不能只引用主机 `dpkg` 输出。

运行镜像必须包含 `/usr/local/bin/start_triton.py` 和 `/opt/triton/model_sets.json`。只更新 Compose 或模型仓库后重启旧容器不会得到启动器和 manifest。Jetson 节点持有完整源码时，可由 Compose 调用 `Dockerfile.triton.jetson` 在本机重建并强制替换容器：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build --force-recreate triton-server
```

集中发布时，发布方仍可在 ARM64 构建环境直接重建并推送 runtime tag；Jetson 部署节点再 pull 并强制重建容器：

```bash
docker build -f Dockerfile.triton.jetson \
  -t registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime .
docker push registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime
docker compose -f docker-compose.jetson.yaml pull
docker compose -f docker-compose.jetson.yaml up -d --force-recreate
```

若节点只负责部署、不在节点构建，使用上述 `pull` 流程，并确认 pull 到的是待部署版本的不可变 digest。若节点从本地源码构建，使用 Compose 的 `--build` 流程，不需要额外执行 `docker push`。两种路径都必须继续执行下文的容器内文件、配置与就绪检查。

## 前置检查

```bash
uname -m
cat /etc/nv_tegra_release
docker version
docker compose version
docker info | grep -i runtime
```

预期架构为 `aarch64`，Docker runtime 列表包含 `nvidia`。

登录阿里云 registry 时在终端交互输入凭据，不要把密码写入仓库：

```bash
docker login registry.cn-hangzhou.aliyuncs.com
```

<a id="yolo-plan-准备"></a>

## 原生 TensorRT plan 准备

仓库共 48 个模型目录：11 个原生 TensorRT、3 个 ONNX Runtime、20 个 Python、14 个 ensemble。启动所选原生链路前，在 Jetson 目标 GPU 上离线构建 plan。原生白名单包括 `clip_image_trt`、`clip_text_trt`、`TBPS_clip_image_trt`、`TBPS_clip_text_trt`、`sbs_r50_ibn_onnx`、`w600k_r50_trt`、`face_qa`、`face_keypoint_2d106` 和三个 YOLO（`yolov8n_face_onnx`、`yolo26m_pose_onnx`、`yolo26s_seg_onnx`）。仅 `adaface_ir50_webface4m_onnx`、`osnet_ain_x1_0_onnx`、`sixdrepnet_onnx` 继续使用 ONNX Runtime。全部 14 份 `1/model.onnx` 及原有哈希记录均须保留，11 个原生节点从完整 ONNX 构建并作数值参考；活动运行时只读取 B1～4 `1/model_throughput.plan`。

核对目标容器实际加载的 TensorRT 版本，并准备匹配的 ARM64 SDK 和 `trtexec`。主机与容器 TensorRT 的补丁版本不同，不能仅依据主机 SDK 判断兼容。RTX 3090 的 plan 不能复制到 Jetson；ORT TensorRT EP 子图缓存也不能改名充当完整 plan。

先按[模型下载与转换](模型下载与转换.md)准备 `artifacts/onnx-v1`，再在 Jetson 匹配 SDK 的构建环境中执行：

```bash
TRTEXEC='<匹配目标容器运行库的 trtexec 路径>'
python3 -m tools.model_pipeline plan \
  --models native --onnx-root artifacts/onnx-v1 \
  --output artifacts/plan-jetson-b1-v1 --max-batch 1 --opt-batch 1 \
  --workspace-mib 1024 --trtexec "$TRTEXEC" --dry-run
python3 -m tools.model_pipeline plan \
  --models native --onnx-root artifacts/onnx-v1 \
  --output artifacts/plan-jetson-b1-v1 --max-batch 1 --opt-batch 1 \
  --workspace-mib 1024 --trtexec "$TRTEXEC"
python3 -m tools.model_pipeline plan \
  --models native --onnx-root artifacts/onnx-v1 \
  --output artifacts/plan-jetson-b4-v1 --max-batch 4 --opt-batch 1 \
  --workspace-mib 1024 --trtexec "$TRTEXEC"
```

`--models native` 选择 11 个原生模型，也可指定 alias 或内部名称子集。构建器不读取 `.env`。Face QA、106 点默认 FP32，其余模型默认 FP16；某模型未通过精度验证时，在新的目录加 `--precision fp32` 重建。文本 INT64 输入、图像 FP32 输入和全部 FP32 输出均按模型契约保留。

每个候选目录包含 `<model>/1/model_bN.plan`、同名 JSON 与构建日志。构建使用 `--skipInference`，成功后仍需完成下文的目标 GPU 推理验证。B1/B4/B8 可作独立候选实验；当前只把验证过的 B1～4 候选发布为 `model_throughput.plan`，其旁车和日志路径也需与部署位置一致。旧 B1 `model.plan` 只作历史制品保留，不加载、不改名或重新构建其字节。构建本身只写候选，不会加载到 Triton。

保存 plan、对应旁车、源 ONNX 与根配置。源文件或目标运行环境变化后重新构建，启动与首次推理不会自动生成缺失的 engine。底层 `tools/build_tensorrt_plans.py` 与 YOLO 兼容 CLI 都固定生成 B1～4 `model_throughput.plan`，没有运行配置参数；日常候选实验使用上述通用入口。

## 启动

`TRITON_MODEL_SET` 选择 `all` 或逗号分隔的 alias；省略时加载全部模型。完整 14 个 alias 闭包见[模型接口与运维](模型接口与运维.md)。先备齐活动模型制品，再在仓库根目录渲染配置并启动：

```bash
mkdir -p logs
docker compose -f docker-compose.jetson.yaml config
docker compose -f docker-compose.jetson.yaml pull
TRITON_MODEL_SET=all \
docker compose -f docker-compose.jetson.yaml up -d
docker compose -f docker-compose.jetson.yaml ps
```

子集启动示例：

```bash
TRITON_MODEL_SET=yolov8n_face,face_qa \
docker compose -f docker-compose.jetson.yaml up -d --force-recreate
```

受支持的加载范围是 `all` 与 subset。Jetson 默认 `TRITON_MODEL_CONFIG=orin_gpu`，两种范围均启用 `--model-control-mode=explicit` 并只加载选中 alias 的 runtime 闭包。YOLO、TBPS 图像、CLIP/SBS 图像以及人脸关键点、质量、识别使用同进程融合入口，读取的 plan/helper 目录仍完整保留；其 readiness 不要求挂载在 `/models` 下的其他目录 READY。Triton 模型控制端口必须只在可信网络中开放，并由防火墙或认证网关阻止不可信调用 repository load/unload API。

AMD64 与显式 CPU 前后处理基线使用根配置：全链路 `max_batch_size: 4`，34 个可执行节点中 `clip_image_preprocess` 为 `count: 2`，其余 33 个节点为 `count: 1`；14 个推理节点（11 个 TensorRT、3 个 ONNX Runtime）配置 `dynamic_batching { max_queue_delay_microseconds: 2000 }`，最多额外等待 2 ms 组批，20 个 Python 前后处理节点保持空 `dynamic_batching {}`，不额外等待。14 个 ensemble 保持原有调度，配置文件不显式设置 `preferred_batch_size`。Jetson 默认的 `TRITON_MODEL_CONFIG=orin_gpu` 使用 face 4、pose 2、seg 2、TBPS 图像 1 个 GPU 实例，pose/seg 在 B≤4 内进行同尺寸跨 RPC 合批，零额外等待；这些具名配置位于各模型的 `configs/orin_gpu.pbtxt`，修改后需重新加载模型或重启 Triton 才会生效。如果任一所选链路超出 Jetson GPU/系统内存预算，停止发布并回滚到前一个已验证的批量发布；不得静默把 `count` 或 `max_batch_size` 改小。

显式设置 `TRITON_MODEL_CONFIG=` 可选择 CPU 前后处理基线；GPU overlay 会强制设回 `orin_gpu`。CLIP/SBS 与三个 face 公开入口使用 `pipeline_backend=orin_gpu`，在同一个 Python backend 实例中完成 CUDA 预处理和 TensorRT，避免预处理结果跨进程传输。五个入口均为 2 个 GPU 实例；CLIP/SBS 每实例分别有 4/2 个 CPU 解码线程，最多等待 500 µs；三个 face 入口最多等待 250 µs，均以 B4 为目标。图片解码和小型 FP64 仿射矩阵计算仍在 CPU。单实例的 SBS/face 曾出现单位图片 CPU 成本降低但高并发吞吐下降；双实例 v4 在本机 C16 下恢复为高于同轮基线的吞吐，不能据此认定其他设备或负载也最优。独立 pre 节点保留 `preprocess_backend=orin_gpu` 配置，默认融合链路不再单独加载它们。

合批 runtime 与预处理 helper 通过镜像 COPY 提供，更新时必须构建新镜像，不能仅修改宿主 pbtxt 或 reload。图片/人脸路径需要 `image_orin_preprocess.py`、`face_orin_preprocess.py`、`orin_trt_runtime.py`、`orin_pipeline_metrics.py`、`image_decode.py`、`image_orin_pipeline.py`、`face_orin_pipeline.py` 七份共享脚本。启动器拒绝缺失的融合入口配置或 wrapper，并保留原资源闭包校验；发布 smoke 核对实际加载的 GPU 参数与 B1～B4 输出。构建、数值对照与 datasets 验证见 [Orin 图片与人脸 GPU 预处理](Orin图片与人脸GPU预处理.md)。

本机 v4 与 v3 复用同一不可变镜像，只调整四个入口的具名 profile；此时必须同时归档镜像 ID、运行配置/源码清单、八个镜像脚本哈希（上述七份加 `start_triton.py`）、plan 哈希及正确性/性能报告 binding。镜像 tag 相同不能证明挂载 profile 相同。v4 已完成隔离验证，生产服务是否切换须查看实际部署记录，不能由隔离端点 READY 推断。

不要同时在同一个 Docker 主机启动 AMD64 和 Jetson 两份 Compose；它们使用相同的宿主机端口。

## Batch 与 YOLO shape bucket

根配置中的 11 个原生节点均保持单个 GPU instance、optimization profile `0`。`orin_gpu` 融合链路复用其 plan，每个公开入口实例各持有一个 TensorRT context，默认不再单独加载对应原生实例。八个固定输入模型仅 batch 维动态，min/opt 均 B1，max 为 B4；图像空间尺寸固定，Chinese-CLIP/TBPS 文本长度分别固定为 52/77 且输入始终为 INT64。三个 YOLO 的动态范围为 min `images:1x3x192x192`、opt `images:1x3x384x640`、max `images:4x3x640x640`。原有公开名称、tensor schema、预处理、后处理及契约版本保持不变。

1024 MiB workspace 是构建预算，不是总物理内存上限。离线构建须记录统一物理内存峰值；ready 后预热各固定形状及 YOLO 七档尺寸，验证 batch 1～4，并按每个模型所选精度核对 ONNX 参考与业务结果。精度、内存与性能结论以 Jetson 目标设备上的测量为准。构建、输入形状、数值验证及显式后端回滚见[原生 TensorRT plan](模型接口与运维.md#541-原生-tensorrt-plan)。

运行时接受显式 `1 <= B <= 4`。同一个 dense 请求中，所有样本的非 batch 维度必须完全一致；多输入入口的相关 tensor 还必须具有相同 `B`。三个 face 融合入口仅对原图 input 启用 `allow_ragged_batch`，允许不同 H/W 的独立请求进入同次 Python execute；每个请求仍须遵守原 dense Tensor 契约，bbox/landmark 输入不启用 ragged。

YOLO 的 `RAW_IMAGE [B,H,W,3]` 使用七档固定矩形 letterbox（公共契约 v2）：高×宽为 `192×640`、`384×640`、`480×640`、`640×640` 及前三档的转置；保持最长边缩放到 640，选择能容纳缩放结果的最小档位。先按原始 `(H,W)` 建 `shape bucket`，不同 H/W 图片作为独立请求发送；预处理后的固定 `(H',W')` 是下游 TensorRT 的第二层 bucket key，只有该尺寸一致的结果才可动态合并。每个模型共用一个动态 engine 并保留 ONNX 源。前后处理、batch 公共契约及 ALog v2 必须同步发布，具体规则见[固定矩形尺寸分桶](模型接口与运维.md#54-yolo-固定矩形尺寸分桶)。

## 架构与版本检查

```bash
docker compose -f docker-compose.jetson.yaml exec triton-server uname -m
docker compose -f docker-compose.jetson.yaml exec triton-server \
  bash -lc 'printf "NGC=%s Triton=%s CUDA=%s\n" \
    "$NVIDIA_TRITON_SERVER_VERSION" "$TRITON_SERVER_VERSION" "$CUDA_VERSION"'
docker compose -f docker-compose.jetson.yaml exec triton-server \
  bash -lc 'ldconfig -p | grep -E "libnvinfer|libcudart"'
docker compose -f docker-compose.jetson.yaml logs --tail=200 triton-server
```

第一条命令必须输出 `aarch64`。保存 Triton、CUDA、TensorRT 和 backend 版本日志，检查是否出现 CUDA 初始化、TensorRT ABI、模型加载或内存错误。

## 服务健康检查

默认部署检查 Triton live、ready 和直接暴露的指标端口：

```bash
curl --fail http://127.0.0.1:8000/v2/health/live
curl --fail http://127.0.0.1:8000/v2/health/ready
curl --fail http://127.0.0.1:8003/metrics | head
```

仅当已取消 `docker-compose.jetson.yaml` 中 Prometheus/Grafana 服务注释并重新执行 `config`、`pull`、`up -d` 后，再检查：

```bash
docker compose -f docker-compose.jetson.yaml config --services
curl --fail http://127.0.0.1:9090/api/v1/targets
```

启用监控后，Prometheus 返回的 Triton target 应为 `up`，容器网络内抓取地址应是 `triton-server:8002`；Grafana 才可从 Jetson 本机的 `http://127.0.0.1:3003` 访问。默认部署访问 `9090` 或 `3003` 失败是预期结果，不代表 Triton 不健康。

## 分阶段发布检查

先用 subset 加载一个代表性闭包，验证通过后再扩大 alias 集合。发布窗口中保留一个终端运行 `tegrastats`，另一个终端执行检查：

```bash
cat /etc/nv_tegra_release
tegrastats --interval 1000
```

```bash
docker compose -f docker-compose.jetson.yaml logs --tail=200 triton-server
curl -fsS http://127.0.0.1:8000/v2/health/ready
curl -fsS -X POST http://127.0.0.1:8000/v2/repository/index
curl -fsS http://127.0.0.1:8000/v2/models/yolov8n_face_ensemble/config
curl -fsS http://127.0.0.1:8000/v2/models/yolov8n_face_ensemble/ready
docker compose -f docker-compose.jetson.yaml exec triton-server \
  test -x /usr/local/bin/start_triton.py
docker compose -f docker-compose.jetson.yaml exec triton-server \
  test -r /opt/triton/model_sets.json
```

repository index 中应审阅所选 runtime 闭包；公开入口 config 必须与所选 profile 一致（默认 YOLO、TBPS 图像、CLIP/SBS 与三个 face 入口为 `configs/orin_gpu.pbtxt`），ready 必须成功。未单独加载的 plan/pre/post 资源目录不属于 readiness 目标。观察 `tegrastats` 的 GPU/系统内存、swap、温度和降频，同时检查 Triton 日志中的 OOM、backend 初始化和 shape 错误。任一门禁失败都停止扩大范围。

<a id="onnx-与-yolo-tensorrt-gpu-实际推理检查"></a>

## ONNX 与原生 TensorRT GPU 实际推理检查

仅当 `TRITON_MODEL_SET=all`，或逗号分隔的 selector 包含 `clip_text` alias、因而 `clip_text_trt` 闭包已加载时，才运行下面的请求。先核对其 config 为 `backend: "tensorrt"`、所选 plan 文件及 profile `0`。请求向 `[1,52]` INT64 输入写入零，并验证有限 FP32 `[1,512]` 输出，用于检查原生 TensorRT plan 的加载与执行；零 token 探针不代表真实文本准确性或 FP16 资格通过。subset 未选择 `clip_text` 时，应从当前 selected closure 中选择对应的 ONNX Runtime 或 TensorRT 可执行节点，并按该节点 config 构造等价 GPU smoke；不要调用未选中的 `clip_text_trt`。

```bash
python3 - <<'PY'
import json
import math
import urllib.request

request_body = {
    "inputs": [
        {
            "name": "text",
            "shape": [1, 52],
            "datatype": "INT64",
            "data": [0] * 52,
        }
    ],
    "outputs": [{"name": "unnorm_text_features"}],
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v2/models/clip_text_trt/infer",
    data=json.dumps(request_body).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=60) as response:
    result = json.load(response)

output = result["outputs"][0]
assert output["name"] == "unnorm_text_features", output
assert output["shape"] == [1, 512], output
assert output["datatype"] == "FP32", output
assert len(output["data"]) == 512, output
assert all(math.isfinite(value) for value in output["data"]), output
print("clip_text_trt native TensorRT GPU smoke test passed")
PY
```

已选择 `clip_text` 但模型未加载或请求失败时，先查看 `docker compose -f docker-compose.jetson.yaml logs triton-server`，不要用 CPU 回退掩盖 GPU 初始化问题。subset 未选择该 alias 时，`clip_text_trt` 返回 `NOT_FOUND` 只表示它不在所选加载目标中，不得误判为所选闭包的加载故障。

上述 CLIP 文本请求只验证一个原生节点的 B1 基础路径。CLIP 图像融合入口默认不单独加载 `clip_image_trt`，检查它时应调用 `clip_ensemble_trt`；其他融合入口同理。全部所选原生 plan 都须核对文件名和 profile `0`，再按[模型使用手册](模型接口与运维.md#541-原生-tensorrt-plan)执行固定业务金样本与 ONNX/plan 对照，覆盖所选 batch；文本侧使用真实 token 验证配对图文和检索排序，YOLO 额外覆盖七档尺寸及完整公开入口业务结果。若闭包还选择 AdaFace、OSNet 或 6DRepNet，则分别验证其 ONNX Runtime 路径。原生 TensorRT 无法在不支持算子时回退到 CUDA EP 或 CPU；解析、构建和加载错误必须处理。

## 升级与回滚

回滚目标必须是前一个已验证的批量发布，并成组恢复 launcher 镜像、根配置、模型制品和 alias 闭包 manifest。完整发布需要 11 份匹配 Jetson 环境的 `model_throughput.plan` 和 3 份 ORT 活动 ONNX；subset 发布只要求对应闭包的制品。升级前记录启动时实际生效的 `TRITON_MODEL_SET`，包含 Shell 临时覆盖后的值；回滚时显式恢复该记录，不能假定 `.env` 中的值就是实际部署集合。旧 B1 plan 不是活动回退。下面以原发布使用 `yolov8n_face,face_qa` 为例，执行时替换为发布记录中的实际模型集合：

```bash
TRITON_MODEL_SET=yolov8n_face,face_qa \
docker compose -f docker-compose.jetson.yaml up -d --force-recreate
curl -fsS http://127.0.0.1:8000/v2/health/ready
curl -fsS -X POST http://127.0.0.1:8000/v2/repository/index
```

Compose 当前使用固定 tag，而不是 digest 或镜像环境变量。升级前必须在仓库外建立回滚目录，记录 tag 当前解析到的不可变 `RepoDigests`，并归档模型仓库及其逐文件身份；其中包括全部 14 份 ONNX 活动文件/构建参考源、已构建的 plan、`.plan.json`、固定成功日志 `.plan.build.log` 及独立尝试日志 `.plan.build-UUID.log`，并保留失败构建的诊断记录。以下命令不依赖 Git，也不修改 `.git`：

归档并解包复核时会同时保留原目录、tar 和 staging 副本。脚本动态计算两个模型目录的实际字节数，并按“实际字节数的两倍加 1 GiB”计算最低需求，再用 `df` 检查回滚盘可用空间；空间不足时在归档前停止。

```bash
set -euo pipefail
repo_root=$(pwd -P)
rollback_root='../triton-rollbacks'
mkdir -p "$rollback_root"
rollback_root=$(cd "$rollback_root" && pwd -P)
release_id='pre-upgrade-001'
rollback_dir="$rollback_root/$release_id"
test ! -e "$rollback_dir" || { echo 'release_id already exists' >&2; exit 1; }
mkdir "$rollback_dir"
case "$rollback_dir" in
  "$repo_root"|"$repo_root"/*) echo 'rollback_dir must be outside the repository' >&2; exit 1 ;;
esac

archive_check_stage=''
cleanup_archive_check() {
  if [ -n "${archive_check_stage:-}" ] && [ -d "$archive_check_stage" ]; then
    case "$archive_check_stage" in
      "$rollback_root"/archive-check.*) rm -rf -- "$archive_check_stage" ;;
      *) echo 'refusing to clean unexpected staging path' >&2 ;;
    esac
  fi
}
trap cleanup_archive_check EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# archive-check/failed/restore 尚未创建，先比较它们的父目录设备号；
# 同时检查实际将移动的模型仓库目录，拒绝挂载在其他设备上的目录。
rollback_device=$(stat -c '%d' "$rollback_root")
repo_device=$(stat -c '%d' "$repo_root")
test "$repo_device" = "$rollback_device" || {
  echo 'repository and rollback_root must be on the same filesystem' >&2
  exit 1
}
for source_path in "$repo_root/model_repository"; do
  source_device=$(stat -c '%d' "$source_path")
  test "$source_device" = "$rollback_device" || {
    echo "$source_path and rollback_root must be on the same filesystem" >&2
    exit 1
  }
done

archive_bytes=$(du -sb model_repository | awk '{total += $1} END {print total}')
available_bytes=$(df -PB1 "$rollback_root" | awk 'NR == 2 {print $4}')
required_bytes=$((archive_bytes * 2 + 1073741824))
test "$available_bytes" -ge "$required_bytes" || {
  echo "insufficient rollback space: need $required_bytes bytes" >&2
  exit 1
}

triton_image='registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime'
current_container=$(docker compose -f docker-compose.jetson.yaml ps -q triton-server)
test -n "$current_container"
TRITON_MODEL_SET=$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$current_container" |
    sed -n 's/^TRITON_MODEL_SET=//p'
)
test -n "$TRITON_MODEL_SET"
export TRITON_MODEL_SET
printf '%s\n' "$TRITON_MODEL_SET" > "$rollback_dir/model-set.txt"
TRITON_MODEL_CONFIG=$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$current_container" |
    sed -n 's/^TRITON_MODEL_CONFIG=//p'
)
case "$TRITON_MODEL_CONFIG" in ""|orin_gpu) ;; *) exit 1 ;; esac
export TRITON_MODEL_CONFIG
printf '%s\n' "$TRITON_MODEL_CONFIG" > "$rollback_dir/model-config.txt"
docker compose -f docker-compose.jetson.yaml config > /dev/null
cp docker-compose.jetson.yaml "$rollback_dir/docker-compose.jetson.yaml"
docker compose -f docker-compose.jetson.yaml config > "$rollback_dir/compose.rendered.yaml"
rollback_digest=$(
  docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$triton_image" |
    grep '^registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver@sha256:' |
    head -n 1
)
test -n "$rollback_digest"
printf '%s\n' "$rollback_digest" > "$rollback_dir/triton-image.digest"

find model_repository -type f -print0 |
  sort -z |
  xargs -0 sha256sum > "$rollback_dir/model-artifacts.sha256"
tar -cpf "$rollback_dir/model-artifacts.tar" model_repository
(cd "$rollback_dir" && sha256sum model-artifacts.tar > model-artifacts.tar.sha256)
(cd "$rollback_dir" && sha256sum -c model-artifacts.tar.sha256)

archive_check_stage=$(mktemp -d "$rollback_root/archive-check.XXXXXX")
tar -xpf "$rollback_dir/model-artifacts.tar" -C "$archive_check_stage"
(cd "$archive_check_stage" && sha256sum -c "$rollback_dir/model-artifacts.sha256")
cleanup_archive_check
archive_check_stage=''
trap - EXIT
trap - INT TERM

docker compose -f docker-compose.jetson.yaml pull
docker compose -f docker-compose.jetson.yaml up -d --build
docker compose -f docker-compose.jetson.yaml ps
```

从仍使用具名备选配置的安装升级时，必须在重启旧容器前执行上述 `up -d --build`，让 launcher 镜像包含不再注入具名配置的新启动器；模型仓库已不再提供这些具名配置。这次迁移不要求重建 engine，原有模型 alias 选择与制品 provenance 继续保留。

升级后依次执行架构与版本检查、健康检查，以及与当前 selected closure 匹配的 ONNX Runtime / TensorRT GPU 推理检查；只有闭包已加载 `clip_text_trt` 时才使用上面的固定 smoke，全部所选原生节点还需验证 plan 与业务结果。任一步失败时，从前一个已验证的批量发布归档恢复，不依赖可变 tag 猜测回滚版本。归档脚本从运行容器读取实际模型集合与 profile，写入 `model-set.txt`、`model-config.txt`；恢复脚本在 Compose 预检前重新导出该值，覆盖当前 Shell 或 `.env` 的选择。旧归档缺少此文件时，先按原发布记录补齐实际模型集合，脚本不会默认改用 `all`：

```bash
set -euo pipefail
repo_root=$(pwd -P)
rollback_root='../triton-rollbacks'
rollback_root=$(cd "$rollback_root" && pwd -P)
release_id='pre-upgrade-001'
rollback_dir="$rollback_root/$release_id"
rollback_compose="$rollback_dir/docker-compose.jetson.yaml"
test -f "$rollback_compose"
TRITON_MODEL_SET=$(cat "$rollback_dir/model-set.txt")
test -n "$TRITON_MODEL_SET"
export TRITON_MODEL_SET
test -f "$rollback_dir/model-config.txt"
TRITON_MODEL_CONFIG=$(cat "$rollback_dir/model-config.txt")
case "$TRITON_MODEL_CONFIG" in ""|orin_gpu) ;; *) exit 1 ;; esac
export TRITON_MODEL_CONFIG
triton_image='registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime'

rollback_stage=''
preserve_rollback_stage=0
cleanup_rollback_stage() {
  if [ -n "${rollback_stage:-}" ] && [ -d "$rollback_stage" ]; then
    if [ "${preserve_rollback_stage:-0}" -eq 1 ]; then
      echo "preserving verified rollback staging: $rollback_stage" >&2
      echo "preserving failed deployment directories: ${failed_dir:-not-created}" >&2
      return
    fi
    case "$rollback_stage" in
      "$rollback_root"/restore.*) rm -rf -- "$rollback_stage" ;;
      *) echo 'refusing to clean unexpected restore path' >&2 ;;
    esac
  fi
}
trap cleanup_rollback_stage EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# restore/failed 尚未创建，比较它们的父目录；当前模型仓库目录也必须
# 与 rollback_root 同设备，才能保证后续 mv 是同文件系统 rename。
rollback_device=$(stat -c '%d' "$rollback_root")
repo_device=$(stat -c '%d' "$repo_root")
test "$repo_device" = "$rollback_device" || {
  echo 'repository and rollback_root must be on the same filesystem' >&2
  exit 1
}
for source_path in "$repo_root/model_repository"; do
  source_device=$(stat -c '%d' "$source_path")
  test "$source_device" = "$rollback_device" || {
    echo "$source_path and rollback_root must be on the same filesystem" >&2
    exit 1
  }
done

(cd "$rollback_dir" && sha256sum -c model-artifacts.tar.sha256)
rollback_stage=$(mktemp -d "$rollback_root/restore.XXXXXX")
tar -xpf "$rollback_dir/model-artifacts.tar" -C "$rollback_stage"
(cd "$rollback_stage" && sha256sum -c "$rollback_dir/model-artifacts.sha256")

rollback_digest=$(cat "$rollback_dir/triton-image.digest")
case "$rollback_digest" in
  registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver@sha256:*) ;;
  *) echo 'invalid recorded Triton digest' >&2; exit 1 ;;
esac
docker pull "$rollback_digest"
docker tag "$rollback_digest" "$triton_image"

docker compose -f docker-compose.jetson.yaml config > /dev/null
docker compose --project-directory "$repo_root" -f "$rollback_compose" config > /dev/null
failed_dir=$(mktemp -d "$rollback_root/failed.XXXXXX")
preserve_rollback_stage=1
docker compose -f docker-compose.jetson.yaml down
mv model_repository "$failed_dir/"
mv "$rollback_stage/model_repository" "$repo_root/"
docker compose --project-directory "$repo_root" -f "$rollback_compose" up -d --force-recreate
docker compose --project-directory "$repo_root" -f "$rollback_compose" ps
preserve_rollback_stage=0
cleanup_rollback_stage
rollback_stage=''
trap - EXIT
trap - INT TERM
```

进入 `down` 前脚本会开启 staging 保留标志；此后失败、`INT` 或 `TERM` 都会经 `EXIT` 输出 `rollback_stage` 和 `failed_dir`，不会自动删除即时恢复材料。目录切换前，已验证旧制品位于 `rollback_stage`；切换后，旧制品位于仓库活动目录，升级后的制品位于 `failed_dir`。根据中断位置从这三个路径手工恢复并重新执行 Compose 预检。

只有 `up -d` 与 `ps` 都成功后，脚本才依次关闭保留标志、安全清理空 staging、清空变量并注销 `EXIT`/`INT`/`TERM` trap，因此成功路径不会误报需要人工恢复。`failed_dir` 和回滚归档仍会保留；重复全部验证后再手工处理它们，不使用未核对路径的递归删除。

日常停止、恢复或移除 Compose 服务可使用：

```bash
docker compose -f docker-compose.jetson.yaml stop
docker compose -f docker-compose.jetson.yaml start
docker compose -f docker-compose.jetson.yaml down
```

`down` 不删除绑定挂载的模型和 `logs`。不要在未明确目标路径时递归删除模型目录。

## 故障定位

- `unknown or invalid runtime name: nvidia`：检查 NVIDIA Container Runtime 安装与 Docker daemon runtime 配置。
- `exec format error`：检查镜像远程清单和容器内架构，必须为 `linux/arm64` / `aarch64`。
- Triton readiness 失败：检查完整启动日志、backend 版本、模型加载错误和 Orin 内存。
- 已启用 Prometheus 且 target 为 `down`：确认配置使用 `triton-server:8002`，不要改成 `127.0.0.1:8003`；默认未启用监控时不执行此检查。
- Python backend 导入失败：核对运行镜像中的固定依赖版本，不在运行容器内临时 `pip install`。
- CUDA/TensorRT 错误：同时记录主机 L4T/JetPack 包版本和容器实际库版本，再判断驱动挂载或 ABI 问题。
