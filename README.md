# Triton 模型推理服务

使用 NVIDIA Triton Inference Server 提供人脸、行人、CLIP、姿态和分割推理。当前仓库有 48 个模型目录：11 个 TensorRT 节点、3 个 ONNX Runtime 节点、20 个 Python 前后处理节点和 14 个公开 ensemble。

## 文档导航

首次部署从本页的[快速开始](#快速开始)开始，再按使用场景查阅下面的指南。

| 使用场景 | 文档 | 内容 |
| --- | --- | --- |
| 准备模型 | [模型下载与转换](docs/模型下载与转换.md) | 下载源文件、导出 ONNX、构建 B1/B4/B8 plan |
| 核对来源 | [模型来源与制品身份](docs/模型来源与制品身份.md) | 实际模型名称、固定下载地址、SHA256 与许可证 |
| 接入与运维 | [模型接口与运维](docs/模型接口与运维.md) | 14 个公开入口、调用示例、batch/shape、部署验证和排障 |
| 部署到 Jetson | [Jetson 部署指南](docs/Jetson部署指南.md) | AGX Orin ARM64 环境、目标设备构建、启动、监控和回滚 |
| 图片与人脸预处理 | [Orin 图片与人脸 GPU 预处理](docs/Orin图片与人脸GPU预处理.md) | 同进程 TensorRT 融合、有界图片解码、数据集对照与镜像生效检查 |
| 发布独立部署镜像 | [Jetson 镜像发布](docs/Jetson镜像发布.md) | 内置模型镜像、两文件交付、版本发布与回滚 |
| 维护转换工具 | [模型维护工具说明](docs/模型维护工具说明.md) | 统一转换入口、脚本职责、本地源文件和输出目录 |

正式指南统一放在 `docs/`，使用中文文件名；第三方源码的来源说明随 `tools/vendor/` 中的代码保留。已完成的实施计划归档在 `artifacts/历史任务记录/`，下载缓存、转换候选和验证产物存放在 `artifacts/`；该目录由 Git 忽略。

## 模型与运行配置

启动器通过 `TRITON_MODEL_SET` 选择 `all` 或逗号分隔的模型 alias；省略时加载全部模型。完整 alias 和依赖闭包见[接口文档](docs/模型接口与运维.md#23-14-个模型-alias-与闭包)。

| 全链路 batch 上限 | TensorRT 部署文件 | 调度 |
| ---: | --- | --- |
| `max_batch_size: 4` | `model_throughput.plan` | 34 个可执行节点单实例；14 个推理节点最多等待 2 ms，20 个 Python 节点不额外等待 |

11 个 TensorRT 节点包括三个 YOLO、两套 CLIP 的图像/文本编码器、SBS、w600k_r50、Face QA 和 106 点关键点。AdaFace、OSNet、6DRepNet 使用 ONNX Runtime。内部名称的 `_trt` / `_onnx` 后缀是兼容名称，实际后端以 `config.pbtxt` 为准。

三个 YOLO 共用七档模型输入尺寸（高×宽）：`192×640`、`384×640`、`480×640`、`640×640`、`640×480`、`640×384`、`640×192`。每个模型使用一个动态 engine 覆盖这些尺寸，调用方仍须遵守[分桶规则](docs/模型接口与运维.md#54-yolo-固定矩形尺寸分桶)。

## 快速开始

### 1. 准备环境

需要 Docker Compose、NVIDIA GPU 驱动和可让 Docker 访问 GPU 的容器运行环境。本地模型维护另外需要 Python 3。

AMD64 使用 `docker-compose.yaml` 和 `Dockerfile.triton`；Jetson 使用 `docker-compose.jetson.yaml` 和 `Dockerfile.triton.jetson`。

Jetson 开发与独立交付默认使用 `TRITON_MODEL_CONFIG=orin_gpu`：pose/seg 各 2 个 GPU 实例，将已排队的同尺寸 RPC 合成 B≤4，零额外等待。face 保留双输出，其他模型保持原 Tensor 契约。显式设置 `TRITON_MODEL_CONFIG=` 可选择 CPU 前后处理基线；AMD64 Compose 不读取此变量。升级必须构建包含 `yolo_orin_pipeline.py`、`yolo_request_batching.py`、`yolo_orin_metrics.py` 的镜像，仅改 pbtxt 或 reload 模型不会更新镜像内脚本，详见[YOLO GPU 前后处理](docs/YOLO_GPU处理.md)。

`orin_gpu` 还把 CLIP/SBS 与人脸关键点、质量、识别五个公开入口融合为同进程 Python/CUDA/TensorRT 执行：有界 CPU 线程池解码图片，CUDA 完成 resize/仿射等像素处理，GPU tensor 直接交给同实例的 TensorRT context，仅下载推理结果。五个入口最终均使用 2 个 GPU 实例；CLIP/SBS 每实例分别有 4/2 个 CPU 解码线程，组批等待为 500 µs，三个 face 入口为 250 µs，均以 B4 为目标。公开 Tensor 契约保留；启动器只加载融合入口，原 plan/helper 目录仍作为资源保留。入口要求 `pipeline_backend=orin_gpu`；独立 pre 节点的 `preprocess_backend` 仅用于兼容和诊断。2026-10-21 本机 v4 的完整正确性与 C1/C16、B1/B4 共 80 个性能窗口通过，C16 各入口吞吐高于同轮 GPU-host 基线；这属于直接 Triton 口径，不能替代 ALog 全链路验收。生产切换另行记录。升级需要包含七份共享脚本的新镜像；单独改变已验收镜像外的实例 profile 仍须重新验证并记录 profile 哈希，详见[Orin 图片与人脸 GPU 预处理](docs/Orin图片与人脸GPU预处理.md)。

首次配置时，在仓库根目录将 [`.env.example`](.env.example) 复制为 `.env`，再按需修改；已有 `.env` 时按示例补齐配置即可。

```powershell
Copy-Item .env.example .env
```

POSIX shell：

```bash
cp .env.example .env
```

两套 Compose 都会读取根目录的 `.env`，默认 `TRITON_MODEL_SET=all`。示例文件列出了全部模型 alias；Shell 中同名环境变量优先于 `.env`。本地 `.env` 被 Git 忽略，示例文件纳入版本管理。

### 2. 准备模型

日常维护从仓库根目录运行统一入口：

```powershell
python -m tools.model_pipeline list
python -m tools.model_pipeline acquire --models all --cache artifacts/sources
```

随后按[模型下载与转换](docs/模型下载与转换.md)安装两套导出环境，执行 `onnx --no-reference` 和 `plan`。完整链路为：

```text
固定来源 → artifacts/sources → ONNX 候选 → 目标 GPU 的 plan 候选
```

无需已有 ONNX 即可导出。候选默认放在 `artifacts/`，转换入口不读取其他项目的 `.env`，模型子集由 `--models` 显式选择。B8 engine 的构建方法见[plan 构建](docs/模型下载与转换.md#3-onnx-转为-plan)；扩大 engine 的 batch 后还需同步验收 Triton 和调用方的批量配置。

部署模型位于：

```text
model_repository/<内部模型名>/1/
```

全部 14 份 `model.onnx` 需要保留：3 份供 ONNX Runtime 加载，11 份供 TensorRT 构建和参考验证。原生节点还需要 `model_throughput.plan` 与同名 `.json`。这些二进制文件和构建日志被 Git 忽略，应在启动前准备好。

B1/B4/B8 可以在独立目录构建为候选实验；当前部署只接受验证过的 B1～4 engine，活动文件名为 `model_throughput.plan`。旧 B1 `model.plan` 只作历史制品保留，不加载、不改名或重建其字节。plan 需要匹配目标 GPU、架构和 TensorRT 运行库，Jetson 必须在目标环境重新构建。部署核对项见[原生 TensorRT plan](docs/模型接口与运维.md#541-原生-tensorrt-plan)。

### 3. 启动

在仓库根目录执行。PowerShell 需设置 Compose 挂载使用的 `PWD`：

```powershell
$env:PWD = (Get-Location).Path
docker compose up -d --build
```

POSIX shell：

```bash
docker compose up -d --build
```

Jetson：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build
```

模型子集示例（POSIX）：

```bash
TRITON_MODEL_SET=yolov8n_face,face_qa \
docker compose up -d
```

从旧的双运行配置版本升级时，必须用 `up -d --build` 重建 launcher 镜像，因为新启动器已写入镜像，且模型仓库不再提供具名备选配置。原有 `TRITON_MODEL_SET` 值可以保留。修改 Dockerfile、启动器或模型集合 manifest 后，也需要重新构建镜像。修改模型文件或调度配置后，需要重新加载模型或重启服务。

### 4. 检查就绪状态

Windows 使用 `curl.exe`；POSIX 使用 `curl`：

```powershell
curl.exe --fail --show-error http://localhost:8000/v2/health/live
curl.exe --fail --show-error http://localhost:8000/v2/health/ready
curl.exe --fail-with-body --show-error -X POST http://localhost:8000/v2/repository/index
```

命令退出码应为 0。live 只检查进程；ready 检查启动时选择的模型闭包；repository index 用于逐项确认预期模型为 `READY`。继续对所选入口执行实际请求，示例见[模型接口与运维](docs/模型接口与运维.md#6-对外-ensemble-完整调用)。

## 仓库结构

```text
model_repository/               Triton 配置、Python 模型及活动制品
artifacts/                      下载缓存及独立 ONNX/plan 候选（Git 忽略）
tools/models/                   可选本地源文件，子目录用 .gitkeep 保留
profiles/model_sets.json        alias 与模型依赖闭包
scripts/                       服务启动、日志和镜像发布
tools/model_pipeline/           源文件清单和统一转换入口
tools/model_pipeline/catalog.json 模型来源、转换配方与当前 ONNX 身份
tools/vendor/                  固定模型结构源码与许可证
test/                          离线测试及推理客户端
docs/                          当前功能文档
```

## 监控与排障

| 宿主机端口 | 用途 |
| ---: | --- |
| `8000` | Triton HTTP |
| `8001` | Triton gRPC |
| `8003` | Triton metrics，容器端口为 `8002` |
| `9090` | Prometheus，启用对应服务时开放 |
| `3003` | Grafana，启用对应服务时开放 |

Prometheus 在宿主机运行时采集 `127.0.0.1:8003`；在同一 Compose 网络内运行时采集 `triton-server:8002`。Triton HTTP/gRPC 本身没有认证或 TLS，repository control API 应限制在可信网络内。

模型未就绪时先查看日志：

```bash
docker compose logs --tail 200 triton-server
```

核对版本目录、模型文件、plan 兼容性、配置中的 tensor 名/类型/维度，以及 Python 后端依赖。完整检查顺序见[排障](docs/模型接口与运维.md#104-排障顺序)。
