# Jetson 内置模型镜像发布与部署

[返回文档导航](../README.md#文档导航)

`deploy/jetson/` 用于构建包含模型的 Jetson 发布镜像。部署设备只需要 `compose.yaml`、`.env`，以及访问镜像仓库的权限，不需要本仓库、模型文件夹或 Python 开发环境。根目录的 Dockerfile、两套 Compose、模型目录和启动器仍用于原来的开发调试流程。

## 1. 文件与交付范围

| 文件 | 职责 |
| --- | --- |
| `deploy/jetson/Dockerfile` | 继承 Jetson runtime，内置模型、前后处理、词表、配置与启动命令 |
| `deploy/jetson/Dockerfile.dockerignore` | 发布构建文件筛选；不改变根目录开发构建规则 |
| `deploy/jetson/compose.yaml` | 独立部署模板，只引用镜像，不在部署设备构建 |
| `deploy/jetson/.env.example` | 镜像引用、模型集合、端口和日志配置示例 |
| `deploy/jetson/release_models.json` | 构建时包含的服务 alias 清单，决定实际打包范围 |
| `deploy/jetson/release.sh` | 发布入口，可从任意工作目录调用 |
| `deploy/jetson/release.py` | 制品检查、构建快照、验收、交付文件生成和显式推送 |
| `deploy/jetson/smoke.py` | 镜像内容校验及目标设备 HTTP 推理验收 |

当前发布镜像暂不包含 **AdaFace、OSNet、6DRepNet** 三个服务及其专用前后处理、ensemble 和模型文件。`release_models.json` 明确选择其余 11 个公开入口及完整依赖，共 38 个模型目录：11 个 TensorRT 节点、16 个 Python 节点、11 个 ensemble。模型配置、词表、已有 plan 旁车和模型版本目录一并保留；排除日志、缓存和隐藏文件。模型实际位于镜像 `/models`，Compose 不向此目录挂载任何卷。

交付镜像与 base Compose 默认选择 `TRITON_MODEL_CONFIG=orin_gpu`。pose/seg 各 2 个实例，支持同尺寸跨 RPC 合批 B≤4、零额外等待；face detector、TBPS 图像以及 CLIP/SBS、人脸关键点、质量、识别使用各自 GPU 融合入口。打包仍保留上述 38 个资源目录，实际加载的 runtime 数量由构建时冻结的 manifest 计算并写入 `default_runtime_model_count`；融合入口读取的 plan/helper 目录不再作为独立实例加载。显式 `TRITON_MODEL_CONFIG=` 保留 CPU 前后处理评估能力。

CLIP/SBS 的 resize、颜色变换与归一化，以及人脸关键点、质量、识别的仿射预处理，在该 profile 下与 TensorRT 放在同进程内执行，公开接口保持不变。最终五个入口各使用 2 个 GPU 实例；CLIP/SBS 每实例解码 worker 为 4/2，组批等待分别为图片入口 500 µs、face 入口 250 µs。图片/人脸路径的七份共享脚本随构建快照进入镜像并记录 SHA256：`image_orin_preprocess.py`、`face_orin_preprocess.py`、`orin_trt_runtime.py`、`orin_pipeline_metrics.py`、`image_decode.py`、`image_orin_pipeline.py`、`face_orin_pipeline.py`。smoke 验证五个融合入口的 `pipeline_backend=orin_gpu`、Python GPU instance 和 B4 上限，并逐个请求 B1～B4。smoke 的有限值/形状检查不替代 datasets 数值与持续性能验收；运行库或 helper 更新需要新镜像，详见 [Orin 图片与人脸 GPU 预处理](Orin图片与人脸GPU预处理.md)。

2026-10-21 本机 v4 完整正确性与 80 个 C1/C16 性能窗口通过，C16 五入口均高于同轮 GPU-host 基线。v3/v4 镜像相同而挂载 profile 不同，验收身份必须包含运行配置/源码清单，不能仅比较镜像 ID；最终打包时应冻结被选中的双实例 profile。此隔离实验不等同于独立交付镜像的默认 GPU `all` 发布验收，也不证明生产服务已完成切换。

11 个 TensorRT 节点的参考 ONNX **不进入镜像**，构建时在发布方的 `reference_onnx/` 归档，哈希仍写入发布记录用于关联 plan 来源。由于三个 ONNX Runtime 服务已排除，当前发布镜像不含 ONNX 模型文件。现有开发目录的全部模型、14 份 ONNX 及根 manifest 不受影响。归档不在 Docker 构建上下文中，因此这些参考文件不会留在镜像的任何一层。

构建时生成仅含所选服务的镜像内 manifest；`TRITON_MODEL_SET=all` 表示加载本镜像内的全部 11 个服务，也可填写其中一个或多个 alias。填写未打包的 `adaface`、`osnet`、`sixdrepnet` 会明确报错。运行时选择子集只改变加载范围，不缩小镜像下载量；构建时的 `release_models.json` 才决定哪些文件进入镜像。日志使用 Docker 自动创建的命名卷，同时保留 `docker compose logs` 和两种日志轮转机制。

以后恢复某个服务时，将对应 alias 加回 `release_models.json`，补齐并验收其模型制品，再构建新版本，并同步更新 `.env.example` 的 alias 说明。构建器按选中 alias 的依赖并集打包，共享节点只保留一份。恢复 ONNX Runtime 服务时，它的活动 ONNX 会随该服务重新进入镜像。

## 2. 支持环境与版本

首版面向本项目的 Jetson AGX Orin / L4T R39.2 环境。主机需要 Docker Engine、支持 `up --wait` 的 Compose 插件，以及已配置的 NVIDIA 容器运行环境。构建和验收都应在目标 Jetson 环境执行；不要在另一类 GPU 上生成 plan 后直接打包给 Jetson。

默认复用已发布的 runtime：

```text
registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime
```

发布脚本读取该本地镜像的真实 image ID，以完整 ID 创建固定的本地构建引用，并在构建后核对基础镜像层。这样也支持已导入本机、远程 digest 已不可用的 runtime；已有仓库 digest 作为来源信息记录，不假设它仍可拉取。缺少本地 runtime 时先拉取；如需要更新 runtime，先按 [Jetson 部署指南](Jetson部署指南.md) 的现有流程准备它。内置模型镜像构建期间不安装依赖、不下载模型，也不生成 TensorRT engine。

应用镜像使用独立名称和唯一版本，例如：

```text
registry.cn-hangzhou.aliyuncs.com/eapil/camtellsvi-triton:1.0.0-jetson-orin-l4t39.2
```

代码、前后处理、配置、权重或运行环境任一项发生变化，都应生成新版本。发布脚本拒绝 `latest`、`stable`，拒绝覆盖已有本地标签、已有输出目录以及已存在的远程版本标签。建议镜像仓库同时启用标签不可变策略，避免并发发布覆盖同一版本。

发布记录包括 Git 提交、构建时工作区状态、文件 SHA256、runtime image ID 及已有仓库 digest、主机 L4T 信息、最终镜像 ID，以及目标容器验收时实际加载的 TensorRT 库路径和 Python 依赖版本。同为 ARM64 不代表 plan 可以跨 Jetson 型号、JetPack 或 TensorRT 版本使用；支持范围以目标设备验收记录为准。

## 3. 发布方：检查与构建候选

下面的命令在仓库根目录执行。路径参数按当前工作目录解析；脚本自身始终从其所在仓库读取源码和模型。需要 Docker 权限的命令，在没有 Docker socket 权限时可加 `sudo`；仓库登录与推送必须使用同一个 Docker 凭据环境。

先检查所有待打包模型：

```bash
./deploy/jetson/release.sh check
```

检查只验证发布清单选中服务的模型目录、各可执行版本的活动模型文件，以及 TensorRT 节点的参考 ONNX；不会要求被排除服务的制品齐全。已有 `.plan.json` 必须与源 ONNX 和 plan 哈希匹配，并包含构建命令和精度。缺少旁车会输出 `missing_provenance`；候选构建允许保留这一明确状态，正式推送不接受缺失记录。

不要凭现有 plan 字节编造构建记录。应找回对应构建日志/旁车，或者通过现有模型转换工具在匹配环境中生成并验收新候选，然后按原模型维护流程更新待发布制品。

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime

./deploy/jetson/release.sh build \
  --image registry.cn-hangzhou.aliyuncs.com/eapil/camtellsvi-triton:1.0.0-jetson-orin-l4t39.2 \
  --output artifacts/releases/1.0.0-jetson-orin-l4t39.2
```

如使用另一个已准备好的本地 runtime，添加 `--runtime-image 仓库:版本` 或 `--runtime-image 仓库@sha256:...`。基础镜像必须是 Linux ARM64，并包含当前启动器所需的 Python 依赖和 Triton 三种后端。脚本在本地创建的 `jetson-release-runtime:<完整镜像ID>` 标签用于固定构建输入，不会推送该标签。

构建先根据 `release_models.json` 复制选中服务的完整依赖到独立快照，并生成仅包含这些 alias 的 `profiles/model_sets.json`，再验证、计算哈希并构建；不会写入开发模型目录或根 manifest。快照保持仓库相对结构，Dockerfile 仍以这一快照的根目录作为构建上下文。`release.json` 的 `model_sets` 记录本次实际打包的服务清单。构建结束前会在镜像内再次校验文件哈希，发现打包遗漏或内容不一致就失败。

构建需要额外空间保存模型快照与镜像层。失败的输出目录保留以供诊断，重试请选择新目录和新候选版本，不会自动删除已有制品。工作区有未提交变更时可以构建并验收本地候选；正式版本必须从已提交的源码重新构建。

生成目录：

```text
artifacts/releases/1.0.0-jetson-orin-l4t39.2/
├── context/            # 运行文件快照，不含 TensorRT 参考 ONNX
├── reference_onnx/     # TensorRT 参考 ONNX，仅由发布方归档
├── release.json        # 源码、运行环境、制品身份
├── build.json          # 候选标签、镜像 ID、默认 profile 和 runtime 节点数
└── deploy/             # 交付文件，构建时使用候选标签
    ├── compose.yaml
    ├── compose.gpu.yaml # 可选的显式 GPU profile；base 已默认启用
    └── .env
```

不要在快照生成期间修改源模型；不同源文件的业务一致性仍需通过模型维护和验收流程保证。`artifacts/` 已由 Git 忽略；真实 `.env` 也不提交到 Git。

## 4. 发布方：目标设备验收

```bash
./deploy/jetson/release.sh verify \
  --output artifacts/releases/1.0.0-jetson-orin-l4t39.2
```

默认验收 `orin_gpu`，直接使用实际交付的 base Compose，不依赖 GPU overlay 覆盖配置。验收会将 `compose.yaml` 和 `.env` 复制到仓库之外的临时目录，使用唯一 Compose 项目名、仅本机可访问的随机端口和独立日志卷启动同一个候选镜像。它不停止已有开发服务，不挂载宿主机源码或模型；但测试服务会额外占用 GPU 和系统内存，因此应在资源足够的维护时段执行。

自动验收内容包括：

- 镜像文件与内置 SHA256 清单一致，11 份 TensorRT 参考 ONNX 未进入镜像，容器使用构建时记录的镜像 ID。
- 交付 Compose 与构建时快照一致，实际 smoke profile、模型范围和 runtime 闭包数量与请求范围一致。
- pose/seg 实际配置为 `max_batch_size: 4`、2 个 GPU 实例、零额外等待的动态合批。
- Triton readiness 和所选依赖闭包中每个模型的 readiness。
- 推理节点配置为 GPU 实例。
- 全部所选公开入口的真实 HTTP 推理，覆盖 batch 1、2、3、4；三个 YOLO 同时覆盖七档图像尺寸。
- 输出名称、dtype、shape、batch、二进制长度及数值有限性。

输入使用程序生成的图像、文本和有效人脸关键点。此检查证明推理链路和部署契约可执行；模型精度、业务结果、性能与内存预算仍按 [模型接口与运维](模型接口与运维.md) 完成目标设备验收，并保存报告。

`--timeout 600` 设置等待服务就绪的秒数。调试时可使用 `--model-set sbs` 等子集，但完整发布镜像的正式推送要求最后一次验收为 `all`。默认 GPU 验收保存 `verification-orin_gpu.json` 与 `verification-orin_gpu.log`，并清理本次测试容器、网络和日志卷；清理失败会报告唯一项目名，供人工处理。验收不会常驻服务。

CPU 前后处理基线可用 `release.sh verify --output <候选目录> --model-config ""` 单独验收，保存 `verification.json` / `verification.log`。发布只接受同一镜像的默认 GPU `all` 报告，不能用 CPU 报告替代。旧候选缺少默认 profile 或 runtime 节点数记录时必须重建；镜像内嵌记录与外部 build 记录也必须一致。

## 5. 发布方：显式推送与交付

正式推送前，提交已审查的代码，用新的输出目录和版本重新构建并验收；确认没有缺失的 plan 构建记录，并准备目标设备的数值/业务验收报告。

```bash
docker login registry.cn-hangzhou.aliyuncs.com

./deploy/jetson/release.sh publish \
  --output artifacts/releases/1.0.0-jetson-orin-l4t39.2 \
  --acceptance-report /absolute/path/to/jetson-acceptance.txt
```

只有 `publish` 会推送镜像；`check`、`build`、`verify` 均不会推送。脚本在推送前核对候选身份、镜像内嵌的源码状态和构建记录、同一镜像的默认 `orin_gpu` 全量验收结果、验收报告及远程标签不存在。它保存报告及其 SHA256，但不会代替报告编写者判断业务精度是否合格。

成功后保存 `published.json`，并把交付目录 `.env` 的 `TRITON_IMAGE` 固定为推送所得的仓库 digest。将 `deploy/` 交付给部署人员即可（启动只需 base Compose 和 `.env`）；其余快照与报告由发布方留存。不要删除上一版镜像和发布记录。

## 6. 部署方：启动、升级与回滚

在任意部署目录放置交付的 `compose.yaml` 和 `.env`，根据需要修改 `.env` 的端口和 `TRITON_MODEL_SET`。无需复制仓库根目录 `.env`，也不需要设置 `PWD`。

```bash
docker login registry.cn-hangzhou.aliyuncs.com
docker compose config --quiet
docker compose pull
docker compose up -d --wait --wait-timeout 600
docker compose ps
docker compose logs --tail 100 triton-server
```

私有仓库通常只需首次登录。镜像来源与模型集合是必需的有效配置：示例文件未填写镜像时 Compose 会报错；空模型集合由原启动器明确拒绝。Shell 中同名变量会覆盖 `.env`，部署前应清理意外的 `TRITON_*`/`COMPOSE_*` 环境覆盖。

首次部署可以从 `deploy/jetson/compose.yaml` 和 `.env.example` 手动准备两份文件，但镜像值必须填写实际已发布版本。基础 runtime 镜像未内置模型，不能填作发布镜像。

升级前保留上一版 `.env`，特别是 `TRITON_IMAGE` 的 digest、`TRITON_MODEL_SET` 和端口。将镜像引用改成新版本，再执行 `pull` 和 `up -d --wait --wait-timeout 600`。普通发版沿用同一 Compose；若新增或改变部署配置项，应随发布说明交付对应的 Compose 模板。

回滚时恢复上一版配置，再执行相同命令。单实例升级、回滚均会重启并加载模型，存在服务中断；Compose 不会在健康检查失败后自动回滚。不要对 `/models` 添加挂载来替换模型，否则会失去完整镜像版本的可追溯性。

日常停止服务可用 `docker compose down`；不加 `--volumes` 时会保留日志卷。彻底离线部署还需预先导入镜像归档，两个配置文件本身不包含镜像内容。
