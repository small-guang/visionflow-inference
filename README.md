# VisionFlow Inference

VisionFlow Inference 是 VisionFlow 的模型推理服务，使用 Triton Inference Server 管理模型和前后处理，通过 HTTP / gRPC 暴露检测、特征提取、姿态及分割接口。两套仓库组成视频关键帧分析流水线：VisionFlow 负责消费消息与编排，VisionFlow Inference 负责模型推理。

```text
视频关键帧 → VisionFlow → Triton 推理服务 → 检测 / 特征结果
                    └── 对象存储、Kafka 和向量数据库
```

## 主要能力

- 多模型服务：以模型集合控制需要加载的模型及依赖；支持目标检测、姿态、分割、人脸特征和图文特征等接口。
- 可复现的模型维护：记录来源、SHA256、转换配置和制品身份，支持从源权重导出 ONNX 并在目标设备上构建 TensorRT engine。
- 推理调度：按模型配置使用动态 batch；Jetson Orin 提供可选 GPU 前后处理路径。
- 可观测性：Triton 暴露健康状态和 Prometheus 指标；提供模型就绪检查和运维说明。

当前 Triton 根配置使用 `max_batch_size: 4`；TensorRT 原生节点加载 `model_throughput.plan`。模型更换或运行环境变更后需要重新验证 batch、精度和性能。

模型名、输入输出类型及维度以 [模型接口与运维](docs/模型接口与运维.md) 为准。`yolov8n_face`、`yolo26s_seg` 是兼容别名，实际模型大小见[来源清单](docs/模型来源与制品身份.md#来源清单)。

## 已落地：GPU 作品展示

[GPU 身份演示入口](gpu_demo/README.md)使用独立 Triton GPU Compose，加载 YOLO26m Pose、YuNet、SFace 和 OSNet。模型与许可均有固定来源和摘要，不要求许可未核实的其他身份模型。

本机 RTX3090 已验证15秒真实视频：225个输出帧，A→B→A分别确认为Demo-P001→Demo-P002→Demo-P001；无脸人形仅为候选。主仓 `VisionFlow/demo/gpu/` 保留最终视频、逐帧身份证据、登记图库与流程示意。

前后处理和身份决策运行在主机，神经网络使用GPU实例。NVIDIA运行库按自身条款使用，不宣称全栈开源。两个仓库同级下载后按演示入口复现；原根Compose与模型集合保留给完整多模型环境。

## 运行前必读

当前 Triton 部署需要 NVIDIA GPU、Docker Compose、可用的 NVIDIA 容器运行环境，以及按目标设备准备的模型制品。仓库不包含完整模型权重或预构建 engine，不能仅凭 `docker compose up` 完成模型准备。TensorRT engine 需要匹配目标 GPU、架构和运行库；Jetson 应在目标环境构建并验证。

仓库代码使用 [Apache-2.0](LICENSE)，**不代表第三方模型代码、权重、镜像及 GPU 运行库均采用该许可证**。部分权重有限制或许可待核实；在下载、使用或分发模型之前，逐项检查 [模型来源与许可](docs/模型来源与制品身份.md#许可证信息)。Triton 为开源服务，NVIDIA GPU 软件和容器中的其他组件适用各自条款。

## 快速开始：GPU 部署

1. 安装 Docker Compose、GPU 驱动和 NVIDIA 容器运行环境。确认镜像、CUDA、TensorRT 与实际设备兼容。
2. 按[模型下载与转换](docs/模型下载与转换.md)为需要的模型准备 ONNX 和 engine；先核实对应模型许可。可通过 `python -m tools.model_pipeline list` 查看目录。
3. 将 `.env.example` 复制为 `.env`，根据需要设置 `TRITON_MODEL_SET`；默认 `all` 会加载全部模型，需准备完整模型闭包。
4. 在仓库根目录启动：

   ```bash
   docker compose up -d --build
   ```

   Jetson 需要先在 `.env` 中显式设置经过设备兼容性验证的 `TRITON_BASE_IMAGE`，再使用 `docker compose -f docker-compose.jetson.yaml up -d --build`。请先阅读 [Jetson 部署指南](docs/Jetson部署指南.md)。

5. 检查服务和选定模型是否就绪：

   ```bash
   curl -fsS http://localhost:8000/v2/health/live
   curl -fsS http://localhost:8000/v2/health/ready
   curl -fsS -X POST http://localhost:8000/v2/repository/index
   ```

`live` 只表明进程存活；还需检查模型状态并执行实际推理请求。接口示例、批量和形状要求见[模型接口与运维](docs/模型接口与运维.md)。GPU 部署需要在目标环境验证，本文不把健康检查作为推理正确性的证明。

## 项目结构

```text
model_repository/          Triton 模型配置和前后处理
profiles/                  模型集合及依赖闭包
scripts/                   服务启动和运行时脚本
tools/model_pipeline/      模型来源、下载和转换工具
deploy/jetson/             Jetson 交付镜像工具
docs/                      模型、接口和部署文档
```

## 文档导航

| 主题 | 内容 |
| --- | --- |
| [模型下载与转换](docs/模型下载与转换.md) | 下载、导出及构建 TensorRT engine |
| [模型来源与制品身份](docs/模型来源与制品身份.md) | 模型来源、版本、摘要及许可证记录 |
| [模型接口与运维](docs/模型接口与运维.md) | 模型契约、调用示例、健康检查和排障 |
| [Jetson 部署指南](docs/Jetson部署指南.md) | Jetson 运行环境与验证 |
| [Jetson 镜像发布](docs/Jetson镜像发布.md) | 模型镜像构建和验收 |
| [模型维护工具说明](docs/模型维护工具说明.md) | 转换工具和缓存目录 |
| [YOLO GPU 处理](docs/YOLO_GPU处理.md)、[TBPS GPU 处理](docs/TBPS_GPU处理.md)、[图片与人脸 GPU 预处理](docs/Orin图片与人脸GPU预处理.md) | Orin 可选优化路径及约束 |

Triton HTTP / gRPC 接口未在此项目中提供认证或 TLS；请将服务限制在可信网络内，并按实际环境配置访问控制。
