# Jetson 部署指南

[返回文档导航](../README.md#文档导航)

本指南说明 Jetson AGX Orin 开发部署。Jetson 需要与其 JetPack / L4T、GPU、CUDA、TensorRT 相匹配的 Triton 容器和模型 engine；ARM64 架构相同并不意味着这些版本可以互换。具体设备兼容性和实际推理必须在目标设备验证。

## 准备

1. 安装 Docker Compose，并配置 NVIDIA 容器运行环境。记录 Jetson 型号、JetPack / L4T、驱动、CUDA、TensorRT 和 Triton 镜像版本。
2. 阅读[模型来源与许可](模型来源与制品身份.md#许可证信息)，仅准备有权使用的模型；按[模型下载与转换](模型下载与转换.md)在目标环境生成并验证对应 engine。
3. 复制根目录 `.env.example` 为 `.env`，设置所需 `TRITON_MODEL_SET`。默认 `all` 要求准备全部模型及其依赖；可使用别名选择子集，具体参见[模型接口](模型接口与运维.md#23-11-个模型-alias-与闭包)。
4. 在 `.env` 中将 `TRITON_BASE_IMAGE` 设置为已针对目标 Jetson 的 JetPack / L4T、CUDA、TensorRT 及 Triton 后端验证兼容的基础镜像；没有通用默认值。检查 `Dockerfile.triton.jetson` 中依赖版本是否适合所选镜像，不能直接复用其他设备生成的 plan。

## 启动和检查

在仓库根目录运行：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build
docker compose -f docker-compose.jetson.yaml logs --tail 200 triton-server
curl -fsS http://localhost:8000/v2/health/live
curl -fsS http://localhost:8000/v2/health/ready
curl -fsS -X POST http://localhost:8000/v2/repository/index
```

仓库当前 Jetson 默认配置为 `TRITON_MODEL_CONFIG=orin_gpu`；显式设置为空可选择 CPU 前后处理基线，推理本身仍需要相应 GPU 后端。有关预处理、输出契约和验证要求，参见 [YOLO](YOLO_GPU处理.md)、[TBPS](TBPS_GPU处理.md) 和[图片与人脸](Orin图片与人脸GPU预处理.md)指南。

启动成功不等于全部模型可用：还需逐项检查选定模型的 `READY` 状态，并根据[模型接口与运维](模型接口与运维.md)运行真实请求，比较输出类型、维度与数值。若失败，检查服务日志、模型依赖闭包、engine 和运行库兼容性；变更配置或版本后重新验证。

例如，在已准备 `clip_text` 模型并启动服务时，按[文本推理请求示例](模型接口与运维.md#6-对外-ensemble-完整调用)发出实际请求，检查返回的特征类型、形状和有限值。独立交付镜像可使用 [Jetson 镜像发布](Jetson镜像发布.md)中的 `release.sh verify` 做目标设备 smoke 检查；两者都不能代替模型数值或性能验收。

## 升级与回滚

升级前保存已验证镜像的不可变 digest、`.env`、配置、模型及 plan 的来源和哈希，确认备份可读取并有足够空间；不要覆盖唯一一份可用制品。先在隔离环境构建和验证新版本，记录镜像、模型与配置的对应关系，再切换运行服务。出现问题时恢复已验证镜像和与其匹配的模型及配置，重新启动并检查模型 READY 和实际请求。TensorRT plan 随 GPU、运行库或模型版本变化可能不兼容，不能只恢复镜像或只恢复模型。

## 可选监控与发布

Triton 容器内 metrics 默认在 8002，Compose 映射至主机 8003。若启用 Prometheus，应从容器网络使用 `triton-server:8002`；实际端口以 Compose 配置为准。面向独立设备交付时参见 [Jetson 镜像发布](Jetson镜像发布.md)。
