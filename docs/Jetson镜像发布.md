# Jetson 内置模型镜像发布

[返回文档导航](../README.md#文档导航)

`deploy/jetson/` 提供将选定模型、运行配置和启动器打包到 Jetson 镜像的工具。模型和运行库有独立的许可及分发条件；构建或发布之前，应逐项确认对代码、模型权重和基础镜像的使用与再分发权限。

## 构建前准备

- 在目标 Jetson 环境准备并验证所选模型的 ONNX、TensorRT plan 及 provenance。plan 必须匹配 GPU 和运行库，不能假设跨设备通用。
- 检查 `deploy/jetson/release_models.json` 中的服务及依赖；运行时的 `TRITON_MODEL_SET` 只能选择已打包的服务。
- 准备可在目标设备运行的 ARM64 Triton runtime 镜像，并确认其中包括所需的 Triton 后端和 Python 依赖。
- 镜像版本应唯一，不要使用 `latest` 或 `stable`。构建脚本会记录镜像 ID、模型文件摘要及配置；这些记录不能替代数值和性能验证。

## 构建与验证

在仓库根目录执行，将示例镜像名称替换为自己有权使用的名称：

```bash
./deploy/jetson/release.sh check
./deploy/jetson/release.sh build \
  --runtime-image visionflow-inference:jetson-26.04 \
  --image example/visionflow-inference:1.0.0-jetson \
  --output artifacts/releases/1.0.0-jetson
./deploy/jetson/release.sh verify --output artifacts/releases/1.0.0-jetson
```

`check` 检查所选模型文件和身份记录。`build` 冻结必要输入并创建候选镜像；`verify` 在目标设备启动隔离部署并运行 smoke 检查。默认 GPU profile 的验收结果与 CPU 前后处理基线的验收结果应分别记录。实际性能和数值等价还需使用具有授权的数据集及代表性负载验证。

确认模型授权、候选镜像和验证结果后，可按 `./deploy/jetson/release.sh publish --help` 查看发布命令。推送由发布者显式触发；部署交付文件及报告位于所选输出目录。更多运行配置见 [Jetson 部署指南](Jetson部署指南.md)。
