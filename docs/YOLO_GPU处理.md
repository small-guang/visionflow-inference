# YOLO GPU 前后处理

[返回文档导航](../README.md#文档导航)

Jetson 的 `TRITON_MODEL_CONFIG=orin_gpu` 将 YOLO 前处理、TensorRT 推理和后处理放在同一个 Python backend 进程中执行。显式设置 `TRITON_MODEL_CONFIG=` 可选择 CPU 前后处理基线；模型推理仍使用 TensorRT GPU 后端。

三个公开入口保留原名称和输入输出契约，输入是 `RAW_IMAGE` uint8 BGR，支持 B1～4 及七档 letterbox 尺寸。pose / seg 可将已排队的同尺寸 RPC 合并，单次 TensorRT batch 不超过 4；face 保留逐请求执行。具体 tensor 和客户端分桶约束见[模型接口与运维](模型接口与运维.md)。

`scripts/yolo_request_batching.py` 负责校验、分组及响应拆分；`scripts/yolo_orin_pipeline.py` 负责请求生命周期和 GPU 执行；`scripts/yolo_orin_metrics.py` 提供指标。非法输入、取消、同组 GPU 故障的处理应在目标版本中分别验证。发生 GPU 错误时不会自动回退到 CPU 算法。

运行镜像需要匹配目标设备的 CUDA、CuPy、TensorRT 及 Python bindings。镜像构建和 engine 兼容性见 [Jetson 部署指南](Jetson部署指南.md)；改动镜像内脚本须重新构建镜像，仅重新加载模型不会更新脚本。

修改 GPU 实现或调度配置后，应使用具有授权的代表性图片，在真实设备对照 CPU 路径，验证检测框、关键点、mask 和输出顺序，再记录 batch、并发、端到端延迟及吞吐量。模型加载成功或 smoke 检查通过不能代替数值和性能验证。
