# Orin 图片与人脸 GPU 预处理

[返回文档导航](../README.md#文档导航)

`TRITON_MODEL_CONFIG=orin_gpu` 可将 CLIP、SBS、人脸关键点、质量和识别入口的 CUDA 预处理与 TensorRT 推理放在同一 Python backend 进程。公开 Tensor 契约和 B1～4 批量约束见[模型接口与运维](模型接口与运维.md)。

| 模块 | 职责 |
| --- | --- |
| `scripts/image_decode.py` | 有界 CPU 图片解码 |
| `scripts/image_orin_preprocess.py`、`scripts/face_orin_preprocess.py` | 图片和人脸 GPU 预处理 |
| `scripts/orin_trt_runtime.py` | TensorRT context 和输入输出绑定 |
| `scripts/image_orin_pipeline.py`、`scripts/face_orin_pipeline.py` | 请求分组、推理、响应拆分与错误处理 |
| `scripts/orin_pipeline_metrics.py` | 请求、组批、阶段耗时和传输指标 |

图像解码和几何计算与 GPU 像素处理分开；请求可在满足输入与字节预算约束时分组。GPU 错误不会静默回退。原 plan 和辅助资源仍需保留，具名 profile 和相关脚本发生变化时需要重建镜像。

部署方法和设备兼容性见 [Jetson 部署指南](Jetson部署指南.md)。CPU 测试可覆盖请求分组和错误处理，但不能代替真实设备的 CUDA/TensorRT 数值比较。修改模型、engine、预处理或实例配置后，应使用有权使用的代表性样本进行数值对照、坏请求隔离与持续负载测试。记录硬件、镜像、模型摘要、batch、并发和测量范围；直接 Triton 推理吞吐不等同于 VisionFlow 全链路吞吐。
