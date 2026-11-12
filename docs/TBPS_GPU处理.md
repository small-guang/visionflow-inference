# TBPS 图像 GPU 前处理

[返回文档导航](../README.md#文档导航)

Jetson 的 `TRITON_MODEL_CONFIG=orin_gpu` 可选择 `TBPS_clip_image_ensemble` 的 GPU 前处理。Pillow 在 CPU 解码图片，CUDA 执行缩放和归一化，然后在同一进程调用 TensorRT；模型输出仍为 `image_feat` FP32 `[B,512]`，输入为 `IMAGE_BYTES` BYTES `[B,1]`，支持 B1～4。

显式设置 `TRITON_MODEL_CONFIG=` 选择 CPU 前处理基线；推理仍需要兼容的 GPU 和 TensorRT plan。GPU 路径由 `scripts/tbps_orin_pipeline.py` 加载预处理和 plan，模型资源目录仍需完整保留，CUDA / TensorRT 故障不会自动回退。具体部署见 [Jetson 部署指南](Jetson部署指南.md)。

更换模型、设备或运行库后，请用获得授权的代表性图片对照 CPU 预处理与 GPU 路径的数值结果，再测量延迟、吞吐与资源占用。健康检查不能证明下游检索效果一致。
