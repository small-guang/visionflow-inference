# TBPS 图像 GPU 前处理

Jetson AGX Orin 的默认 `orin_gpu` 配置现支持 `tbps_clip_image`，公开入口仍为 `TBPS_clip_image_ensemble`。同一配置也保留已有三个 YOLO GPU 包装器。显式设置 `TRITON_MODEL_CONFIG=` 可使用原 CPU 前处理与 TensorRT 推理基线；CLIP、SBS、TBPS 文本及其他模型维持各自原有路径。

## 在 Orin 开启与关闭

开发部署在项目根目录执行：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build
```

交付部署在交付目录设置有效的 `TRITON_IMAGE` 后执行：

```bash
docker compose -f compose.yaml up -d
```

两个 base Compose 和 Jetson 镜像均默认选择 `TRITON_MODEL_CONFIG=orin_gpu`，GPU overlay 保留为显式 profile 入口。保留当前环境或 `.env` 中完整的 `TRITON_MODEL_SET`，不要将现有列表替换成单独的 `tbps_clip_image`。当前选择包含 `tbps_clip_image`（或使用 `all`）时，TBPS 图像入口会采用 GPU 包装器；其他已选择服务继续加载。该 alias 的完整资源链仍保留在 `profiles/model_sets.json`，无需修改清单。

切换 CPU 基线时显式设置 `TRITON_MODEL_CONFIG=`，只使用 base Compose 重新 `up -d`。客户端契约保持 `IMAGE_BYTES` BYTES `[B,1]` 输入、`image_feat` FP32 `[B,512]` 输出，支持 B1～4；编码图片可以混合尺寸组成一批。

## 执行与镜像资源

Pillow 在 CPU 解码原始图片为 RGB；CUDA 执行兼容 Pillow 的 BICUBIC 缩放到 224×224、FP32 归一化和 HWC→CHW，再在同一 Python backend 进程和 CUDA stream 中调用原生 TensorRT plan。最终特征下载后返回。CUDA/TRT 错误会返回错误，不自动改用 CPU 前处理。

`TBPS_clip_image_ensemble/configs/orin_gpu.pbtxt` 选择 GPU 包装器，`scripts/tbps_orin_pipeline.py` 加载 `TBPS_clip_image_preprocess/1/gpu.py`，并按 `TBPS_clip_image_trt/config.pbtxt` 读取原生 plan。GPU 启动只加载公开包装器，不再单独加载 TBPS CPU 前处理或原生 TensorRT 实例；这些模型目录及 plan 仍必须完整保留，供包装器读取。开发镜像包含运行模块，模型辅助文件由现有仓库挂载提供；冻结交付镜像同时打包二者。依赖沿用已有 Orin CuPy、Pillow 和 TensorRT 运行环境。

## 评测与正式验证

此前的[隔离评测报告](../artifacts/image-gpu-eval/report.md)使用 25 张不同 SHA256 真实图片，TBPS GPU 缩放候选的 B1/B4 HTTP P50 相对同次 CPU 基线分别降低 37.3% / 40.8%，所测特征逐元素相同。这是接入前的候选结果；串行延迟不能代表多模型并发吞吐，样本上的特征一致也不能证明下游检索准确率完全一致。

本次正式入口已在 Orin 独立容器通过验证：25 张真实图片的 B1～4 特征与 CPU 路径逐元素一致，B1/B4 HTTP P50 分别为 55.66→35.55 ms、229.30→133.89 ms，降低 36.1% / 41.6%。该次评测当时保留默认 CPU，生产服务未重启；详见[正式接入报告](../artifacts/tbps-orin-gpu/report.md)。同步代码到 Orin 后需重新构建启动。

冻结发行镜像默认验收 `orin_gpu`，GPU 报告保存为 `verification-orin_gpu.json`，同一镜像的默认 GPU `all` 验收是发布要求。可另用 `--model-config ""` 验证 CPU 基线并保存 `verification.json`，该报告不能代替默认 GPU 发布验收。命令与发布流程见[YOLO GPU 前后处理](YOLO_GPU处理.md)及[Jetson 镜像发布](Jetson镜像发布.md)。
