# YOLO GPU 前后处理

Jetson AGX Orin 默认使用 `orin_gpu` 配置，将三个 YOLO 的 CUDA 前处理、TensorRT 推理和 CUDA 后处理放在同一 Python backend 进程、同一 CUDA stream 中。显式设置 `TRITON_MODEL_CONFIG=` 可选择 NumPy/OpenCV 前后处理基线；六个根配置前后处理节点显式使用 `KIND_CPU`，模型推理仍由 TensorRT 在 GPU 上执行。

`orin_gpu` 还支持 TBPS 图像的 CUDA 缩放与归一化，开启方法及资源要求见[TBPS 图像 GPU 前处理](TBPS_GPU处理.md)。CLIP、SBS 和其他未迁移模型维持原有路径。

## 在 Orin 开启与关闭

在 Orin 的项目根目录启动开发部署：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build
```

交付部署先在交付目录的 `.env` 设置有效的 `TRITON_IMAGE`，然后执行：

```bash
docker compose -f compose.yaml up -d
```

开发和交付 base Compose 及 Jetson 镜像均默认选择 `TRITON_MODEL_CONFIG=orin_gpu`；原 GPU overlay 仍可用于显式固定 GPU profile，已有调用无需删除。保留当前环境或 `.env` 中完整的 `TRITON_MODEL_SET`，原有服务选择继续有效；默认 `all` 保留全部模型服务。启动器在 GPU 模式为已选择的三个 YOLO 服务及 `tbps_clip_image` 加载公开包装器，不再加载它们独立的前处理、原生推理及 YOLO 后处理实例。其他服务保留原链路，plan 和 helper 目录仍须完整保留。

切换 CPU 前后处理基线时，在 `.env` 或 shell 显式设置 `TRITON_MODEL_CONFIG=`，只使用 base Compose 重新 `up -d`；GPU overlay 会强制恢复 `orin_gpu`。无需修改客户端：公开模型名、`RAW_IMAGE` uint8 BGR 输入、输出名称和 dtype、B1～4、七档 letterbox、阈值、排序、坐标及 packed mask 格式保持原契约。

## 优化路径

`configs/orin_gpu.pbtxt` 将三个 `*_ensemble` 公开入口选择为 Python GPU 包装器，`scripts/yolo_orin_pipeline.py` 直接读取对应版本的 TensorRT plan。原始图片在请求入口上传，预处理图像、检测输出和 mask proto 留在同一进程的 GPU 内存，最终公开结果下载后返回。该路径不依赖 Python backend 进程之间的 CUDA IPC。

当前 Orin GPU 配置为 face 4 个实例、pose 2 个实例、seg 2 个实例。每个实例各自持有 Python 进程、TensorRT engine/context、CUDA stream 和缓存；增加实例能让请求并行执行，也会增加内存占用和与其他模型的 GPU 竞争。`count` 不改变 B1～4 契约，也不代表跨请求自动合批。

2026-10-20 的本机 ALog 混合负载对照最终保留 pose 2 个实例。使用同一批历史 Kafka 输入，每轮完整处理 3000 张图片，count 1、下载保留线程 0 的完整吞吐为 16.4395 图/秒，count 2、下载保留线程 4 为 16.4377 图/秒，count 3、下载保留线程 4 为 16.4728 图/秒。三轮实质持平，每组仅一次长轮，未证明持续吞吐提升；投递速率 50 图/秒不是处理吞吐。最终采用 2 个实例，相比 3 个实例少一份 engine/context，而本次长轮吞吐差距约 0.21%。ALog 的下载保留线程降低了下载排队，但上述结果不支持宣称其带来稳定吞吐收益；count 与下载保留配置同时变化的对照也不能将全部差异单独归因于实例数。后续优化应比较整链吞吐和全部模型的排队、执行时间及内存余量，不能只看 pose 自身等待下降。

固定 4 张 `2560×1440` 图片的 B1/B4 并发对照中，count 2/3 与 count 1 的三个公开输出张量逐值相同，保持原检测顺序；该小样本检查不代表全部输入尺寸或身份结果等价。

### pose/seg 跨 RPC 合批

pose/seg 的具名配置开启 `dynamic_batching {}`，只合并已排队的同尺寸输入，不额外设置等待窗口；pose 仍为 2 个实例。Python backend 的一次 `execute(requests)` 将多个原请求按相同 H/W 分组，保持每个 RPC 的 B 连续且不拆开，总 B 不超过 4，再执行一次 CUDA→TensorRT→CUDA。face 保留逐请求执行。Triton 的动态调度只负责把请求交给 backend，真正减少 TensorRT 调用需要此处显式拼接和拆分，参见 [Python backend execute](https://github.com/triton-inference-server/python_backend/blob/main/README.md#execute) 和 [Dynamic Batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)。

职责分为三个模块：`scripts/yolo_request_batching.py` 负责纯 NumPy 校验、组计划、输入拼接和响应拆分；`scripts/yolo_orin_pipeline.py` 负责请求生命周期、单 CUDA stream 和 TensorRT runner；`scripts/yolo_orin_metrics.py` 负责固定标签集合的计数。输入先独立校验，非法请求不会加入共享组；每个原请求始终获得一个原位置的响应。执行前取消的请求会移出组；执行后取消只取消该请求的响应，同组其他响应继续返回。GPU 失败影响当前组并返回明确错误，后续组仍可处理；不自动重试或切换 CPU。GPU 临时数组保留到 `finally` 中的 stream 同步结束。

同 shape 合批不改变每张图片的检测顺序。pose 保留固定 100 slots；seg 按每个原 RPC 自身的 `max(1, max(NUM_DETECTIONS))` 恢复 slot 宽度，保持原有 zero padding 和 packed mask 字节。多个 RPC 的响应数组拥有独立连续存储；单 RPC 输入和已有连续独占输出无需额外拼接/拆分拷贝。每次只分配当前组的输入拼接数组，最多 `4×8,388,608×3 = 96 MiB`，与原单 RPC B4 的最大输入预算一致；完成后释放再处理下一组。该上限不包含 Triton 已接收的输入和待返回输出，并非容器总内存上限。GPU memory pool 仍可能缓存分配后的显存。

Prometheus 新增 `yolo_orin_trt_calls_total{batch_size="1".."4"}`，在实际 `execute_async_v3` 调用处计数；失败 enqueue 同样算一次 attempt，并增加 `yolo_orin_trt_failures_total`。`yolo_orin_group_failures_total` 表示已校验组执行失败。`yolo_orin_copy_bytes_total{direction="host_pack|host_split|input_h2d|output_d2h"}` 统计输入和最终输出的逻辑 tensor 拷贝字节，不包含 metadata、后处理中间下载或驱动流量，也不代表物理 PCIe/DRAM 总传输量；host pack/split 耗时由 `yolo_orin_host_copy_seconds_total` 累计。标签仅包含模型、版本、实例及固定 batch/direction。实际平均 TRT batch 应按各 B 计数加权计算，不能用 RPC 数替代 TensorRT 调用数。

该实现已通过纯 CPU 契约测试，包括 B1～4 分组、混合尺寸、非法输入/取消隔离、face 双输出、seg 原 RPC canonical slots、实际 enqueue 计数、stream 异常同步和组间内存释放。新增代码尚需候选镜像的真实 GPU 数值对照与整链回放；此前 count 1/2/3 的结果不能证明本轮合批收益。尤其 TensorRT 批大小变化可能选择不同计算路径，seg mask 必须继续逐 bit 验收，不能仅验证接口 shape。

共享 runtime 和两个新增模块都通过 Dockerfile COPY 到 `/usr/local/bin`；仅修改宿主脚本或重新 load 模型不会更新已运行镜像。验证应先从当前运行镜像构建增量候选，只加载 pose/seg，在独立端口与私有 IPC 下比较同一图片、不同 RPC 组合，并确认新增 B2～4 counter 实际增长。完整开发镜像及冻结发布的 snapshot/hash 清单均包含三个模块。回滚本轮需恢复旧镜像及 pose/seg 原具名调度配置；前述 pose count 仍可独立保留为 2。

在以 `--model-control-mode=explicit --model-config-name=orin_gpu` 运行的开发部署中，调整实例数只修改 `yolo26m_pose_ensemble/configs/orin_gpu.pbtxt`。先保存实际已加载配置，再通过 `/v2/repository/models/yolo26m_pose_ensemble/load` 应用仅含实例数量变化的配置，检查模型 ready、GET config 的实际数量和对应 Python backend 进程数，并核对其他已加载模型配置。HTTP `parameters.config` 覆盖只影响当前运行态，不能代替持久化具名配置。回滚时将此文件的 `count` 恢复为 `1`，以相同步骤应用并核验；默认 `config.pbtxt` 的 ensemble 链路保持原配置。仅改变实例配置时 Triton 支持更新模型实例，见 [Model Management](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html) 与 [Model Repository Extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_model_repository.html)。

| 部分 | GPU 计算 | CPU 工作 |
| --- | --- | --- |
| 三种预处理 | resize、letterbox、BGR→RGB、CHW、FP32 /255 融合 kernel | 小型插值系数表和 LETTERBOX_META |
| yolov8n_face | 框筛选、坐标恢复、NMS | 整批 score 表下载后用 NumPy 排序，保持原同分顺序 |
| yolo26m_pose | 小候选集筛选、稳定 top100、框与关键点坐标恢复融合 kernel | 输入预算和 metadata 校验 |
| yolo26s_seg | person 筛选、稳定 top100、坐标恢复融合 kernel；mask GEMM、crop 采样、阈值与打包 | 几何校验和确定可变输出 slot 数所需的小型 count |

pose/seg 在候选数不超过 1024 时使用融合选择 kernel，较大合法输入保留通用 GPU 路径。分割 GEMM 的 scratch 分块不超过 16 MiB；crop 保留 OpenCV 的 1/32 插值量化、BORDER_REPLICATE 和严格 `>0` 阈值。原始图片预算在上传前检查，CUDA/TRT 失败返回错误，不自动改用 CPU 算法。

## 镜像与发布验收

Orin 运行镜像需 CuPy、NVRTC、CUDA headers，以及与镜像原生 TensorRT 库匹配的 Python bindings。当前 `Dockerfile.triton.jetson` 使用 `cupy-cuda13x==13.6.0`、NumPy 1.26.4 和 TensorRT 10.16.1.11 的 `tensorrt-cu13-bindings` / `tensorrt-cu13`，以 `--no-deps` 安装 bindings，复用平台原生库。初始化会核对 Python 与原生 TensorRT 的 major/minor/patch，再加载 plan。

换用其他 CUDA/TensorRT 基础镜像时，需要同时覆盖 `CUPY_PACKAGE`、`TENSORRT_PYTHON_PACKAGE` 和 `TENSORRT_BINDINGS_PACKAGE`，并使用目标设备适配的 plan。不要通过安装通用依赖替换平台的 TensorRT/CUDA 库。plan 必须暴露原 FP32 I/O，并覆盖 B1～4 与全部七档输入尺寸。

已构建冻结候选镜像后，在目标 Orin 分别运行默认和 GPU 验收：

```bash
bash deploy/jetson/release.sh verify --output artifacts/releases/orin-candidate
bash deploy/jetson/release.sh verify --output artifacts/releases/orin-candidate --model-config ""
```

默认验收直接使用冻结的 base `compose.yaml`，检查实际 Python GPU 包装器及 pose/seg 的 B4、2 实例、零等待合批配置，并对所选服务执行 B1～4 smoke，其中 YOLO 使用七档尺寸，TBPS 图像使用编码图片输入。默认 GPU 报告为 `verification-orin_gpu.json`；显式 `--model-config ""` 的 CPU 基线报告为 `verification.json`。正式发布要求同一镜像的默认 GPU `all` 验收，按冻结 manifest 的实际 runtime 闭包数量检查覆盖；CPU 报告不能代替，缺少默认 profile/节点数记录的旧候选须重新构建。发布操作见[Jetson 镜像发布](Jetson镜像发布.md)。本次在 Orin 成功构建开发 Dockerfile，使用 `TRITON_BASE_IMAGE` 覆盖为已有 `26.04-py3-arm64-runtime`，并完成独立容器实测；默认 Dockerfile 的原始基础镜像未重新拉取构建，冻结发行镜像的硬件 `release.sh verify` 尚未执行。

## 真实图片对照与结果

在同一 Orin 启动独立 CPU 基线和 `orin_gpu` 候选，使用同一 TensorRT plan、相同本机 HTTP 路径，然后运行：

```bash
python3 tools/benchmark_yolo_gpu.py \
  --dataset-root /datasets \
  --cpu-url localhost:18010 --gpu-url localhost:18000 \
  --models face pose seg --warmup 3 --repeats 10 \
  --output artifacts/yolo-gpu-opt/orin-optimized-real-benchmark.json
```

基准工具不会启动或重新加载服务。默认选取 `D:/hdg/camtellsvi_alog/datasets` 的全部 19 张 test 图及 6 张 original，共 25 张唯一图片；记录 SHA256，按原尺寸组成 B1/B4，重复补齐的 slot 单独记录。读取 BGR，图片解码不计入推理耗时。

2026-10-15 的 Orin 实测：96 组、960 次 CPU/GPU 比较全部通过，包括 packed mask 严格逐 bit 一致。相对同次重测的 CPU 基线，HTTP P50 降低约 8.2%～29.1%；完整 P50/P95、执行统计和限制见[优化实测报告](../artifacts/yolo-gpu-opt/report.md)。三模型 B1～4 × 七档尺寸的 GPU smoke 和默认 CPU 回退 smoke 均完成 84 个 probe；构建后的开发镜像再次通过 84 个 GPU probe 及 31 项 CUDA 测试。

验收保持检测数和类别完全一致；框/关键点坐标绝对误差上限 0.25 pixel，confidence 上限 1e-4；mask 解包后逐 bit 比较，失败返回非零。新包装器的服务端 `compute_infer` 包含整条 CUDA→TRT→CUDA 链路及内部同步和最终下载，不能拆读为独立前后处理阶段或单 kernel 时间。顺序请求结果不能推断并发吞吐和 CPU 占用。

## 数值边界与历史结果

旧分拆 GPU 实验配置和调试产物已清理，GPU 启动统一使用 `orin_gpu`。2026-10-14 的测试中，跨 Python backend 进程的 GPU 张量传输在 Orin 报 `operation not supported`，在 Windows Docker/WSL 报 `invalid resource handle`；使用 CPU 张量连接后的分拆方案整体变慢。历史耗时摘要保留在[最终实测报告](../artifacts/yolo-gpu-opt/report.md#单独保留的历史运行)，不能与当前同次 CPU/GPU 对照混合计算加速比。

CPU BLAS 与 CUDA BLAS 的 FP32 累加顺序仍可能在接近零的 mask logit 处产生 bit 差异。历史随机 fixture 出现 CPU `-1.49e-08`、GPU `+4.47e-08`，RTX 3090 历史实测也有 1 bit 差异。本次 Orin 的 25 张图片严格通过不代表任意 FP32 输入均能逐 bit 相同；阈值未放宽，也未加入 CPU 数值纠正。
