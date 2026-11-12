# Orin 图片与人脸 GPU 预处理

[返回文档导航](../README.md#文档导航)

Jetson 的 `TRITON_MODEL_CONFIG=orin_gpu` 将 CLIP、SBS、人脸关键点、质量、识别五个公开入口配置为同进程 Python/CUDA/TensorRT 融合执行。公开名称、输入输出 Tensor 契约、B≤4、原 TensorRT plan 与业务后处理保留。GPU 预处理结果直接绑定同一实例的 TensorRT context，仅将模型推理结果下载到 CPU，再生成公开响应。

本轮替代的是已经验收的“CUDA 预处理 → host Tensor → 原生 TensorRT”路径。2026-10-21 本机完成 v4 数值、混尺寸/坏请求压力与 C1/C16 性能验证，最终五个入口均为 2 实例。SBS/三个 face 入口的单实例版本虽然降低 CPU 时间/图，却在高并发下退化，因此不能只按复制次数或单请求延迟选择实例数。以下记录为直接 Triton 结果；生产切换与 ALog 全链路验证另行记录。

## 入口与模块职责

| 公开入口 | 复用的原生 plan 节点 | TRT 输入（省略 B） | TRT 输出（省略 B） | GPU 实例 / CPU 解码线程 | 最长组批等待 |
| --- | --- | --- | --- | --- | --- |
| `clip_ensemble_trt` | `clip_image_trt` | `image [3,224,224]` | `unnorm_image_features [512]` | 2 / 每实例 4 | 500 µs |
| `sbs_r50_ibn_image_ensemble` | `sbs_r50_ibn_onnx` | `image [3,384,128]` | `embedding [2048]` | 2 / 每实例 2 | 500 µs |
| `face_keypoint_2d106_ensemble` | `face_keypoint_2d106` | `data [3,192,192]` | `fc1 [212]` | 2 / 无压缩图片解码 | 250 µs |
| `face_qa_ensemble` | `face_qa` | `input [3,112,112]` | `quality [10]` | 2 / 无压缩图片解码 | 250 µs |
| `face_recognition_pipeline` | `w600k_r50_trt` | `input.1 [3,112,112]` | `683 [512]` | 2 / 无压缩图片解码 | 250 µs |

表中为内部 TRT Tensor，不改变公开输出：关键点仍为 `LANDMARKS_106 [B,106,2]`，QA 为 `quality_score [B,1]`，识别为 `EMBEDDING [B,512]`。五个入口均以 B4 为 preferred batch，单请求可为 B1～B4。CPU 根配置仍使用原 ensemble/pre/native/post 链路。

七份共享脚本各自承担一个职责：

- `image_decode.py`：持久、有界的 CPU 解码池；B1 直接解码，每实例最多处理一个 B≤4 窗口，错误保存在对应 slot。
- `image_orin_preprocess.py`：OpenCV 图片解码、缓存的小型 cubic 系数表，以及 CUDA resize、BGR→RGB、归一化、HWC→CHW。
- `face_orin_preprocess.py`：输入校验、小型 FP64 仿射矩阵/坐标表与 CUDA bilinear warp、边界填充、通道变换。
- `orin_trt_runtime.py`：固定空间维的 TensorRT runner；校验 named DEVICE/LINEAR/FP32 I/O 和 B1～B4 profile，一个实例持有一个 context，并预分配 B4 输出。
- `image_orin_pipeline.py`：CLIP/SBS 请求分组、并行解码、设备预处理、TRT、响应拆分与错误隔离。
- `face_orin_pipeline.py`：三个 face 入口的请求分组、仿射、同进程 TRT 及原业务后处理。
- `orin_pipeline_metrics.py`：统一请求、实际 TRT batch、阶段耗时、错误与逻辑传输字节指标。

CPU 图片解码和几何计算与 GPU 像素处理边界明确。线程池只执行 CPU 解码，不在解码线程中调用 CUDA。TRT runner 不替调用方同步；pipeline 持有输入、输出和临时 tensor，直到 `finally` 中的 stream 同步完成。

## 精度与调度边界

CLIP/SBS 保留 OpenCV ARM cubic 的定点系数和舍入规则，包括小型坐标表生成时的 FP32 融合乘加和像素计算的 NEON 运算顺序。不能直接用独立 NumPy FP32 乘加替代：部分尺寸会使 11-bit 系数相差 1，产生零星像素差异。系数表使用缓存的精确融合舍入计算，像素仍由 CUDA kernel 处理；测试覆盖 `884×764`、`676×779` 等边界尺寸。

人脸保留 OpenCV `INTER_LINEAR` 定点坐标、插值相位和黑色边界。关键点预处理保持 RGB 0～255，因为 plan 内已有归一化；QA 保持 `(pixel / 255 - 0.5) / 0.5` 的 FP32 运算顺序，识别保持 `(pixel - 127.5) * float32(1 / 127.5)`。FP64 仿射矩阵和关键点坐标逆变换保留。质量门槛、身份判定与 ALog 的整图→人脸→人型处理顺序未在这里调整。

同一次 Python `execute(requests)` 内按请求顺序组合，总 B 不超过 4，且不拆分一个请求。CLIP/SBS 可合并不同原图尺寸，CUDA 预处理用指针表读取各图片。三个 face 入口仅对原图 input 启用 `allow_ragged_batch`：同尺寸请求一次 warp；不同尺寸请求各自 warp 成固定形状，再在 GPU 上拼接为一次 TRT batch。bbox/landmark 不启用 ragged，每个原图请求本身仍是 dense tensor，相关输入的 B 必须一致。服务端跨尺寸调度行为必须由真实 Triton 请求验证，单测不能替代。

face 分组另限制原图总字节数为 `4 * 8_388_608 * 3`，超预算单请求在 GPU 工作前拒绝。坏 slot 使所属整个图片请求失败，其他有效请求继续合批；意外解码异常和 GPU 故障返回带 incident 的内部错误，不向客户端泄漏内部细节。响应分别构造，已取消或成功的结果不被相邻请求的响应构造异常覆盖。GPU 组失败后仍同步已提交工作，再处理后续组。

上一轮在本机 Orin / 对应 Triton 镜像中，`ipcGetMemHandle` 返回 `cudaErrorNotSupported`，Python backend 的 device 输出跨进程进入 TensorRT ensemble 也失败。旧路径因此显式下载预处理 Tensor，再交原生 TensorRT 上传；单节点 HTTP 可返回 GPU tensor 不代表跨进程传输可用。当前实现通过同进程绑定设备指针消除这段传输，并未启用 CUDA IPC。独立 pre 节点继续保留 host 输出，供兼容与诊断；默认融合 runtime 不加载它们。

## 配置、构建与生效

五个公开入口使用具名 `configs/orin_gpu.pbtxt`，例如 CLIP：

```protobuf
backend: "python"
max_batch_size: 4
parameters { key: "pipeline_backend" value { string_value: "orin_gpu" } }
parameters { key: "jpeg_decode_workers" value { string_value: "4" } }
instance_group [ { kind: KIND_GPU count: 2 } ]
dynamic_batching { preferred_batch_size: [4] max_queue_delay_microseconds: 500 }
```

单独设置 `KIND_GPU` 不会自动融合或将 OpenCV 像素运算变成 CUDA。模型初始化要求 backend 参数、GPU instance、API 版本和 Tensor 契约一致；不静默回退 CPU。独立 pre 节点的 GPU profile 使用 `preprocess_backend=orin_gpu`，与融合入口的 `pipeline_backend` 含义不同。

启动器只加载所选融合公开入口，原 manifest 中的完整 plan/helper 资源闭包仍校验并保留。根 `config.pbtxt` 保持 CPU 对照实现，显式 `TRITON_MODEL_CONFIG=` 选择该路径；这不同于本轮数值对照使用的已验收 GPU-host 基线镜像。

开发部署必须重建镜像：

```bash
docker compose -f docker-compose.jetson.yaml up -d --build
curl --fail http://127.0.0.1:8000/v2/health/ready
curl --fail http://127.0.0.1:8000/v2/models/clip_ensemble_trt/config
curl --fail http://127.0.0.1:8000/v2/models/face_recognition_pipeline/config
```

只检查实际选择的入口。返回应为 Python、`pipeline_backend=orin_gpu`、`KIND_GPU` 及预期 batching 配置；旧 pre/native 资源默认不单独 READY。七份共享脚本通过镜像 COPY 提供，改宿主文件、pbtxt 或 reload 不会更新镜像内脚本。独立发布冻结其内容并记录 SHA256；启动器拒绝缺失入口 wrapper/profile，smoke 校验实际 GPU 参数及 B1～B4 形状和有限输出。smoke 不替代数值正确性验收。

## 验证与性能口径

CPU 单测覆盖请求分组、坏输入/取消/内部异常隔离、设备引用生命周期、profile 和部署闭包：

```bash
python3 -m unittest test.test_image_decode test.test_image_orin_pipeline \
  test.test_face_orin_pipeline test.test_orin_trt_runtime \
  test.test_orin_pipeline_metrics test.test_orin_preprocessing_deployment \
  test.test_jetson_gpu_deployment test.test_benchmark_orin_fused
```

在与服务相同的 Orin CUDA/CuPy/OpenCV 环境中，可单独执行原 CUDA helper 的真实算子对照：

```bash
RUN_ORIN_GPU_TESTS=1 TRITON_RUN_GPU_TESTS=1 \
python3 -m unittest test.test_image_orin_preprocess test.test_face_orin_preprocess
```

`tools/validate_orin_preprocessing.py` 保留上一轮 CPU pre 与独立 GPU pre 的对照和真实裁图采集用途；它不是本轮融合入口持续压测器。本轮使用 `tools/benchmark_orin_fused.py`，baseline 为已验收的 GPU-host 镜像 `sha256:c213eee77b9af813e9d1d102306f615a5a93ec17a713d6befdff9a61e4bc6e78`，candidate 为本轮新构建的融合镜像。两侧 TRT 均在 GPU。服务使用隔离端口，本机基线为 18300、候选为 18400；工具不启动服务、不消费 Kafka、不写 Milvus 或对象存储。

示例要求事先准备两份实际服务身份 JSON，以及固定的人脸、人物 manifest。身份文件中的 `image_id` 必须是实际不可变镜像 ID，不得用手填标识替代运行状态取证。

```bash
python3 tools/benchmark_orin_fused.py \
  --dataset /path/to/datasets/test \
  --baseline-url http://127.0.0.1:18300 \
  --candidate-url http://127.0.0.1:18400 \
  --baseline-identity artifacts/fused/baseline-identity.json \
  --candidate-identity artifacts/fused/candidate-identity.json \
  --face-manifest artifacts/preprocess/faces.json \
  --person-manifest artifacts/preprocess/people.json \
  --clip-batch-diagnostic artifacts/fused/clip-batch-diagnostic.json \
  --exclude-invalid-images --phase all \
  --concurrency 1 16 --seconds 5 --window-repeats 2 \
  --output artifacts/fused/validation.json
```

数据默认要求 86 张完整原图、16 个人脸样本和 32 个人物裁图。完整解码与 SHA256 校验保留，默认损坏文件使验证失败；示例显式允许排除损坏图片，报告仍记录其身份和原因。SBS 人物裁图来自真实 pose bbox 并以 JPEG 编码，不含 ALog 分割掩码。不能放宽精度阈值或用随机输入替代业务样本；B4 重复填充槽位在报告中明示。

CLIP 原 plan 在同一图片的原生 B1 与 B2～B4 之间存在超出原阈值的数值差异；同实际 batch 下，预处理 Tensor、基线公开入口、候选公开入口及 native 输出的诊断对照精确一致。因此并发验收须使用同实际 native 总 batch 的完整请求参考，不能一律比较串行 B1。`--clip-batch-diagnostic` 需要真实诊断 JSON、其同名 `-summary.json` 与绑定的 NPZ；工具验证这些证据后构建参考库，不能凭空创建文件或放宽阈值。该参数和参考库身份也进入门禁绑定。

先完成全部 B1/B4 数值对照与并发混尺寸、坏请求隔离验证，再进入持续 C1/4/8/16 窗口。性能报告将正确性结果与实际镜像、配置、数据、工具身份绑定；任一请求错误或数值门禁失败均使窗口无效。分开执行时用 `--phase correctness` 保存门禁，再用 `--phase performance --correctness-report ...` 引用匹配报告。阶段参数与样本清单变化后不得复用过期门禁。

可选 `--baseline-pid` / `--candidate-pid` 指向同一 PID namespace 中真实服务主进程，用于完整进程树 CPU 统计；缺失时报告明确标记 CPU 指标不可用。默认 metrics 地址为 18302/18402 的 `/metrics`。客户端 CPU 时间与服务进程树 CPU 时间分别统计，配合设备侧 GPU、统一内存、EMC、温度和降频观测，避免只看 GPU 使用率瞬时峰值。

`orin_pipeline_requests_total` 统计入口请求，`orin_pipeline_trt_batches_total{batch_size}`、`orin_pipeline_group_requests_total{batch_size}` 和 `orin_pipeline_cross_request_groups_total` 说明实际提交的组批；`orin_pipeline_stage_seconds_total` / `stage_calls_total` 覆盖 decode、prepare、preprocess、warp、trt_download、postprocess；错误按 invalid/internal/cancelled 分类。阶段时间是 host wall time，不是 CUDA kernel elapsed。`orin_pipeline_copy_bytes_total{direction}` 是逻辑输入上传/最终输出下载字节，不含 driver 或 metadata 的内部流量。

HTTP 时间包含传输、输出物化和排队；GPU 结果下载、CPU 解码和后处理仍可能成为瓶颈。真实 helper 数值对照应记录候选脚本 SHA256、解码配置与来源，独立报告上传/分配/同步/下载范围，不能仅比较 kernel 执行时间。这里的 requests/s 或 images/s 是直接 Triton 口径；ALog 的目标 QPS、身份顺序和 Kafka/对象存储/Milvus 全链路仍需单独压测。

## 2026-10-21 本机 v4 验收

基线镜像为上述 `sha256:c213eee77b9af813e9d1d102306f615a5a93ec17a713d6befdff9a61e4bc6e78`，融合镜像为 `sha256:e50fe9446b7a573e88ef585241dff6d548535056c173ebb17da455a159b3b275`。v3/v4 镜像不变，v4 只将 SBS 与三个 face 入口的 profile count 从 1 调至 2。身份记录必须同时保留 `candidate-final-v4-launch.json`、`candidate-final-v4-source.json`、八个镜像脚本哈希（七份共享模块加启动器）及实际 plan 哈希。

`correctness-final-v4.json` 的 488 条成对检查与 20 条两侧压力条目通过；`performance-final-v4.json` 完成 80 个窗口，覆盖五模型、B1/B4、C1/C16、两侧各两轮，每窗口目标 5 秒。两报告绑定值为 `ad1a5b5d2b07af7d3758818f2ec1e763359a662a3ba5d455fec2484bcb82cf59`。以下为 C16 两轮按实际 wall 加权的图片吞吐、CPU 时间/图和 GPU 采样均值：

| 模型 / B | 基线→v4，图/s | 倍率 | 服务 CPU ms/图，基线→v4 | GPU 均值%，基线→v4 |
| --- | ---: | ---: | ---: | ---: |
| CLIP / 1 | 82.10 → 149.56 | 1.82 | 26.11 → 25.98 | 14.70 → 39.30 |
| CLIP / 4 | 82.18 → 166.25 | 2.02 | 25.10 → 24.89 | 29.00 → 59.45 |
| SBS / 1 | 403.62 → 501.48 | 1.24 | 4.31 → 3.79 | 53.70 → 70.80 |
| SBS / 4 | 455.11 → 588.33 | 1.29 | 3.22 → 3.12 | 59.00 → 67.20 |
| 关键点 / 1 | 515.99 → 651.91 | 1.26 | 4.63 → 3.02 | 52.40 → 33.70 |
| 关键点 / 4 | 985.38 → 1055.39 | 1.07 | 2.29 → 1.85 | 60.00 → 50.20 |
| QA / 1 | 505.78 → 730.85 | 1.44 | 4.09 → 2.83 | 37.78 → 32.70 |
| QA / 4 | 1140.16 → 1278.70 | 1.12 | 1.86 → 1.68 | 45.20 → 44.89 |
| 识别 / 1 | 370.20 → 488.42 | 1.32 | 4.64 → 3.32 | 88.00 → 63.67 |
| 识别 / 4 | 622.25 → 733.86 | 1.18 | 1.95 → 1.85 | 80.20 → 89.60 |

CLIP 收益主要体现为更高图片吞吐，而 CPU 时间/图变化较小；其余四模型双实例恢复了单实例在高并发下的退化。不同入口的 GPU 均值并非都升高：应结合吞吐、CPU 成本与延迟评估，不能把“利用率更高”单独当作目标。阶段重叠改善是解释假设，现有 host wall 指标不足以证明唯一原因。

v4 没有重跑 C4/C8；此前 v3 有完整 160 窗口矩阵。face B4 在当前样本分组下含重复图片，图/s 不等同于不同图片/s。1 秒粒度的 GPU 采样及整秒时间戳，对 5 秒窗口只能提供粗粒度观察。测量期间正式服务另有 CLIP/三个 YOLO 各 20 次、关键点 28 次、QA 1 次成功请求，时序未定位到单窗口，不能宣称设备独占；整段换页增加 249 页 swap-in、3724 页 swap-out（4096 字节/页），也不能归因于某个模型窗口。最终是否达到 ALog 50 QPS 或硬件饱和，仍需全链路与更长稳定负载验证。
