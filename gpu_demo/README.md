# GPU identity video example

This final profile runs YOLO pose, YuNet face detection, SFace face embeddings
and OSNet person embeddings through an isolated Triton GPU service. The
companion VisionFlow example registers local demo labels, confirms subsequent
face matches and associates body candidates. Face and body vectors use separate
embedding spaces.

## Prepare and run

Install `gpu_demo/requirements.txt` in a Python environment. Use
`gpu_demo/requirements-test.txt` to also run the Triton protobuf regressions.

```powershell
python gpu_demo/prepare_identity.py --source-repository /path/to/authorized/pose/model_repository --osnet-onnx /path/to/authorized/osnet/model.onnx
python -m unittest gpu_demo.test_prepare gpu_demo.test_prepare_identity -v
docker build -f gpu_demo/Dockerfile -t visionflow-gpu-demo:24.12 .
$env:TRITON_GPU_IMAGE = 'visionflow-gpu-demo:24.12'
docker compose -f gpu_demo/compose.identity.yaml up -d
```

The pose source is read only. Its ONNX, native TensorRT plan, embedded AGPL
license, license text and GPU configuration must match the approved identities.
`prepare.py` is a shared validation/collection library; it does not publish a
pose-only staging repository. `prepare_identity.py` publishes the complete final
`identity_model_repository/` directly and records `identity-model-manifest.json`.

After initial preparation, the final repository and retained license files can
be reused offline, with the same checks and without a download cache:

```powershell
python gpu_demo/prepare_identity.py --source-repository /path/to/authorized/pose/model_repository --offline
```

The existing final OSNet file is used if `--osnet-onnx` is omitted. Unknown
weights are rejected. OpenCV Zoo downloads are bound to the fixed commit and
Git LFS SHA256 values in `identity-source-manifest.json`; the official OpenCV
Hugging Face mirror is accepted only when its bytes match those identities.
`--proxy`, `--local-ports` and `--cache-dir` are optional acquisition settings.
`--model-repository` selects an existing verified identity repository as an
input, and `--output-dir` can build an independent verification copy.

The Compose project is `vf-open-gpu-identity`; HTTP/gRPC/metrics listen on
localhost ports `18010/18011/18012`. Its seven nodes are the four-node pose
closure plus `yunet_gpu`, `sface_gpu` and `osnet_gpu`. The committed image default
is the official Triton24.12 image. The supplied derivative Dockerfile adds the
OpenCV/NumPy dependencies used by the existing Python pose preprocessing. A
verified compatible image can be selected through `TRITON_GPU_IMAGE`.

## Contracts

| Model | Input | Output |
| --- | --- | --- |
| `yolo26m_pose_ensemble` | `RAW_IMAGE` UINT8 `[B,H,W,3]`, BGR | `DETECTIONS` FP32 `[B,100,6]`; `KEYPOINTS` FP32 `[B,100,17,3]`; `NUM_DETECTIONS` INT32 `[B,1]` |
| `yunet_gpu` | `input` FP32 `[1,3,640,640]`, BGR raw0..255 | `cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`, `kps_8/16/32` |
| `sface_gpu` | `data` FP32 `[1,3,112,112]`, aligned RGB raw0..255 | `fc1` FP32 `[1,128]`; L2-normalize before cosine matching |
| `osnet_gpu` | `image` FP32 `[B,3,256,128]`, RGB/ImageNet normalization | `embedding` FP32 `[B,512]`, L2-normalized |

Pose and OSNet accept batch1..4; images within a pose request have the same H/W.
Only the first `NUM_DETECTIONS` pose slots are valid; boxes use original-image
xyxy coordinates followed by score/class. YuNet stride8/16/32 outputs have
6400/1600/400 rows; cls/obj have one channel, bbox four, and kps ten. SFace uses
static batch1 and includes `(x-127.5)/128` inside its graph. OSNet preprocessing
is RGB, bilinear resize128x256, divide255, ImageNet mean/std, NCHW.

## Evidence and licenses

`gpu-validation.json` records the actual isolated GPU video replay and binds the
model manifest used for that run. Repreparing identical model bytes does not
constitute a new GPU validation. The companion VisionFlow example retains the
video, event evidence and result manifest referenced by that receipt.

The pose model uses native TensorRT with no CPU model fallback. The identity
models request native ONNX Runtime CUDA execution on GPU0. Their configuration
alone does not prove every operator's placement; stock Triton24.12 does not
expose a generic `session.disable_cpu_ep_fallback` parameter. Existing Python
pose preprocessing/postprocessing run on CPU.

Necessary source and permission snapshots remain under `licenses/`. YOLO
weights/source retain AGPL-3.0; YuNet MIT; SFace Apache-2.0; OSNet weights retain
the author's pinned MIT model-card evidence. The parent Apache license does
not replace these terms. The models' training datasets are not bundled or
relicensed. CUDA, TensorRT, NVIDIA images and OpenCV binary dependencies retain
their own terms; this is an open-source application with a NVIDIA GPU runtime.

Demo labels are local gallery labels, not civil identities. Body similarity is
a candidate association signal and must not upgrade an unconfirmed person to a
confirmed identity without the face-anchor rules.
