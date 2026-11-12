"""Export a FastReID SBS R50-IBN checkpoint to normalized ONNX."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "tools" / "models" / "sbs_r50_ibn" / "duke_sbs_R50-ibn.pth"
)
DEFAULT_OUTPUT = (
    ROOT / "model_repository" / "sbs_r50_ibn_onnx" / "1" / "model.onnx"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a5cd265ebee661edcc9e393b37009037998fe055ac05ec9c60497d8606f023ca"
)
EXPECTED_MODEL_KEY_COUNT = 418
EXPECTED_EMBEDDING_SIZE = 2048
DEFAULT_EXPECTED_NUM_CLASSES = 702
INPUT_HEIGHT = 384
INPUT_WIDTH = 128
OPSET_VERSION = 17
MINIMUM_COSINE_SIMILARITY = 0.9999


class NormalizedSBS(nn.Module):
    """Run the already-normalized image through FastReID without preprocessing twice."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model.backbone(image)
        embedding = self.model.heads(features)
        return F.normalize(embedding, p=2, dim=1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an SBS R50-IBN checkpoint to a normalized ONNX model."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
        help="Trusted SHA256 for the checkpoint; export aborts on mismatch.",
    )
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=DEFAULT_EXPECTED_NUM_CLASSES,
        help=(
            "Expected classifier row count in the training checkpoint "
            f"(default: {DEFAULT_EXPECTED_NUM_CLASSES} for DukeMTMC)."
        ),
    )
    args = parser.parse_args()
    if args.expected_num_classes <= 0:
        parser.error("--expected-num-classes must be positive")
    return args


def build_sbs_model(expected_num_classes: int) -> nn.Module:
    cfg = get_cfg()
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.META_ARCHITECTURE = "Baseline"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]

    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.LAST_STRIDE = 1
    cfg.MODEL.BACKBONE.FEAT_DIM = EXPECTED_EMBEDDING_SIZE
    cfg.MODEL.BACKBONE.NORM = "BN"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.WITH_SE = False
    cfg.MODEL.BACKBONE.WITH_NL = True
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = ""

    cfg.MODEL.HEADS.NAME = "EmbeddingHead"
    cfg.MODEL.HEADS.NORM = "BN"
    cfg.MODEL.HEADS.NUM_CLASSES = expected_num_classes
    cfg.MODEL.HEADS.EMBEDDING_DIM = 0
    cfg.MODEL.HEADS.WITH_BNNECK = True
    cfg.MODEL.HEADS.NECK_FEAT = "after"
    cfg.MODEL.HEADS.POOL_LAYER = "GeneralizedMeanPoolingP"
    cfg.MODEL.HEADS.CLS_LAYER = "CircleSoftmax"
    cfg.MODEL.HEADS.MARGIN = 0.35
    cfg.MODEL.HEADS.SCALE = 64
    cfg.INPUT.SIZE_TEST = [INPUT_HEIGHT, INPUT_WIDTH]
    cfg.freeze()
    model = build_model(cfg).eval()
    # The legacy checkpoint persists these buffers, while FastReID 1.4 marks
    # them non-persistent. Make them part of state_dict so all 418 keys are
    # consumed by one strict load instead of silently dropping either value.
    model._non_persistent_buffers_set.discard("pixel_mean")
    model._non_persistent_buffers_set.discard("pixel_std")
    return model


def load_checkpoint(
    path: Path, expected_num_classes: int
) -> Mapping[str, torch.Tensor]:
    # The trusted hash is checked before torch.load because this is a legacy pickle.
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError("Checkpoint must be a dictionary containing a 'model' state")
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint 'model' value must be a state dictionary")
    if len(state_dict) != EXPECTED_MODEL_KEY_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_KEY_COUNT} model keys, got {len(state_dict)}"
        )

    classifier = state_dict.get("heads.classifier.weight")
    if classifier is None or tuple(classifier.shape) != (
        expected_num_classes,
        EXPECTED_EMBEDDING_SIZE,
    ):
        raise RuntimeError(
            "Unexpected classifier shape: expected "
            f"({expected_num_classes}, {EXPECTED_EMBEDDING_SIZE}), got "
            f"{None if classifier is None else tuple(classifier.shape)}"
        )
    return state_dict


def remap_legacy_head_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Map only the two head naming changes between the checkpoint and 1.4.0."""

    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    remapped_count = 0
    for name, value in state_dict.items():
        if name.startswith("heads.bnneck."):
            name = "heads.bottleneck.0." + name.removeprefix("heads.bnneck.")
            remapped_count += 1
        elif name == "heads.classifier.weight":
            name = "heads.weight"
            remapped_count += 1
        if name in remapped:
            raise RuntimeError(f"Checkpoint key remapping collision: {name}")
        remapped[name] = value

    # BNNeck has weight, bias, running_mean, running_var and batch counter.
    if remapped_count != 6:
        raise RuntimeError(
            f"Expected exactly 6 legacy head keys to be remapped, got {remapped_count}"
        )
    if len(remapped) != EXPECTED_MODEL_KEY_COUNT:
        raise RuntimeError("Checkpoint key remapping changed the model key count")
    return remapped


def validate_torch_output(wrapper: nn.Module, example: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = wrapper(example)
    if tuple(output.shape) != (example.shape[0], EXPECTED_EMBEDDING_SIZE):
        raise RuntimeError(f"Unexpected PyTorch output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("PyTorch output contains NaN or infinite values")
    expected_norms = torch.ones(example.shape[0], dtype=output.dtype)
    if not torch.allclose(
        output.norm(p=2, dim=1), expected_norms, atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError("PyTorch output is not L2-normalized")
    return output.cpu().numpy()


def validate_onnx(
    output_path: Path, wrapper: nn.Module, examples: tuple[torch.Tensor, ...]
) -> None:
    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].name != "image":
        raise RuntimeError("Unexpected ONNX input name")
    if session.get_outputs()[0].name != "embedding":
        raise RuntimeError("Unexpected ONNX output name")

    for example in examples:
        expected = validate_torch_output(wrapper, example)
        actual = session.run(
            ["embedding"], {"image": example.cpu().numpy()}
        )[0]
        expected_shape = (example.shape[0], EXPECTED_EMBEDDING_SIZE)
        if tuple(actual.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected ONNX output shape: {tuple(actual.shape)}"
            )
        if actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise RuntimeError("ONNX output is not finite FP32")
        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"ONNX output is not L2-normalized: norms={norms}")

        cosine = np.sum(expected * actual, axis=1) / (
            np.linalg.norm(expected, axis=1) * norms
        )
        if float(np.min(cosine)) < MINIMUM_COSINE_SIMILARITY:
            raise RuntimeError(
                "PyTorch/ONNX cosine similarity is below "
                f"{MINIMUM_COSINE_SIMILARITY}: {cosine}"
            )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing SBS checkpoint: {checkpoint}")

    actual_checkpoint_sha256 = file_sha256(checkpoint)
    expected_checkpoint_sha256 = args.expected_sha256.strip().lower()
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            "Checkpoint SHA256 mismatch: expected "
            f"{expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )

    state_dict = remap_legacy_head_keys(
        load_checkpoint(checkpoint, args.expected_num_classes)
    )
    model = build_sbs_model(args.expected_num_classes)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    wrapper = NormalizedSBS(model).eval()

    torch.manual_seed(2026)
    example = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    validate_torch_output(wrapper, example)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(output),
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={
                "image": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

    validate_onnx(
        output,
        wrapper,
        (
            example,
            torch.randn(2, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32),
        ),
    )
    output_sha256 = file_sha256(output)
    print(f"Checkpoint SHA256: {actual_checkpoint_sha256}")
    print(f"Loaded model keys: {len(state_dict)}")
    print(f"Classifier classes: {args.expected_num_classes}")
    print(f"Exported: {output}")
    print(f"ONNX SHA256: {output_sha256}")


if __name__ == "__main__":
    main()
