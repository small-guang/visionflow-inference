from pathlib import Path

import torch
from torch import nn

from .onnx_export_validation import publish_onnx_atomically, validate_onnx_output


EXPORT_SEED = 20260826


class NormalizedImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


class NormalizedTextEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_text(text)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


def _make_image_export_example() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(EXPORT_SEED)
    return torch.randn(
        (1, 3, 224, 224),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


def _make_distinct_b2(inputs: torch.Tensor) -> torch.Tensor:
    second = inputs.clone()
    flattened = second.reshape(1, -1)
    position = min(1, flattened.shape[1] - 1)
    if torch.is_floating_point(second):
        delta = max(float(second.detach().abs().max().item()), 1.0)
        flattened[0, position] += delta
    else:
        flattened[0, position] = 0 if flattened[0, position].item() else 1
    return torch.cat((inputs, second), dim=0)


def export_one(
    model: nn.Module,
    inputs: torch.Tensor,
    output_path: Path,
    input_name: str,
    output_name: str,
) -> None:
    if inputs.ndim < 1 or inputs.shape[0] != 1:
        raise ValueError("export example must contain exactly one batch item")

    validation_inputs = _make_distinct_b2(inputs)
    with torch.no_grad():
        expected_b1 = model(inputs).detach().cpu()
        expected_b2 = torch.cat(
            [model(row) for row in validation_inputs.split(1)],
            dim=0,
        ).detach().cpu()
    if torch.allclose(expected_b2[0], expected_b2[1], rtol=1e-4, atol=1e-5):
        raise RuntimeError("B2 validation samples must produce distinct outputs")

    def build_and_validate(candidate_path: Path) -> None:
        torch.onnx.export(
            model,
            inputs,
            str(candidate_path),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                input_name: {0: "batch"},
                output_name: {0: "batch"},
            },
            dynamo=False,
        )
        validate_onnx_output(
            candidate_path,
            input_name,
            output_name,
            inputs.detach().cpu().numpy(),
            expected_b1.numpy(),
        )
        validate_onnx_output(
            candidate_path,
            input_name,
            output_name,
            validation_inputs.detach().cpu().numpy(),
            expected_b2.numpy(),
        )

    publish_onnx_atomically(output_path, build_and_validate)
