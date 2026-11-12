import logging
import uuid

import numpy as np
import triton_python_backend_utils as pb_utils


logger = logging.getLogger("face_qa_postprocess")


class InputValidationError(ValueError):
    pass


def _validate_quality(quality):
    if not isinstance(quality, np.ndarray):
        raise InputValidationError("quality must be a numpy array")
    if quality.ndim != 2 or quality.shape[1] != 10:
        raise InputValidationError("quality batch must have shape [B,10]")
    if quality.dtype != np.float32:
        raise InputValidationError("quality must have dtype float32")
    if quality.shape[0] == 0:
        raise InputValidationError("quality batch must be non-empty")
    for batch_index, row in enumerate(quality):
        if not np.isfinite(row).all():
            raise InputValidationError(
                f"batch index {batch_index}: quality values must be finite"
            )


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request_index, request in enumerate(requests):
            try:
                quality_tensor = pb_utils.get_input_tensor_by_name(
                    request, "quality"
                )
                if quality_tensor is None:
                    raise InputValidationError(
                        "missing required input quality"
                    )
                quality = quality_tensor.as_numpy()
                _validate_quality(quality)

                quality_score = np.mean(
                    quality, axis=1, dtype=np.float32
                ) * np.float32(10.0)
                quality_score = np.ascontiguousarray(
                    quality_score.reshape(quality.shape[0], 1),
                    dtype=np.float32,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("quality_score", quality_score)
                        ]
                    )
                )
            except InputValidationError as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            f"INVALID_INPUT: {exc}"
                        ),
                    )
                )
            except Exception:
                incident = uuid.uuid4().hex[:12]
                logger.exception(
                    "face_qa_postprocess internal failure "
                    "request_index=%d incident=%s",
                    request_index,
                    incident,
                )
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(
                            "INTERNAL_ERROR: face_qa_postprocess failed; "
                            f"incident={incident}"
                        ),
                    )
                )
        return responses

    def finalize(self):
        print("Cleaning up...")
