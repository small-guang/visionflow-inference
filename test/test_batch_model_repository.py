import json
import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1] / "model_repository"

ALOG_PUBLIC_MODELS = frozenset(
    {
        "TBPS_clip_image_ensemble",
        "clip_ensemble_trt",
        "face_keypoint_2d106_ensemble",
        "face_qa_ensemble",
        "face_recognition_pipeline",
        "sbs_r50_ibn_image_ensemble",
        "sixdrepnet_ensemble",
        "yolo26m_pose_ensemble",
        "yolo26s_seg_ensemble",
        "yolov8n_face_ensemble",
    }
)


MODEL_IO_CONTRACTS = {
    "adaface_ir50_webface4m_onnx": {
        "inputs": (("input", "TYPE_FP32", (3, 112, 112)),),
        "outputs": (("embedding", "TYPE_FP32", (512,)),),
    },
    "adaface_preprocess": {
        "inputs": (
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("LANDMARK", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("input", "TYPE_FP32", (3, 112, 112)),),
    },
    "adaface_recognition_pipeline": {
        "inputs": (
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("LANDMARK", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("EMBEDDING", "TYPE_FP32", (512,)),),
    },
    "clip_ensemble_trt": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("unnorm_image_features", "TYPE_FP32", (512,)),),
    },
    "clip_image_preprocess": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("image", "TYPE_FP32", (3, 224, 224)),),
    },
    "clip_image_trt": {
        "inputs": (("image", "TYPE_FP32", (3, 224, 224)),),
        "outputs": (("unnorm_image_features", "TYPE_FP32", (512,)),),
    },
    "clip_text_ensemble": {
        "inputs": (("TEXT", "TYPE_STRING", (1,)),),
        "outputs": (("unnorm_text_features", "TYPE_FP32", (512,)),),
    },
    "clip_text_preprocess": {
        "inputs": (("TEXT", "TYPE_STRING", (1,)),),
        "outputs": (("INPUT_IDS", "TYPE_INT64", (52,)),),
    },
    "clip_text_trt": {
        "inputs": (("text", "TYPE_INT64", (52,)),),
        "outputs": (("unnorm_text_features", "TYPE_FP32", (512,)),),
    },
    "face_keypoint_2d106": {
        "inputs": (("data", "TYPE_FP32", (3, 192, 192)),),
        "outputs": (("fc1", "TYPE_FP32", (212,)),),
    },
    "face_keypoint_2d106_ensemble": {
        "inputs": (
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("BBOX", "TYPE_FP32", (4,)),
        ),
        "outputs": (("LANDMARKS_106", "TYPE_FP32", (106, 2)),),
    },
    "face_keypoint_2d106_post": {
        "inputs": (
            ("RAW_OUTPUT", "TYPE_FP32", (212,)),
            ("AFFINE_MATRIX", "TYPE_FP64", (2, 3)),
        ),
        "outputs": (("LANDMARKS_106", "TYPE_FP32", (106, 2)),),
    },
    "face_keypoint_2d106_pre": {
        "inputs": (
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("BBOX", "TYPE_FP32", (4,)),
        ),
        "outputs": (
            ("PREPROCESSED_IMAGE", "TYPE_FP32", (3, 192, 192)),
            ("AFFINE_MATRIX", "TYPE_FP64", (2, 3)),
        ),
    },
    "face_qa": {
        "inputs": (("input", "TYPE_FP32", (3, 112, 112)),),
        "outputs": (("quality", "TYPE_FP32", (10,)),),
    },
    "face_qa_ensemble": {
        "inputs": (
            ("RAW_FACE_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("LANDMARK", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("quality_score", "TYPE_FP32", (1,)),),
    },
    "face_qa_postprocess": {
        "inputs": (("quality", "TYPE_FP32", (10,)),),
        "outputs": (("quality_score", "TYPE_FP32", (1,)),),
    },
    "face_qa_pre": {
        "inputs": (
            ("RAW_FACE_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("LANDMARK", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("QA_PREPROCESSED_IMAGE", "TYPE_FP32", (3, 112, 112)),),
    },
    "face_rec_pre": {
        "inputs": (
            ("raw_image", "TYPE_UINT8", (-1, -1, 3)),
            ("landmark", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("processed_input", "TYPE_FP32", (3, 112, 112)),),
    },
    "face_recognition_pipeline": {
        "inputs": (
            ("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),
            ("LANDMARK", "TYPE_FP32", (5, 2)),
        ),
        "outputs": (("EMBEDDING", "TYPE_FP32", (512,)),),
    },
    "osnet_ain_x1_0_image_ensemble": {
        "inputs": (("IMAGE_BYTES", "TYPE_UINT8", (-1,)),),
        "outputs": (("embedding", "TYPE_FP32", (512,)),),
    },
    "osnet_ain_x1_0_image_preprocess": {
        "inputs": (("IMAGE_BYTES", "TYPE_UINT8", (-1,)),),
        "outputs": (("image", "TYPE_FP32", (3, 256, 128)),),
    },
    "osnet_ain_x1_0_onnx": {
        "inputs": (("image", "TYPE_FP32", (3, 256, 128)),),
        "outputs": (("embedding", "TYPE_FP32", (512,)),),
    },
    "sbs_r50_ibn_image_ensemble": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("embedding", "TYPE_FP32", (2048,)),),
    },
    "sbs_r50_ibn_image_preprocess": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("image", "TYPE_FP32", (3, 384, 128)),),
    },
    "sbs_r50_ibn_onnx": {
        "inputs": (("image", "TYPE_FP32", (3, 384, 128)),),
        "outputs": (("embedding", "TYPE_FP32", (2048,)),),
    },
    "sixdrepnet_ensemble": {
        "inputs": (("RAW_FACE_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (("HEAD_POSE", "TYPE_FP32", (3,)),),
    },
    "sixdrepnet_onnx": {
        "inputs": (("image", "TYPE_FP32", (3, 224, 224)),),
        "outputs": (("rotation_matrix", "TYPE_FP32", (3, 3)),),
    },
    "sixdrepnet_postprocess": {
        "inputs": (("ROTATION_MATRIX", "TYPE_FP32", (3, 3)),),
        "outputs": (("HEAD_POSE", "TYPE_FP32", (3,)),),
    },
    "sixdrepnet_preprocess": {
        "inputs": (("RAW_FACE_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (("PREPROCESSED_IMAGE", "TYPE_FP32", (3, 224, 224)),),
    },
    "TBPS_clip_image_ensemble": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("image_feat", "TYPE_FP32", (512,)),),
    },
    "TBPS_clip_image_preprocess": {
        "inputs": (("IMAGE_BYTES", "TYPE_STRING", (1,)),),
        "outputs": (("image", "TYPE_FP32", (3, 224, 224)),),
    },
    "TBPS_clip_image_trt": {
        "inputs": (("image", "TYPE_FP32", (3, 224, 224)),),
        "outputs": (("image_feat", "TYPE_FP32", (512,)),),
    },
    "TBPS_clip_text_ensemble": {
        "inputs": (("TEXT_IN", "TYPE_STRING", (1,)),),
        "outputs": (("text_feat", "TYPE_FP32", (512,)),),
    },
    "TBPS_clip_text_preprocess": {
        "inputs": (("TEXT_IN", "TYPE_STRING", (1,)),),
        "outputs": (("TEXT_TOKENS", "TYPE_INT64", (77,)),),
    },
    "TBPS_clip_text_trt": {
        "inputs": (("text", "TYPE_INT64", (77,)),),
        "outputs": (("text_feat", "TYPE_FP32", (512,)),),
    },
    "w600k_r50_trt": {
        "inputs": (("input.1", "TYPE_FP32", (3, 112, 112)),),
        "outputs": (("683", "TYPE_FP32", (512,)),),
    },
    "yolo26m_pose_ensemble": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (100, 6)),
            ("KEYPOINTS", "TYPE_FP32", (100, 17, 3)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolo26m_pose_onnx": {
        "inputs": (("images", "TYPE_FP32", (3, -1, -1)),),
        "outputs": (("output0", "TYPE_FP32", (-1, 57)),),
    },
    "yolo26m_pose_postprocess": {
        "inputs": (
            ("RAW_OUTPUT", "TYPE_FP32", (-1, 57)),
            ("LETTERBOX_META", "TYPE_FP32", (7,)),
        ),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (100, 6)),
            ("KEYPOINTS", "TYPE_FP32", (100, 17, 3)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolo26m_pose_preprocess": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("PREPROCESSED_IMAGE", "TYPE_FP32", (3, -1, -1)),
            ("LETTERBOX_META", "TYPE_FP32", (7,)),
        ),
    },
    "yolo26s_seg_ensemble": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (-1, 6)),
            ("MASK_CROPS_PACKED", "TYPE_UINT8", (-1, 160, 20)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolo26s_seg_onnx": {
        "inputs": (("images", "TYPE_FP32", (3, -1, -1)),),
        "outputs": (
            ("output0", "TYPE_FP32", (-1, 38)),
            ("output1", "TYPE_FP32", (32, -1, -1)),
        ),
    },
    "yolo26s_seg_postprocess": {
        "inputs": (
            ("RAW_OUTPUT", "TYPE_FP32", (-1, 38)),
            ("MASK_PROTO", "TYPE_FP32", (32, -1, -1)),
            ("LETTERBOX_META", "TYPE_FP32", (7,)),
        ),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (-1, 6)),
            ("MASK_CROPS_PACKED", "TYPE_UINT8", (-1, 160, 20)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolo26s_seg_preprocess": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("PREPROCESSED_IMAGE", "TYPE_FP32", (3, -1, -1)),
            ("LETTERBOX_META", "TYPE_FP32", (7,)),
        ),
    },
    "yolov8n_face_ensemble": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (100, 6)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolov8n_face_onnx": {
        "inputs": (("images", "TYPE_FP32", (3, -1, -1)),),
        "outputs": (("output0", "TYPE_FP32", (5, -1)),),
    },
    "yolov8n_face_postprocess": {
        "inputs": (
            ("RAW_OUTPUT", "TYPE_FP32", (5, -1)),
            ("LETTERBOX_META", "TYPE_FP32", (5,)),
        ),
        "outputs": (
            ("DETECTIONS", "TYPE_FP32", (100, 6)),
            ("NUM_DETECTIONS", "TYPE_INT32", (1,)),
        ),
    },
    "yolov8n_face_preprocess": {
        "inputs": (("RAW_IMAGE", "TYPE_UINT8", (-1, -1, 3)),),
        "outputs": (
            ("PREPROCESSED_IMAGE", "TYPE_FP32", (3, -1, -1)),
            ("LETTERBOX_META", "TYPE_FP32", (5,)),
        ),
    },
}


class TextprotoScannerTests(unittest.TestCase):
    def test_strip_textproto_comment_ignores_markers_inside_quoted_strings(self):
        stripper = globals().get("_strip_textproto_comment")
        self.assertTrue(callable(stripper), "comment scanner helper must exist")
        if not callable(stripper):
            return

        self.assertEqual(
            'name: "https://example.test/#fragment" ',
            stripper('name: "https://example.test/#fragment" # trailing comment'),
        )
        self.assertEqual(
            'value: "escaped \\\" // still quoted" ',
            stripper('value: "escaped \\\" // still quoted" // trailing comment'),
        )

    def test_dynamic_batching_field_matches_only_real_field_declarations(self):
        matcher = globals().get("_is_dynamic_batching_field")
        self.assertTrue(callable(matcher), "dynamic batching matcher helper must exist")
        if not callable(matcher):
            return

        for line in (
            "dynamic_batching {",
            "dynamic_batching {}",
            "dynamic_batching: {",
            "dynamic_batching: { preferred_batch_size: [1] }",
            "dynamic_batching",
            "dynamic_batching:",
        ):
            self.assertTrue(matcher(line), line)

        for line in (
            'value: "dynamic_batching"',
            '# dynamic_batching {',
            'url: "https://example.test/dynamic_batching"',
            "dynamic_batching: false",
            "my_dynamic_batching {",
        ):
            self.assertFalse(matcher(line), line)


def _strip_textproto_comment(line):
    """Remove # and // comments that occur outside double-quoted strings."""
    in_string = False
    escaped = False
    for index, character in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "#":
            return line[:index]
        elif character == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index]
    return line


def _is_dynamic_batching_field(line):
    active_line = _strip_textproto_comment(line)
    return bool(
        re.match(
            r"^\s*dynamic_batching\s*:?\s*(?:\{.*|$)",
            active_line,
        )
    )


def _active_lines(config_path):
    """Return config lines with trailing # and // comments removed."""
    active_lines = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = _strip_textproto_comment(line)
        if line.strip():
            active_lines.append(line)
    return active_lines


_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|-?\d+|[A-Za-z_][A-Za-z0-9_]*|[{}\[\]:,]'
)


def _tokenize_textproto(config_text):
    active_text = "\n".join(
        _strip_textproto_comment(line) for line in config_text.splitlines()
    )
    tokens = []
    cursor = 0
    for match in _TOKEN.finditer(active_text):
        skipped = active_text[cursor : match.start()]
        if skipped.strip():
            raise ValueError(f"Unsupported textproto syntax near {skipped!r}")
        tokens.append(match.group(0))
        cursor = match.end()
    if active_text[cursor:].strip():
        raise ValueError(
            f"Unsupported textproto syntax near {active_text[cursor:]!r}"
        )
    return tokens


class _TextprotoParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def _peek(self):
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _pop(self):
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of textproto")
        self.index += 1
        return token

    def _value(self):
        token = self._pop()
        if token == "{":
            value = self._message("}")
            if self._pop() != "}":
                raise AssertionError("parser lost closing brace")
            return value
        if token == "[":
            values = []
            while self._peek() != "]":
                values.append(self._value())
                if self._peek() == ",":
                    self._pop()
                if self._peek() is None:
                    raise ValueError("Unterminated textproto list")
            self._pop()
            return values
        if token.startswith('"'):
            return json.loads(token)
        if re.fullmatch(r"-?\d+", token):
            return int(token)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            return token
        raise ValueError(f"Unexpected textproto value {token!r}")

    def _message(self, terminator=None):
        message = {}
        while self._peek() is not None and self._peek() != terminator:
            field_name = self._pop()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
                raise ValueError(f"Expected field name, got {field_name!r}")
            if self._peek() == ":":
                self._pop()
            value = self._value()
            message.setdefault(field_name, []).append(value)
            if self._peek() == ",":
                self._pop()
        return message

    def parse(self):
        message = self._message()
        if self._peek() is not None:
            raise ValueError(f"Unexpected trailing token {self._peek()!r}")
        return message


def parse_textproto(config_text):
    """Parse the config textproto subset without being fooled by comments."""
    return _TextprotoParser(_tokenize_textproto(config_text)).parse()


def parse_config(config_path):
    return parse_textproto(config_path.read_text(encoding="utf-8"))


def _scalar(message, field_name):
    values = message.get(field_name, [])
    if len(values) != 1 or isinstance(values[0], (dict, list)):
        raise AssertionError(
            f"Expected exactly one scalar {field_name}, got {values!r}"
        )
    return values[0]


def _repeated_messages(message, field_name):
    blocks = []
    for occurrence in message.get(field_name, []):
        items = occurrence if isinstance(occurrence, list) else [occurrence]
        for item in items:
            if not isinstance(item, dict):
                raise AssertionError(
                    f"Expected message in {field_name}, got {item!r}"
                )
            blocks.append(item)
    return blocks


def _parameters(config):
    parameters = {}
    for entry in _repeated_messages(config, "parameters"):
        key = _scalar(entry, "key")
        values = _repeated_messages(entry, "value")
        if len(values) != 1:
            raise AssertionError(
                f"Expected exactly one parameter value for {key!r}, got {values!r}"
            )
        if key in parameters:
            raise AssertionError(f"Duplicate parameter key {key!r}")
        parameters[key] = _scalar(values[0], "string_value")
    return parameters


def _tensor_contracts(config, field_name):
    contracts = []
    seen_names = set()
    for tensor in _repeated_messages(config, field_name):
        name = _scalar(tensor, "name")
        data_type = _scalar(tensor, "data_type")
        dims_values = tensor.get("dims", [])
        if len(dims_values) != 1 or not isinstance(dims_values[0], list):
            raise AssertionError(f"Malformed dims for {name}: {dims_values!r}")
        dims = tuple(dims_values[0])
        if name in seen_names:
            raise AssertionError(f"Duplicate {field_name} tensor name {name!r}")
        seen_names.add(name)
        contracts.append((name, data_type, dims))
    return tuple(contracts)


def _map_entries(step, field_name):
    entries = []
    for mapping in _repeated_messages(step, field_name):
        entries.append((_scalar(mapping, "key"), _scalar(mapping, "value")))
    return entries


def validate_ensemble_flow(configs, ensemble_name):
    """Validate tensor provenance and contracts for one parsed Ensemble."""
    if ensemble_name not in configs:
        raise AssertionError(f"Unknown ensemble {ensemble_name!r}")
    ensemble = configs[ensemble_name]
    if ensemble.get("platform") != ["ensemble"]:
        raise AssertionError(f"{ensemble_name!r} is not an ensemble")

    producers = {}
    consumers = {}
    step_flows = []

    def record(table, alias, contract, source, kind):
        table.setdefault(alias, []).append(
            {
                "contract": contract,
                "source": source,
                "kind": kind,
            }
        )

    public_inputs = _tensor_contracts(ensemble, "input")
    public_outputs = _tensor_contracts(ensemble, "output")
    for name, data_type, dims in public_inputs:
        record(
            producers,
            name,
            (data_type, dims),
            f"{ensemble_name}.input[{name}]",
            "public input",
        )
    for name, data_type, dims in public_outputs:
        record(
            consumers,
            name,
            (data_type, dims),
            f"{ensemble_name}.output[{name}]",
            "public output",
        )

    schedules = _repeated_messages(ensemble, "ensemble_scheduling")
    if len(schedules) != 1:
        raise AssertionError(
            f"{ensemble_name} must have exactly one ensemble_scheduling block"
        )
    steps = _repeated_messages(schedules[0], "step")
    if not steps:
        raise AssertionError(f"{ensemble_name} must have at least one step")

    for step_index, step in enumerate(steps):
        child_name = _scalar(step, "model_name")
        if child_name not in configs:
            raise AssertionError(
                f"{ensemble_name} step {step_index} references unknown model "
                f"{child_name!r}"
            )
        child = configs[child_name]
        child_inputs = {
            tensor[0]: tensor[1:]
            for tensor in _tensor_contracts(child, "input")
        }
        child_outputs = {
            tensor[0]: tensor[1:]
            for tensor in _tensor_contracts(child, "output")
        }
        input_maps = _map_entries(step, "input_map")
        output_maps = _map_entries(step, "output_map")

        mapped_inputs = [key for key, _ in input_maps]
        mapped_outputs = [key for key, _ in output_maps]
        if len(mapped_inputs) != len(set(mapped_inputs)):
            raise AssertionError(
                f"{child_name} step {step_index} has duplicate input_map keys"
            )
        if len(mapped_outputs) != len(set(mapped_outputs)):
            raise AssertionError(
                f"{child_name} step {step_index} has duplicate output_map keys"
            )
        if set(mapped_inputs) != set(child_inputs):
            raise AssertionError(
                f"{child_name} step {step_index} input_map does not match "
                f"child inputs: mapped={sorted(mapped_inputs)}, "
                f"expected={sorted(child_inputs)}"
            )
        if set(mapped_outputs) != set(child_outputs):
            raise AssertionError(
                f"{child_name} step {step_index} output_map does not match "
                f"child outputs: mapped={sorted(mapped_outputs)}, "
                f"expected={sorted(child_outputs)}"
            )

        for key, alias in input_maps:
            record(
                consumers,
                alias,
                child_inputs[key],
                f"{child_name}.input[{key}] at step {step_index}",
                "step input",
            )
        for key, alias in output_maps:
            record(
                producers,
                alias,
                child_outputs[key],
                f"{child_name}.output[{key}] at step {step_index}",
                "step output",
            )
        step_flows.append(
            {
                "description": f"{child_name} at step {step_index}",
                "inputs": {alias for _, alias in input_maps},
                "outputs": {alias for _, alias in output_maps},
            }
        )

    for alias, alias_producers in producers.items():
        if len(alias_producers) > 1:
            sources = [entry["source"] for entry in alias_producers]
            raise AssertionError(
                f"ensemble tensor {alias!r} has multiple producers: {sources}"
            )
        if not consumers.get(alias):
            raise AssertionError(
                f"ensemble tensor {alias!r} producer "
                f"{alias_producers[0]['source']} has no consumers"
            )

    for alias, alias_consumers in consumers.items():
        alias_producers = producers.get(alias, [])
        if not alias_producers:
            sources = [entry["source"] for entry in alias_consumers]
            raise AssertionError(
                f"ensemble tensor {alias!r} consumers {sources} have no producer"
            )
        if len(alias_producers) > 1:
            sources = [entry["source"] for entry in alias_producers]
            raise AssertionError(
                f"ensemble tensor {alias!r} has multiple producers: {sources}"
            )
        producer = alias_producers[0]
        for consumer in alias_consumers:
            if producer["contract"] != consumer["contract"]:
                raise AssertionError(
                    f"ensemble tensor {alias!r} contract mismatch between "
                    f"{producer['source']} {producer['contract']} and "
                    f"{consumer['source']} {consumer['contract']}"
                )
            if (
                consumer["kind"] == "public output"
                and producer["kind"] != "step output"
            ):
                raise AssertionError(
                    f"public output {alias!r} must be produced by a step output, "
                    f"got {producer['source']}"
                )

    available = {name for name, _, _ in public_inputs}
    pending = list(step_flows)
    while pending:
        ready = [
            step for step in pending if step["inputs"].issubset(available)
        ]
        if not ready:
            break
        for step in ready:
            available.update(step["outputs"])
            pending.remove(step)

    if pending:
        blocked = [
            {
                "step": step["description"],
                "missing": sorted(step["inputs"] - available),
            }
            for step in pending
        ]
        raise AssertionError(
            f"{ensemble_name} has cycle/unreachable steps: {blocked}"
        )

    unreachable_outputs = sorted(
        name for name, _, _ in public_outputs if name not in available
    )
    if unreachable_outputs:
        raise AssertionError(
            f"{ensemble_name} public outputs are not reachable: "
            f"{unreachable_outputs}"
        )


class TextprotoParserContractTests(unittest.TestCase):
    def test_message_and_list_forms_parse_all_tensors_and_quoted_comments(self):
        config = parse_textproto(
            r'''
            name: "https://example.test/model#fragment" # real comment
            input {
              name: "first//input"
              data_type: TYPE_FP32
              dims: [2]
            }
            input [
              { name: "second#input" data_type: TYPE_INT64 dims: [1] },
              { name: "third" data_type: TYPE_UINT8 dims: [-1] }
            ]
            output { name: "one" data_type: TYPE_FP32 dims: [3] }
            output [
              { name: "two" data_type: TYPE_INT32 dims: [1] }
            ]
            '''
        )

        self.assertEqual(
            _scalar(config, "name"),
            "https://example.test/model#fragment",
        )
        self.assertEqual(
            _tensor_contracts(config, "input"),
            (
                ("first//input", "TYPE_FP32", (2,)),
                ("second#input", "TYPE_INT64", (1,)),
                ("third", "TYPE_UINT8", (-1,)),
            ),
        )
        self.assertEqual(
            _tensor_contracts(config, "output"),
            (
                ("one", "TYPE_FP32", (3,)),
                ("two", "TYPE_INT32", (1,)),
            ),
        )

    def test_multiple_map_message_and_list_forms_are_all_returned(self):
        step = parse_textproto(
            '''
            input_map { key: "a" value: "A" }
            input_map [
              { key: "b" value: "B" },
              { key: "c" value: "C" }
            ]
            output_map { key: "x" value: "X" }
            output_map [
              { key: "y" value: "Y" },
              { key: "z" value: "Z" }
            ]
            '''
        )
        self.assertEqual(
            _map_entries(step, "input_map"),
            [("a", "A"), ("b", "B"), ("c", "C")],
        )
        self.assertEqual(
            _map_entries(step, "output_map"),
            [("x", "X"), ("y", "Y"), ("z", "Z")],
        )

    def test_duplicate_tensor_names_are_rejected_across_message_forms(self):
        config = parse_textproto(
            '''
            input { name: "same" data_type: TYPE_FP32 dims: [2] }
            input [ { name: "same" data_type: TYPE_FP32 dims: [2] } ]
            '''
        )
        with self.assertRaisesRegex(AssertionError, "Duplicate input tensor"):
            _tensor_contracts(config, "input")

    def test_parameters_parse_message_and_list_forms(self):
        config = parse_textproto(
            '''
            parameters {
              key: "api_contract_version"
              value { string_value: "1" }
            }
            parameters [
              { key: "release" value { string_value: "stable" } },
              { key: "owner" value { string_value: "alog" } }
            ]
            '''
        )

        self.assertEqual(
            {
                "api_contract_version": "1",
                "owner": "alog",
                "release": "stable",
            },
            _parameters(config),
        )

    def test_parameters_reject_duplicate_keys(self):
        config = parse_textproto(
            '''
            parameters { key: "version" value { string_value: "1" } }
            parameters { key: "version" value { string_value: "2" } }
            '''
        )

        with self.assertRaisesRegex(AssertionError, "Duplicate parameter key"):
            _parameters(config)

    def test_parameters_require_exactly_one_value(self):
        cases = {
            "missing": 'parameters { key: "version" }',
            "multiple": '''
                parameters {
                  key: "version"
                  value { string_value: "1" }
                  value { string_value: "2" }
                }
            ''',
        }

        for case_name, config_text in cases.items():
            with self.subTest(case=case_name):
                config = parse_textproto(config_text)
                with self.assertRaisesRegex(
                    AssertionError, "exactly one parameter value"
                ):
                    _parameters(config)

    def test_parameters_require_string_value(self):
        config = parse_textproto(
            '''
            parameters { key: "version" value { int64_value: 1 } }
            '''
        )

        with self.assertRaisesRegex(AssertionError, "scalar string_value"):
            _parameters(config)


class EnsembleFlowGraphTests(unittest.TestCase):
    @staticmethod
    def _child_configs():
        return {
            "source": parse_textproto(
                '''
                input [
                  { name: "x" data_type: TYPE_FP32 dims: [2] },
                  { name: "flag" data_type: TYPE_INT64 dims: [1] }
                ]
                output { name: "y" data_type: TYPE_FP32 dims: [3] }
                '''
            ),
            "sink_one": parse_textproto(
                '''
                input { name: "z" data_type: TYPE_FP32 dims: [3] }
                output { name: "q" data_type: TYPE_FP32 dims: [1] }
                '''
            ),
            "sink_two": parse_textproto(
                '''
                input { name: "z" data_type: TYPE_FP32 dims: [3] }
                output { name: "q" data_type: TYPE_FP32 dims: [1] }
                '''
            ),
        }

    @staticmethod
    def _ensemble(extra_input="", extra_step="", sink_one_alias="MID"):
        return parse_textproto(
            f'''
            platform: "ensemble"
            input [
              {{ name: "PUBLIC" data_type: TYPE_FP32 dims: [2] }},
              {{ name: "FLAG" data_type: TYPE_INT64 dims: [1] }}
              {extra_input}
            ]
            output [
              {{ name: "RESULT_ONE" data_type: TYPE_FP32 dims: [1] }},
              {{ name: "RESULT_TWO" data_type: TYPE_FP32 dims: [1] }}
            ]
            ensemble_scheduling {{
              step [
                {{
                  model_name: "source"
                  input_map [
                    {{ key: "x" value: "PUBLIC" }},
                    {{ key: "flag" value: "FLAG" }}
                  ]
                  output_map {{ key: "y" value: "MID" }}
                }},
                {extra_step}
                {{
                  model_name: "sink_one"
                  input_map {{ key: "z" value: "{sink_one_alias}" }}
                  output_map {{ key: "q" value: "RESULT_ONE" }}
                }},
                {{
                  model_name: "sink_two"
                  input_map {{ key: "z" value: "MID" }}
                  output_map {{ key: "q" value: "RESULT_TWO" }}
                }}
              ]
            }}
            '''
        )

    def _configs(self, ensemble):
        configs = self._child_configs()
        configs["pipeline"] = ensemble
        return configs

    def test_one_producer_can_feed_multiple_contract_matching_consumers(self):
        validate_ensemble_flow(self._configs(self._ensemble()), "pipeline")

    def test_dangling_consumer_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "no producer"):
            validate_ensemble_flow(
                self._configs(self._ensemble(sink_one_alias="MISSING")),
                "pipeline",
            )

    def test_unused_public_input_producer_is_rejected(self):
        extra_input = (
            ', { name: "UNUSED" data_type: TYPE_FP32 dims: [4] }'
        )
        with self.assertRaisesRegex(AssertionError, "no consumers"):
            validate_ensemble_flow(
                self._configs(self._ensemble(extra_input=extra_input)),
                "pipeline",
            )

    def test_multiple_producers_for_one_alias_are_rejected(self):
        extra_step = '''
        {
          model_name: "source"
          input_map [
            { key: "x" value: "PUBLIC" },
            { key: "flag" value: "FLAG" }
          ]
          output_map { key: "y" value: "MID" }
        },
        '''
        with self.assertRaisesRegex(AssertionError, "multiple producers"):
            validate_ensemble_flow(
                self._configs(self._ensemble(extra_step=extra_step)),
                "pipeline",
            )

    def test_declaration_order_does_not_affect_acyclic_reachability(self):
        ensemble = self._ensemble()
        schedule = _repeated_messages(ensemble, "ensemble_scheduling")[0]
        steps = _repeated_messages(schedule, "step")
        schedule["step"] = [list(reversed(steps))]
        validate_ensemble_flow(self._configs(ensemble), "pipeline")

    @staticmethod
    def _cycle_configs(with_unreachable_tail):
        configs = {
            "pass": parse_textproto(
                '''
                input { name: "x" data_type: TYPE_FP32 dims: [2] }
                output { name: "y" data_type: TYPE_FP32 dims: [1] }
                '''
            ),
            "loop_a": parse_textproto(
                '''
                input { name: "x" data_type: TYPE_FP32 dims: [3] }
                output { name: "y" data_type: TYPE_FP32 dims: [3] }
                '''
            ),
            "loop_b": parse_textproto(
                '''
                input { name: "x" data_type: TYPE_FP32 dims: [3] }
                output { name: "y" data_type: TYPE_FP32 dims: [3] }
                '''
            ),
            "tail": parse_textproto(
                '''
                input { name: "x" data_type: TYPE_FP32 dims: [3] }
                output { name: "y" data_type: TYPE_FP32 dims: [1] }
                '''
            ),
        }
        tail_output = (
            ', { name: "ISLAND_RESULT" data_type: TYPE_FP32 dims: [1] }'
            if with_unreachable_tail
            else ""
        )
        tail_step = (
            '''
            {
              model_name: "tail"
              input_map { key: "x" value: "A" }
              output_map { key: "y" value: "ISLAND_RESULT" }
            },
            '''
            if with_unreachable_tail
            else ""
        )
        configs["pipeline"] = parse_textproto(
            f'''
            platform: "ensemble"
            input {{ name: "PUBLIC" data_type: TYPE_FP32 dims: [2] }}
            output [
              {{ name: "RESULT" data_type: TYPE_FP32 dims: [1] }}
              {tail_output}
            ]
            ensemble_scheduling {{
              step [
                {{
                  model_name: "pass"
                  input_map {{ key: "x" value: "PUBLIC" }}
                  output_map {{ key: "y" value: "RESULT" }}
                }},
                {{
                  model_name: "loop_a"
                  input_map {{ key: "x" value: "B" }}
                  output_map {{ key: "y" value: "A" }}
                }},
                {{
                  model_name: "loop_b"
                  input_map {{ key: "x" value: "A" }}
                  output_map {{ key: "y" value: "B" }}
                }},
                {tail_step}
              ]
            }}
            '''
        )
        return configs

    def test_two_step_cycle_is_rejected_as_unreachable(self):
        with self.assertRaisesRegex(AssertionError, "cycle/unreachable"):
            validate_ensemble_flow(
                self._cycle_configs(with_unreachable_tail=False),
                "pipeline",
            )

    def test_unreachable_island_and_public_output_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "cycle/unreachable"):
            validate_ensemble_flow(
                self._cycle_configs(with_unreachable_tail=True),
                "pipeline",
            )


class PublicApiVersionContractTests(unittest.TestCase):
    def test_alog_public_models_have_expected_contract_versions(self):
        configs = {
            path.parent.name: parse_config(path)
            for path in sorted(REPOSITORY_ROOT.glob("*/config.pbtxt"))
        }
        self.assertEqual(
            ALOG_PUBLIC_MODELS,
            ALOG_PUBLIC_MODELS.intersection(configs),
            "all ALog public models must exist in the repository",
        )
        versioned_models = {
            model_name
            for model_name, config in configs.items()
            if "api_contract_version" in _parameters(config)
        }
        self.assertEqual(
            ALOG_PUBLIC_MODELS,
            versioned_models,
            "only ALog public models may declare api_contract_version",
        )

        for model_name in sorted(ALOG_PUBLIC_MODELS):
            with self.subTest(model=model_name):
                expected_version = (
                    "3" if model_name == "yolo26s_seg_ensemble"
                    else "2" if model_name in {
                        "face_qa_ensemble", "yolov8n_face_ensemble",
                        "yolo26m_pose_ensemble", "sbs_r50_ibn_image_ensemble",
                        "TBPS_clip_image_ensemble",
                    } else "1"
                )
                self.assertEqual(
                    expected_version,
                    _parameters(configs[model_name]).get("api_contract_version"),
                )


class BatchRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_paths = sorted(REPOSITORY_ROOT.glob("*/config.pbtxt"))
        cls.configs = {
            path.parent.name: parse_config(path) for path in cls.config_paths
        }

    def test_single_config_inventory_is_48_14_34(self):
        self.assertEqual(len(self.config_paths), 48)
        # Keep one canonical config per model, with explicit Orin GPU opt-ins.
        gpu_paths = list(REPOSITORY_ROOT.glob("*/configs/*.pbtxt"))
        yolo_names = ("yolov8n_face", "yolo26m_pose", "yolo26s_seg")
        expected_gpu_instances = {
            "yolov8n_face_ensemble": 4,
            "yolo26m_pose_ensemble": 2,
            "yolo26s_seg_ensemble": 2,
            "TBPS_clip_image_ensemble": 1,
            "clip_image_preprocess": 2,
            "sbs_r50_ibn_image_preprocess": 1,
            "face_keypoint_2d106_pre": 1,
            "face_qa_pre": 1,
            "face_rec_pre": 1,
            "clip_ensemble_trt": 2,
            "sbs_r50_ibn_image_ensemble": 2,
            "face_keypoint_2d106_ensemble": 2,
            "face_qa_ensemble": 2,
            "face_recognition_pipeline": 2,
        }
        gpu_preprocessors = {
            "clip_image_preprocess", "sbs_r50_ibn_image_preprocess",
            "face_keypoint_2d106_pre", "face_qa_pre", "face_rec_pre",
        }
        fused_faces = {"face_keypoint_2d106_ensemble", "face_qa_ensemble", "face_recognition_pipeline"}
        fused_images = {"clip_ensemble_trt": "4", "sbs_r50_ibn_image_ensemble": "2"}
        self.assertEqual(
            {path.relative_to(REPOSITORY_ROOT).as_posix() for path in gpu_paths},
            {f"{name}/configs/orin_gpu.pbtxt" for name in expected_gpu_instances},
        )
        for name in yolo_names:
            for phase in ("preprocess", "postprocess"):
                default = self.configs[f"{name}_{phase}"]
                default_groups = _repeated_messages(default, "instance_group")
                self.assertEqual(len(default_groups), 1)
                self.assertEqual(default_groups[0].get("kind"), ["KIND_CPU"],
                                 "default YOLO helpers must not inherit Triton's automatic GPU placement")
        for path in gpu_paths:
            gpu_config = parse_config(path)
            name = path.parent.parent.name
            default = self.configs[name]
            self.assertEqual(gpu_config.get("backend"), ["python"])
            self.assertNotIn("platform", gpu_config)
            self.assertNotIn("ensemble_scheduling", gpu_config)
            if path.parent.parent.name in ("yolo26m_pose_ensemble", "yolo26s_seg_ensemble"):
                self.assertEqual(gpu_config.get("dynamic_batching"), [{}])
            elif path.parent.parent.name == "yolov8n_face_ensemble":
                self.assertNotIn("dynamic_batching", gpu_config)
            for field in ("name", "max_batch_size", "version_policy"):
                self.assertEqual(gpu_config.get(field), default.get(field))
            # Protobuf repeated messages accept both input { ... } and input [ ... ].
            for field in ("input", "output"):
                actual = _repeated_messages(gpu_config, field)
                if field == "input" and name in fused_faces:
                    raw_name = "RAW_FACE_IMAGE" if name == "face_qa_ensemble" else "RAW_IMAGE"
                    for item in actual:
                        if _scalar(item, "name") == raw_name:
                            self.assertEqual(item.get("allow_ragged_batch"), ["true"])
                        else:
                            self.assertNotIn("allow_ragged_batch", item)
                    actual = [{key: value for key, value in item.items() if key != "allow_ragged_batch"}
                              for item in actual]
                self.assertEqual(actual, _repeated_messages(default, field))
            self.assertEqual(_parameters(gpu_config).get("api_contract_version"),
                             _parameters(default).get("api_contract_version"))
            if path.parent.parent.name in gpu_preprocessors:
                self.assertEqual(_parameters(gpu_config).get("preprocess_backend"), "orin_gpu")
                self.assertEqual(_parameters(default).get("preprocess_backend", "cpu"), "cpu")
            if name in fused_faces or name in fused_images:
                self.assertEqual(_parameters(gpu_config).get("pipeline_backend"), "orin_gpu")
                scheduler = _repeated_messages(gpu_config, "dynamic_batching")
                self.assertEqual(len(scheduler), 1)
                self.assertEqual(scheduler[0].get("preferred_batch_size"), [[4]])
                self.assertEqual(_scalar(scheduler[0], "max_queue_delay_microseconds"),
                                 250 if name in fused_faces else 500)
                if name in fused_images:
                    self.assertEqual(_parameters(gpu_config).get("jpeg_decode_workers"), fused_images[name])
            groups = _repeated_messages(gpu_config, "instance_group")
            self.assertEqual(len(groups), 1)
            self.assertEqual(_scalar(groups[0], "count"),
                             expected_gpu_instances[path.parent.parent.name])
            self.assertEqual(groups[0].get("kind"), ["KIND_GPU"])
            self.assertEqual(_parameters(gpu_config).get("FORCE_CPU_ONLY_INPUT_TENSORS"), "yes")
        ensembles = {
            name for name, config in self.configs.items()
            if config.get("platform") == ["ensemble"]
        }
        self.assertEqual(len(ensembles), 14)
        self.assertEqual(len(self.configs) - len(ensembles), 34)

    def test_all_configs_have_batch_four_without_preferred_or_ragged_batches(self):
        for name, config in self.configs.items():
            with self.subTest(model=name):
                self.assertEqual(_scalar(config, "max_batch_size"), 4)
                self.assertNotIn("preferred_batch_size", config)
                self.assertNotIn("max_queue_delay_microseconds", config)
                self.assertNotIn("allow_ragged_batch", config)
                for scheduler in _repeated_messages(config, "dynamic_batching"):
                    self.assertNotIn("preferred_batch_size", scheduler)

    def test_executables_have_expected_instances_and_backend_specific_batch_wait(self):
        delayed_nodes = 0
        for name, config in self.configs.items():
            if config.get("platform") == ["ensemble"]:
                continue
            with self.subTest(model=name):
                groups = _repeated_messages(config, "instance_group")
                self.assertEqual(len(groups), 1)
                self.assertEqual(_scalar(groups[0], "count"),
                                 2 if name == "clip_image_preprocess" else 1)
                is_inference = (
                    config.get("backend") == ["onnxruntime"]
                    or config.get("platform") == ["onnxruntime_onnx"]
                    or config.get("backend") == ["tensorrt"]
                    or config.get("platform") == ["tensorrt_plan"]
                )
                expected_scheduler = (
                    {"max_queue_delay_microseconds": [2000]}
                    if is_inference else {}
                )
                self.assertEqual(
                    _repeated_messages(config, "dynamic_batching"),
                    [expected_scheduler],
                )
                delayed_nodes += int(is_inference)
        self.assertEqual(delayed_nodes, 14)


    def test_ensembles_have_no_instance_or_dynamic_scheduler(self):
        for name, config in self.configs.items():
            if config.get("platform") != ["ensemble"]:
                continue
            with self.subTest(model=name):
                self.assertNotIn("instance_group", config)
                self.assertNotIn("dynamic_batching", config)
                self.assertIn("ensemble_scheduling", config)


class PerSampleTensorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_paths = sorted(REPOSITORY_ROOT.glob("*/config.pbtxt"))
        cls.configs = {
            path.parent.name: parse_config(path) for path in cls.config_paths
        }

    def test_exact_contract_covers_all_48_models(self):
        self.assertEqual(48, len(self.config_paths))
        self.assertEqual(set(self.configs), set(MODEL_IO_CONTRACTS))

    def test_every_input_and_output_uses_the_exact_per_sample_contract(self):
        for model_name, expected in MODEL_IO_CONTRACTS.items():
            with self.subTest(model=model_name):
                config = self.configs[model_name]
                self.assertEqual(_scalar(config, "name"), model_name)
                self.assertEqual(
                    _tensor_contracts(config, "input"), expected["inputs"]
                )
                self.assertEqual(
                    _tensor_contracts(config, "output"), expected["outputs"]
                )

    def test_chinese_clip_text_preprocessor_declares_only_input_ids(self):
        outputs = _tensor_contracts(self.configs["clip_text_preprocess"], "output")
        self.assertEqual(outputs, (("INPUT_IDS", "TYPE_INT64", (52,)),))


class EnsembleMappingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            path.parent.name: parse_config(path)
            for path in sorted(REPOSITORY_ROOT.glob("*/config.pbtxt"))
        }

    def test_every_ensemble_mapping_matches_all_child_tensor_contracts(self):
        ensemble_names = sorted(
            model_name
            for model_name, config in self.configs.items()
            if config.get("platform") == ["ensemble"]
        )
        self.assertEqual(14, len(ensemble_names))

        for ensemble_name in ensemble_names:
            with self.subTest(ensemble=ensemble_name):
                validate_ensemble_flow(self.configs, ensemble_name)

    def test_sixdrepnet_ensemble_has_the_exact_three_step_contract(self):
        scheduling = _repeated_messages(
            self.configs["sixdrepnet_ensemble"], "ensemble_scheduling"
        )
        self.assertEqual(1, len(scheduling))
        steps = _repeated_messages(scheduling[0], "step")
        expected = (
            (
                "sixdrepnet_preprocess",
                ("RAW_FACE_IMAGE", "RAW_FACE_IMAGE"),
                ("PREPROCESSED_IMAGE", "preprocessed_image"),
            ),
            (
                "sixdrepnet_onnx",
                ("image", "preprocessed_image"),
                ("rotation_matrix", "rotation_matrix"),
            ),
            (
                "sixdrepnet_postprocess",
                ("ROTATION_MATRIX", "rotation_matrix"),
                ("HEAD_POSE", "HEAD_POSE"),
            ),
        )
        self.assertEqual(len(expected), len(steps))
        for step, (model_name, input_mapping, output_mapping) in zip(
            steps, expected, strict=True
        ):
            self.assertEqual(model_name, _scalar(step, "model_name"))
            self.assertEqual(-1, _scalar(step, "model_version"))
            self.assertEqual(
                {"key": [input_mapping[0]], "value": [input_mapping[1]]},
                _repeated_messages(step, "input_map")[0],
            )
            self.assertEqual(
                {"key": [output_mapping[0]], "value": [output_mapping[1]]},
                _repeated_messages(step, "output_map")[0],
            )


if __name__ == "__main__":
    unittest.main()
