import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient


PALETTE = np.array(
    [
        [56, 56, 255],
        [151, 157, 255],
        [31, 112, 255],
        [29, 178, 255],
        [49, 210, 207],
        [10, 249, 72],
        [23, 204, 146],
        [134, 219, 61],
        [52, 147, 26],
        [187, 212, 0],
        [168, 153, 44],
        [255, 194, 0],
        [147, 69, 52],
        [255, 115, 100],
        [236, 24, 0],
        [255, 56, 132],
    ],
    dtype=np.uint8,
)

COCO_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def resolve_output_path(value, default_name):
    path = Path(value)
    if path.exists() and path.is_dir():
        path = path / default_name
    elif not path.suffix:
        path = path.with_suffix(Path(default_name).suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def color_for(class_id, index):
    return PALETTE[(int(class_id) + index) % len(PALETTE)].tolist()


def class_name(class_id):
    class_id = int(class_id)
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return f"cls {class_id}"


def select_batch_outputs(detections, mask_crops, num_detections, batch_index=0):
    detections = np.asarray(detections)
    counts = np.asarray(num_detections)
    if detections.ndim != 3 or detections.shape[2] != 6:
        raise ValueError("DETECTIONS must have shape [B,M,6]")
    if not 1 <= detections.shape[1] <= 100:
        raise ValueError("DETECTIONS capacity M must be between 1 and 100")
    if counts.ndim != 2 or counts.shape[1] != 1:
        raise ValueError("NUM_DETECTIONS must have shape [B,1]")
    if detections.shape[0] != counts.shape[0]:
        raise ValueError("DETECTIONS and NUM_DETECTIONS batch size must match")
    if batch_index < 0 or batch_index >= detections.shape[0]:
        raise IndexError("batch_index is out of range")

    count = int(counts[batch_index, 0])
    if count < 0 or count > detections.shape[1]:
        raise ValueError("NUM_DETECTIONS is outside the detection capacity")
    selected_crops = None
    if mask_crops is not None:
        mask_crops = np.asarray(mask_crops)
        if mask_crops.ndim != 4 or mask_crops.shape[2:] != (160, 20):
            raise ValueError("MASK_CROPS_PACKED must have shape [B,M,160,20]")
        if mask_crops.shape[:2] != detections.shape[:2]:
            raise ValueError("MASK_CROPS_PACKED must match DETECTIONS batch and capacity")
        if mask_crops.dtype != np.uint8:
            raise ValueError("MASK_CROPS_PACKED must have dtype uint8")
        selected_crops = np.unpackbits(
            mask_crops[batch_index, :count], axis=-1, count=160, bitorder="big"
        )
    return detections[batch_index, :count], selected_crops, count


def select_online_outputs(
    detections,
    mask_crops,
    num_detections,
    *,
    expect_masks,
    expected_image_shape,
):
    detections = np.asarray(detections)
    counts = np.asarray(num_detections)
    if (
        detections.ndim != 3
        or detections.shape[0] != 1
        or detections.shape[2] != 6
        or not 1 <= detections.shape[1] <= 100
    ):
        raise ValueError("DETECTIONS must have shape [1,M,6] with 1 <= M <= 100")
    if counts.shape != (1, 1):
        raise ValueError("NUM_DETECTIONS must have shape [1,1]")

    try:
        image_height, image_width = tuple(int(value) for value in expected_image_shape)
    except (TypeError, ValueError):
        raise ValueError("expected_image_shape must be (H,W)") from None
    if image_height <= 0 or image_width <= 0:
        raise ValueError("expected_image_shape must contain positive H and W")

    if mask_crops is None:
        if expect_masks:
            raise ValueError("MASK_CROPS_PACKED response is required unless --boxes-only is used")
    else:
        mask_crops = np.asarray(mask_crops)
        expected_crops_shape = (1, detections.shape[1], 160, 20)
        if mask_crops.shape != expected_crops_shape:
            raise ValueError(
                "MASK_CROPS_PACKED must have shape [1,M,160,20] matching DETECTIONS capacity "
                f"{detections.shape[1]}"
            )
    return select_batch_outputs(detections, mask_crops, counts, batch_index=0)


def restore_mask_crops(detections, mask_crops, image_shape):
    detections = np.asarray(detections)
    mask_crops = np.asarray(mask_crops)
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("DETECTIONS must have shape [N,6]")
    if mask_crops.ndim != 3 or mask_crops.shape[1:] != (160, 160):
        raise ValueError("decoded mask crops must have shape [N,160,160]")
    if mask_crops.shape[0] != detections.shape[0]:
        raise ValueError("DETECTIONS and decoded mask crops must have the same N")

    try:
        image_height, image_width = tuple(int(value) for value in image_shape)
    except (TypeError, ValueError):
        raise ValueError("image_shape must be (H,W)") from None
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_shape must contain positive H and W")

    masks = np.zeros(
        (detections.shape[0], image_height, image_width), dtype=np.uint8
    )
    for index, detection in enumerate(detections):
        if not np.isfinite(detection[:4]).all():
            raise ValueError(f"detection {index} bbox must contain finite coordinates")
        x1, y1, x2, y2 = (
            int(round(float(coordinate))) for coordinate in detection[:4]
        )
        x1 = int(np.clip(x1, 0, image_width))
        x2 = int(np.clip(x2, 0, image_width))
        y1 = int(np.clip(y1, 0, image_height))
        y2 = int(np.clip(y2, 0, image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"detection {index} bbox must have positive area after clipping")
        resized = cv2.resize(
            np.asarray(mask_crops[index], dtype=np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )
        masks[index, y1:y2, x1:x2] = resized
    return masks


def draw_predictions(
    image, detections, masks=None, alpha=0.45, num_detections=None
):
    if num_detections is not None:
        count = int(num_detections)
        if count < 0 or count > len(detections):
            raise ValueError("num_detections is outside the detection capacity")
        detections = detections[:count]
        if masks is not None:
            masks = masks[:count]
    mask_overlay = image.copy()

    if masks is not None:
        for index, det in enumerate(detections):
            color = color_for(det[5], index)
            mask = masks[index].astype(bool)
            if mask.any():
                mask_overlay[mask] = color

    output = cv2.addWeighted(mask_overlay, alpha, image, 1 - alpha, 0)

    for index, det in enumerate(detections):
        x1, y1, x2, y2, score, class_id = det
        color = color_for(class_id, index)

        if masks is not None:
            mask = masks[index].astype(bool)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, 2)

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(output, p1, p2, color, 2)

        label = f"{class_name(class_id)} {score:.2f}"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = max(p1[0], 0)
        label_y = max(p1[1] - text_size[1] - baseline - 3, 0)
        cv2.rectangle(
            output,
            (label_x, label_y),
            (label_x + text_size[0] + 6, label_y + text_size[1] + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (label_x + 3, label_y + text_size[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=r"D:\workspace\hdg\camtellsvi\test\downloads\541288EPM21C0013F66A6AEB_picture_1782440256481.jpg", help="Local image path")
    parser.add_argument("--url", default="192.168.21.61:8000")
    parser.add_argument("--model", default="yolo26s_seg_ensemble")
    parser.add_argument("--save-result", default=str(DEFAULT_OUTPUT_DIR), help="Path or directory for the annotated image")
    parser.add_argument("--save-mask", default="", help="Optional path to save the first mask PNG")
    parser.add_argument("--boxes-only", action="store_true", help="Do not request MASK_CROPS_PACKED from Triton")
    parser.add_argument("--no-result", action="store_true", help="Skip drawing and saving the annotated image")
    args = parser.parse_args()

    read_start = time.perf_counter()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    read_ms = (time.perf_counter() - read_start) * 1000

    client = httpclient.InferenceServerClient(url=args.url)
    total_start = time.perf_counter()
    image_batch = np.ascontiguousarray(image[None, ...])
    infer_input = httpclient.InferInput("RAW_IMAGE", image_batch.shape, "UINT8")
    infer_input.set_data_from_numpy(image_batch)

    infer_start = time.perf_counter()
    outputs = [
        httpclient.InferRequestedOutput("DETECTIONS"),
        httpclient.InferRequestedOutput("NUM_DETECTIONS"),
    ]
    if not args.boxes_only:
        outputs.append(httpclient.InferRequestedOutput("MASK_CROPS_PACKED"))

    result = client.infer(
        args.model,
        inputs=[infer_input],
        outputs=outputs,
    )
    infer_ms = (time.perf_counter() - infer_start) * 1000

    raw_detections = result.as_numpy("DETECTIONS")
    raw_counts = result.as_numpy("NUM_DETECTIONS")
    raw_mask_crops = None if args.boxes_only else result.as_numpy("MASK_CROPS_PACKED")
    detections, mask_crops, num = select_online_outputs(
        raw_detections,
        raw_mask_crops,
        raw_counts,
        expect_masks=not args.boxes_only,
        expected_image_shape=image.shape[:2],
    )
    if raw_mask_crops is not None:
        print(
            f"raw_mask_crops_shape: {raw_mask_crops.shape}, "
            f"raw_mask_crops_mb: {raw_mask_crops.nbytes / 1024 / 1024:.2f}"
        )
    masks = None
    if mask_crops is not None and (
        (args.save_result and not args.no_result) or args.save_mask
    ):
        masks = restore_mask_crops(detections, mask_crops, image.shape[:2])
    total_ms = (time.perf_counter() - total_start) * 1000
    print(f"image_shape: {image.shape}")
    print(f"read_time_ms: {read_ms:.2f}")
    print(f"infer_time_ms: {infer_ms:.2f}")
    print(f"total_time_ms: {total_ms:.2f}")
    print("num_detections:", num)
    print("detections[x1,y1,x2,y2,score,class_id]:")
    print(detections)
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        print(f"{class_name(class_id)} {score:.3f} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    if args.save_result and not args.no_result:
        save_path = resolve_output_path(args.save_result, "yolo26s_seg_result.jpg")
        draw_start = time.perf_counter()
        drawn = draw_predictions(
            image, detections, masks, num_detections=num
        )
        draw_ms = (time.perf_counter() - draw_start) * 1000
        save_start = time.perf_counter()
        ok = cv2.imwrite(str(save_path), drawn)
        save_ms = (time.perf_counter() - save_start) * 1000
        if not ok:
            raise RuntimeError(f"Failed to write result image: {save_path}")
        print(f"draw_time_ms: {draw_ms:.2f}")
        print(f"save_result_time_ms: {save_ms:.2f}")
        print("saved result:", save_path)

    if args.save_mask and num > 0:
        if masks is None:
            raise ValueError("--save-mask cannot be used together with --boxes-only")
        save_path = resolve_output_path(args.save_mask, "yolo26s_seg_mask.png")
        mask = masks[0] * 255
        ok = cv2.imwrite(str(save_path), mask.astype(np.uint8))
        if not ok:
            raise RuntimeError(f"Failed to write mask image: {save_path}")
        print("saved first mask:", save_path)


if __name__ == "__main__":
    main()
