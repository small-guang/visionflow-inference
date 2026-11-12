import argparse
import time

import cv2
import numpy as np
import tritonclient.http as httpclient


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def select_online_outputs(detections, keypoints, num_detections):
    detections = np.asarray(detections)
    keypoints = np.asarray(keypoints)
    counts = np.asarray(num_detections)
    if detections.shape != (1, 100, 6):
        raise ValueError("DETECTIONS must have shape [1,100,6]")
    if keypoints.shape != (1, 100, 17, 3):
        raise ValueError("KEYPOINTS must have shape [1,100,17,3]")
    if counts.shape != (1, 1):
        raise ValueError("NUM_DETECTIONS must have shape [1,1]")
    count = int(counts[0, 0])
    if count < 0 or count > 100:
        raise ValueError("NUM_DETECTIONS is outside the detection capacity")
    return detections[0, :count], keypoints[0, :count], count


def print_detection(index, detection, keypoints):
    x1, y1, x2, y2, score, class_id = detection
    print(
        f"#{index}: class={int(class_id)} score={score:.3f} "
        f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
    )
    for name, point in zip(COCO_KEYPOINT_NAMES, keypoints):
        x, y, visibility = point
        print(f"  {name}: x={x:.1f}, y={y:.1f}, score={visibility:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Local image path")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="yolo26m_pose_ensemble")
    parser.add_argument("--limit", type=int, default=5, help="Maximum detections to print")
    args = parser.parse_args()

    read_start = time.perf_counter()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    read_ms = (time.perf_counter() - read_start) * 1000

    client = httpclient.InferenceServerClient(url=args.url)
    image_batch = np.ascontiguousarray(image[None, ...])
    infer_input = httpclient.InferInput("RAW_IMAGE", image_batch.shape, "UINT8")
    infer_input.set_data_from_numpy(image_batch)

    infer_start = time.perf_counter()
    result = client.infer(
        args.model,
        inputs=[infer_input],
        outputs=[
            httpclient.InferRequestedOutput("DETECTIONS"),
            httpclient.InferRequestedOutput("KEYPOINTS"),
            httpclient.InferRequestedOutput("NUM_DETECTIONS"),
        ],
    )
    infer_ms = (time.perf_counter() - infer_start) * 1000

    detections, keypoints, num = select_online_outputs(
        result.as_numpy("DETECTIONS"),
        result.as_numpy("KEYPOINTS"),
        result.as_numpy("NUM_DETECTIONS"),
    )

    print(f"image_shape: {image.shape}")
    print(f"read_time_ms: {read_ms:.2f}")
    print(f"infer_time_ms: {infer_ms:.2f}")
    print(f"detections_shape: {detections.shape}")
    print(f"keypoints_shape: {keypoints.shape}")
    print(f"num_detections: {num}")

    limit = max(0, min(args.limit, num))
    for index in range(limit):
        print_detection(index, detections[index], keypoints[index])


if __name__ == "__main__":
    main()
