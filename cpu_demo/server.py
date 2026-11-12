"""Small CPU-only face detection service for the portfolio demo.

This is a separate HTTP API, not an implementation of Triton's V2 protocol.
"""
import base64
import binascii
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


def make_detector():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be loaded")
    lock = threading.Lock()

    def detect(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with lock:
            boxes, _, weights = cascade.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=5, outputRejectLevels=True)
        return [{"box": [int(v) for v in box], "score": float(score)}
                for box, score in zip(boxes, np.asarray(weights).reshape(-1))]
    return detect


def create_server(host="127.0.0.1", port=8080, detector=None):
    detect = detector if detector is not None else make_detector()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready"})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/detect":
                self.respond(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("request body must be 1 byte to 8 MiB")
                payload = json.loads(self.rfile.read(size))
                image_bytes = base64.b64decode(payload["image_base64"], validate=True)
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[0] * image.shape[1] > MAX_PIXELS:
                    raise ValueError("invalid image or image larger than 16 megapixels")
            except (ValueError, TypeError, KeyError, binascii.Error, json.JSONDecodeError, cv2.error):
                self.respond(400, {"error": "invalid image or request"})
                return
            try:
                detections = detect(image)
                self.respond(200, {"image_size": [int(image.shape[1]), int(image.shape[0])], "detections": detections})
            except Exception:
                self.respond(500, {"error": "detection failed"})

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    create_server(os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8080"))).serve_forever()
