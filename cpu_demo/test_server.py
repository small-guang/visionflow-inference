import base64
import json
import threading
import unittest
from http.client import HTTPConnection

import cv2
import numpy as np

from server import create_server, make_detector


class CpuDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0, detector=lambda image: [{"box": [2, 3, 4, 5], "score": 0.75}])
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body, content_type="application/json"):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": content_type})
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()
        return response.status, payload

    def test_inference_returns_detections_for_valid_image(self):
        ok, image = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertTrue(ok)
        body = json.dumps({"image_base64": base64.b64encode(image.tobytes()).decode("ascii")})
        status, result = self.request("/detect", body)
        self.assertEqual(status, 200)
        self.assertEqual(result["image_size"], [32, 32])
        self.assertEqual(result["detections"][0]["box"], [2, 3, 4, 5])

    def test_invalid_image_is_rejected(self):
        status, result = self.request("/detect", json.dumps({"image_base64": "invalid!"}))
        self.assertEqual(status, 400)
        self.assertIn("error", result)

    def test_unknown_route_is_rejected(self):
        status, _ = self.request("/other", "{}")
        self.assertEqual(status, 404)

    def test_bundled_detector_runs_on_blank_image(self):
        self.assertEqual(make_detector()(np.zeros((128, 128, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    unittest.main()
