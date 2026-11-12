"""Orin GPU profile 的同进程人脸入口。"""
import sys

sys.path.insert(0, "/usr/local/bin")
from face_orin_pipeline import FaceOrinPipeline as TritonPythonModel
