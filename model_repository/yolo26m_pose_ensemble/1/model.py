"""Orin-only configuration entry point; defaults retain ensemble scheduling."""
import sys

sys.path.insert(0, "/usr/local/bin")
from yolo_orin_pipeline import TritonPythonModel as OrinPipeline


class TritonPythonModel(OrinPipeline):
    family = "yolo26m_pose"
