"""Used only by configs/orin_gpu.pbtxt; the default remains an ensemble."""
import sys

sys.path.insert(0, "/usr/local/bin")
from tbps_orin_pipeline import TritonPythonModel

