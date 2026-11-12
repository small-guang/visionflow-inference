import importlib.util
import sys
import types


_MISSING = object()


class FakeTensor:
    def __init__(self, name, value):
        self.__name = name
        self.__value = value

    def name(self):
        return self.__name

    def as_numpy(self):
        return self.__value


class FakeRequest:
    def __init__(self, inputs):
        self.inputs = dict(inputs)


class FakeTritonError(RuntimeError):
    pass


class FakeResponse:
    def __init__(self, output_tensors=None, error=None):
        self.__output_tensors = list(output_tensors or [])
        self.__error = error

    def has_error(self):
        return self.__error is not None

    def error(self):
        return self.__error

    def output_tensors(self):
        return list(self.__output_tensors)


def _get_input_tensor_by_name(request, name):
    return request.inputs.get(name)


def make_request(**values):
    return FakeRequest(
        {name: FakeTensor(name, value) for name, value in values.items()}
    )


def load_backend(module_name, model_path):
    pb_utils = types.ModuleType("triton_python_backend_utils")
    pb_utils.Tensor = FakeTensor
    pb_utils.InferenceResponse = FakeResponse
    pb_utils.TritonError = FakeTritonError
    pb_utils.get_input_tensor_by_name = _get_input_tensor_by_name

    previous_pb_utils = sys.modules.get(
        "triton_python_backend_utils", _MISSING
    )
    sys.modules["triton_python_backend_utils"] = pb_utils
    try:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load backend module from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_pb_utils is _MISSING:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous_pb_utils

    try:
        model = module.TritonPythonModel()
    except AttributeError as error:
        raise ImportError(
            f"backend module {model_path} has no TritonPythonModel"
        ) from error
    return module, model
