ARG TRITON_GPU_BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3
FROM ${TRITON_GPU_BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 opencv-python-headless==4.10.0.84
