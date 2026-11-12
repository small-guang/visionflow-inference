# release.sh supplies a local runtime reference pinned to its inspected image ID.
ARG TRITON_RUNTIME_IMAGE=registry.cn-hangzhou.aliyuncs.com/eapil/tritonserver:26.04-py3-arm64-runtime
FROM ${TRITON_RUNTIME_IMAGE}

ARG RELEASE_VERSION
ARG SOURCE_REVISION
LABEL org.opencontainers.image.title="Camtells Triton for Jetson" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"

COPY model_repository/ /models/
COPY scripts/start_triton.py scripts/run_triton_with_rotating_logs.py /usr/local/bin/
COPY scripts/yolo_orin_pipeline.py /usr/local/bin/yolo_orin_pipeline.py
COPY scripts/yolo_request_batching.py /usr/local/bin/yolo_request_batching.py
COPY scripts/yolo_orin_metrics.py /usr/local/bin/yolo_orin_metrics.py
COPY scripts/tbps_orin_pipeline.py /usr/local/bin/tbps_orin_pipeline.py
COPY scripts/image_orin_preprocess.py /usr/local/bin/image_orin_preprocess.py
COPY scripts/face_orin_preprocess.py /usr/local/bin/face_orin_preprocess.py
COPY scripts/orin_trt_runtime.py /usr/local/bin/orin_trt_runtime.py
COPY scripts/orin_pipeline_metrics.py /usr/local/bin/orin_pipeline_metrics.py
COPY scripts/image_decode.py /usr/local/bin/image_decode.py
COPY scripts/image_orin_pipeline.py /usr/local/bin/image_orin_pipeline.py
COPY scripts/face_orin_pipeline.py /usr/local/bin/face_orin_pipeline.py
COPY profiles/model_sets.json /opt/triton/model_sets.json
COPY deploy/jetson/smoke.py /opt/triton/release/smoke.py
COPY release.json /opt/triton/release.json

ENV TRITON_MODEL_SET=all \
    TRITON_MODEL_CONFIG=orin_gpu \
    TRITON_LOG_FILE=/var/log/triton/server.log \
    TRITON_LOG_MAX_SIZE=100M \
    TRITON_LOG_BACKUP_COUNT=5 \
    TZ=Asia/Shanghai

RUN mkdir -p /var/log/triton \
    && python3 /opt/triton/release/smoke.py --integrity-only

EXPOSE 8000 8001 8002
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v2/health/ready', timeout=3)"

CMD ["python3", "/usr/local/bin/start_triton.py", "--manifest", "/opt/triton/model_sets.json", "--model-repository", "/models", "--log-wrapper", "/usr/local/bin/run_triton_with_rotating_logs.py", "--", "tritonserver", "--model-repository=/models", "--backend-config=onnxruntime,default-version=1", "--backend-config=tensorrt,default-version=1", "--backend-config=python,default-version=1", "--strict-readiness=true", "--log-verbose=0", "--log-info=true", "--log-warning=true", "--log-error=true", "--log-format=ISO8601"]
