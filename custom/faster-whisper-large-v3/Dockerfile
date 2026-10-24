# faster-whisper-large-v3: OpenAI audio.transcriptions (word timestamps) on CTranslate2.
# GPU via pip CUDA 12 wheels on the slim CPU base (~1.5 GB instead of a CUDA image); weights mounted at /models.
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
