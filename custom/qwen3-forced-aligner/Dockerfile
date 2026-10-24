# qwen3-forced-aligner: OIP v2 forced alignment (Qwen3-ForcedAligner-0.6B) on the GPU. Weights mounted at /models.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && pip install --no-deps qwen-asr==0.0.6

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
