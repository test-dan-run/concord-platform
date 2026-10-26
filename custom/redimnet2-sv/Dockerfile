# redimnet2-sv: OIP v2 speaker embeddings (ReDimNet2-B6) on the GPU. Weights (and, for redimnet2, the model code) are mounted at /models.
# Base has torch 2.11 + torchaudio 2.11 (CUDA 13.0, sm_120), shared with the other phase C services.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.2.0
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
