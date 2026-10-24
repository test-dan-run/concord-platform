# voxcpm2: OIP v2 voice cloning (VoxCPM2, 48 kHz out) on the GPU. Weights are mounted at /models, never baked in.
# Base has torch 2.11 + torchaudio 2.11 (CUDA 13.0, sm_120); voxcpm imports torchaudio.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.1.1
FROM ${BASE_IMAGE}

# torch.compile (on by default, VOXCPM_OPTIMIZE): Triton builds its CUDA launcher with gcc against Python's headers.
# Measured on the 5090: RTF 1.17 -> 0.22 once warm, for a ~80 s compile at start-up.
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libc6-dev python3.12-dev \
 && rm -rf /var/lib/apt/lists/*
USER 1000:1000

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
 && pip install --no-deps voxcpm==2.0.3

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VOXCPM_OPTIMIZE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
