# pyannote-diarization: OIP v2 speaker diarization (pyannote speaker-diarization-3.1) on the GPU.
# Base has torch 2.11 + torchaudio (the last torchaudio release). Weights mounted at /models.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt constraints.txt /app/
RUN pip install -r /app/requirements.txt -c /app/constraints.txt

COPY server/ /app/server/
# Audio arrives in memory, so torchcodec/FFmpeg file decoding is never used (pyannote warns and continues).
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 PYTHONWARNINGS="ignore::UserWarning:pyannote.audio.core.io"
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
