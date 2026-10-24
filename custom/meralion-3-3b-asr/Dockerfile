# meralion-3-3b-asr: OpenAI audio.transcriptions, MERaLiON-3-3B-ASR on transformers (GPU, bf16).
# Weights and the model's remote code are mounted at /models, never baked in.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-torch211-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
