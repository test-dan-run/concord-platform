# utmos: OIP v2 speech naturalness (UTMOS22 strong learner) on CPU. Weights and model code are mounted at
# /models, never baked in.
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1
FROM ${BASE_IMAGE}

# CPU-only torch wheels (the py-cpu base has no torch); torchaudio 2.11 is the last release, so torch matches it.
RUN pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
