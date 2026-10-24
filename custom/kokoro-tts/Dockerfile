# kokoro-tts: OpenAI audio.speech on the GPU. Weights are mounted at /models, never baked in.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
# Everything is loaded from /models; never reach out to the Hub at runtime.
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
