# nemotron-3-diarization: OIP v2 speaker diarization (NeMo Streaming Sortformer, <= 8 speakers) on CPU.
# Weights (.nemo) mounted at /models.
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt constraints.txt /app/
# NeMo's install expects Cython and packaging to be present first (model card instructions).
RUN pip install -c /app/constraints.txt Cython packaging \
 && pip install -r /app/requirements.txt -c /app/constraints.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1 NEMO_LOG_LEVEL=ERROR
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
