# deepfilternet3: OIP v2 speech denoising on CPU. The deep-filter binary and model are mounted at /models, never baked in.
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.2.0
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
