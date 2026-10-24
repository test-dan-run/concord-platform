# monkey-custom: openai-inspired text.transform transcript normaliser on CPU (transcript-normer @e98d4ae).
# The punctuation model is mounted at /models, never baked in (the project's own Dockerfile bakes it).
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.1.1
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
ENV WEIGHTS_DIR=/models
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
