# wpe-dereverb: OIP v2 dereverberation utility on CPU (nara_wpe, no model weights).
ARG BASE_IMAGE=ghcr.io/owner/py-cpu-base:0.2.0
FROM ${BASE_IMAGE}

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY server/ /app/server/
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
