# syntax=docker/dockerfile:1
# The model catalog as a container: `just catalog-up` (catalog/compose.yml builds and runs it).
#
# Build context is the platform directory. At run time the same directory is mounted at the same path
# (PLATFORM_DIR), so the catalog reads the live manifests and `docker compose` resolves the generated
# compose file's relative build contexts exactly as on the host.

ARG PYTHON=python:3.12.15-slim-bookworm
ARG NODE=node:22.23.3-bookworm-slim

FROM ${NODE} AS web
WORKDIR /web
COPY catalog/ui/package.json catalog/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY catalog/ui/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM docker:29-cli AS dockercli

FROM ${PYTHON}
ARG PLATFORM_DIR=/platform
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH HOME=/tmp PLATFORM_DIR=${PLATFORM_DIR}
COPY --from=uv /uv /usr/local/bin/uv
# docker CLI + compose plugin: the catalog starts and stops model services on the host's daemon (socket mounted)
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
WORKDIR ${PLATFORM_DIR}
# aicore and aisvc are editable path dependencies: installed against PLATFORM_DIR, which is mounted at run time.
COPY pyproject.toml uv.lock ./
COPY clients/aicore clients/aicore
COPY custom/_bases/common custom/_bases/common
RUN uv sync --frozen --no-dev --no-install-project && chmod -R a+rX /opt/venv
COPY --from=web /web/dist /opt/catalog-ui
ENV CATALOG_UI_DIR=/opt/catalog-ui CATALOG_BIND=0.0.0.0 CATALOG_UPSTREAM_HOST={name}
EXPOSE 8700
CMD ["python", "-m", "catalog"]
