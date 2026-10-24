# Personal AI service platform. Run `just` for the recipe list.
# Everything is generated from services/*.yaml; see ../README.md.

set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

py := "uv run --quiet python"

# list recipes
default:
    @just --list --unsorted

# schema + uniqueness + contract checks
validate:
    {{py}} tools/schema.py

# validate, then generate compose.yml, the aicore endpoints map and ui/catalog.html
render:
    {{py}} tools/render.py

# VRAM footprint of every service, or the budget for a set of NAMEs (exit 1 if over util_limit)
vram *names:
    {{py}} tools/vram_check.py {{names}}

# start services by name/alias, one at a time (GPU first, health-gated); stops other GPU services only if VRAM runs out (--no-evict to refuse)
up +names:
    {{py}} tools/stack.py up {{names}}

# stop everything, or only the named services
down *names:
    {{py}} tools/stack.py down {{names}}

# name, kind, contract, VRAM, container state, health, port, lease holders
status:
    {{py}} tools/stack.py status

# GPU service leases held by apps' jobs (`stack.py up --holder`): service, holder, expires in
leases:
    {{py}} tools/stack.py leases

# follow logs for one service
logs name:
    {{py}} tools/stack.py logs {{name}}

# render, then open ui/catalog.html in the default browser
catalog: render
    #!/usr/bin/env bash
    f="{{justfile_directory()}}/ui/catalog.html"
    if command -v wslview >/dev/null; then wslview "$f"
    elif command -v xdg-open >/dev/null; then xdg-open "$f" >/dev/null 2>&1 &
    elif command -v open >/dev/null; then open "$f"
    else echo "open $f in a browser"; fi

# standalone compose block for one service (uses its pushed image, no repo files needed)
snippet name:
    @{{py}} tools/render.py --snippet {{name}}

# custom only: build the image (and its base image if missing)
build name *flags:
    {{py}} tools/images.py build {{name}} {{flags}}

# custom only: build, then push the image and its base to the registry
push name:
    {{py}} tools/images.py push {{name}}

# custom only: download weights into $AI_ASSETS/custom-models/<name>/<version> and verify sha256
fetch-weights name *flags:
    {{py}} tools/fetch_weights.py {{name}} {{flags}}

# run the contract conformance suite against a running service (all services if no NAME); skips when down
conform name="": render
    uv run pytest contracts/conformance -rs {{ if name == "" { "" } else { "--service " + name } }}

# tarball of NAMEs for another machine: manifests, compose, endpoints map, restore README (--name X, --with-images)
bundle +names:
    {{py}} tools/bundle.py {{names}}

# unit tests: platform/tests and custom/*/tests (no GPU, Docker or weights needed)
test:
    uv run pytest

# catalog service (catalog/README.md): build its UI, then run it as a host process on :8700 (development)
catalog-serve: catalog-ui
    {{py}} -m catalog

# The first run creates the master key for hosted endpoints' API keys at ~/.config/aiplatform/catalog.key (0600).
# catalog service as a container on ai-net (restarts with Docker; recreated to pick up code changes); http://127.0.0.1:8700
catalog-up:
    key="${CATALOG_MASTER_KEY_HOST:-$HOME/.config/aiplatform/catalog.key}"; \
    if [ ! -s "$key" ]; then mkdir -p "$(dirname "$key")"; (umask 077; head -c 32 /dev/urandom | base64 | tr '+/' '-_' > "$key"); echo "created the catalog master key at $key"; fi; \
    chmod 600 "$key"; \
    auth="$HOME/.docker/config.json"; \
    if [ ! -f "$auth" ]; then auth="$HOME/.config/aiplatform/docker-config.json"; [ -f "$auth" ] || echo '{}' > "$auth"; fi; \
    data="${AI_DATA:-$HOME/ai-data}"; store="${DATASET_STORE:-${data/#\~/$HOME}/datasets}"; store="${store/#\~/$HOME}"; \
    mkdir -p "$store"; \
    PLATFORM_DIR="{{justfile_directory()}}" CATALOG_UID="$(id -u)" CATALOG_GID="$(id -g)" DOCKER_GID="$(stat -c %g /var/run/docker.sock)" \
        CATALOG_MASTER_KEY_HOST="$key" CATALOG_DOCKER_AUTH_HOST="$auth" DATASET_STORE="$store" \
        docker compose -f catalog/compose.yml up -d --build --force-recreate

# stop the catalog container (model services keep running)
catalog-down:
    PLATFORM_DIR="{{justfile_directory()}}" DOCKER_GID=0 DATASET_STORE=/dev/null docker compose -f catalog/compose.yml down

# follow the catalog container's logs
catalog-logs:
    docker logs -f catalog

# load an Export bundle (folder or .tar) on this host: images, manifests, weights, registrations (catalog/importer.py)
catalog-import bundle *flags:
    {{py}} -m catalog import "{{bundle}}" {{flags}}

# Dataset Catalog clean-up (catalog/datasets/gc.py): a dry run first, then asks before it deletes anything
datasets-gc:
    {{py}} -m catalog.datasets.gc

# build the catalog UI into catalog/ui/dist
catalog-ui:
    cd catalog/ui && npm ci --no-audit --no-fund && npm run build
