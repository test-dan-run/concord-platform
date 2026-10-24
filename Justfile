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

# least GPU memory each service needs to load, or a check of a set of NAMEs (exit 1 with two or more GPU services)
vram *names:
    {{py}} tools/vram_check.py {{names}}

# start services by name/alias, one at a time (GPU first, health-gated); one GPU service at a time, so another running GPU service is stopped (--no-evict to refuse)
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

# upstream model cards (README, LICENSE) at each model's pinned revision into ${AI_ASSETS}/model-cards, for the catalog
fetch-cards *names:
    {{py}} tools/model_cards.py {{names}}

# recommended batch size from the catalog's batch profile (--vram GB, --item-s S or --item-tokens T, --n N)
batch name *flags:
    {{py}} tools/batch.py show {{name}} {{flags}}

# ask the catalog to measure a model's batch profile now (it leases the model; --wait prints the result)
batch-profile name *flags:
    {{py}} tools/batch.py profile {{name}} {{flags}}

# run the contract conformance suite against a running service (all services if no NAME); skips when down
conform name="": render
    uv run pytest contracts/conformance -rs {{ if name == "" { "" } else { "--service " + name } }}

# tarball of NAMEs for another machine: manifests, compose, endpoints map, restore README (--name X, --with-images)
bundle +names:
    {{py}} tools/bundle.py {{names}}

# unit tests: platform/tests and custom/*/tests (no GPU, Docker or weights needed)
test:
    uv run pytest

# model catalog (catalog/README.md): build its UI, then run it as a host process on :8700 (development)
catalog-serve: catalog-ui
    {{py}} -m catalog

# The first run creates the master key for hosted endpoints' API keys at ~/.config/aiplatform/catalog.key (0600).
# model catalog as a container on ai-net (restarts with Docker; recreated to pick up code changes); http://127.0.0.1:8700
catalog-up:
    key="${CATALOG_MASTER_KEY_HOST:-$HOME/.config/aiplatform/catalog.key}"; \
    if [ ! -s "$key" ]; then mkdir -p "$(dirname "$key")"; (umask 077; head -c 32 /dev/urandom | base64 | tr '+/' '-_' > "$key"); echo "created the catalog master key at $key"; fi; \
    chmod 600 "$key"; \
    auth="$HOME/.docker/config.json"; \
    if [ ! -f "$auth" ]; then auth="$HOME/.config/aiplatform/docker-config.json"; [ -f "$auth" ] || echo '{}' > "$auth"; fi; \
    PLATFORM_DIR="{{justfile_directory()}}" CATALOG_UID="$(id -u)" CATALOG_GID="$(id -g)" DOCKER_GID="$(stat -c %g /var/run/docker.sock)" \
        CATALOG_MASTER_KEY_HOST="$key" CATALOG_DOCKER_AUTH_HOST="$auth" \
        docker compose -f catalog/compose.yml up -d --build --force-recreate

# stop the catalog container (model services keep running)
catalog-down:
    PLATFORM_DIR="{{justfile_directory()}}" DOCKER_GID=0 docker compose -f catalog/compose.yml down

# follow the catalog container's logs
catalog-logs:
    docker logs -f catalog

# load an Export bundle (folder or .tar) on this host: images, manifests, weights, registrations (catalog/importer.py)
catalog-import bundle *flags:
    {{py}} -m catalog import "{{bundle}}" {{flags}}

# build the catalog UI into catalog/ui/dist
catalog-ui:
    cd catalog/ui && npm ci --no-audit --no-fund && npm run build
