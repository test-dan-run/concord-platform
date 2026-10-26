# AGENTS.md

A personal AI service platform for one host with one GPU. Each model or utility is a Docker service defined by one
YAML manifest in `services/`, and `tools/` generates the compose file, the client endpoint map and a static service
page from those manifests. The catalog service (`catalog/`, port 8700) is the control plane: it registers, verifies,
schedules and proxies the model services (the Model Catalog) and stores versioned audio datasets (the Dataset
Catalog).

## Stack and layout

Python 3.12 with uv and just. The catalog is FastAPI with SQLite. Its UI is React 19, TypeScript, Vite and
Tailwind. Model services run in Docker, either on a pinned engine (vLLM, SGLang) or as a custom FastAPI image.

| Path | What it holds |
|---|---|
| `services/*.yaml` | One manifest per service. These are the source of truth. |
| `config.yaml`, `engines/engines.yaml` | GPU size, `util_limit`, registry; pinned engine images and command templates. |
| `tools/` | CLI scripts behind the `just` recipes: `schema.py` (validation), `render.py`, `stack.py` (up/down/leases), `images.py`, `vram_check.py`, `batch.py`, `bundle.py`, `fetch_weights.py`, `model_cards.py`. |
| `contracts/` | The three API contracts (`README.md`, `api_types.yaml`), the live conformance suite (`conformance/`) and the dataset version format (`datasets/`). |
| `catalog/` | The catalog service: scheduler, gateway, verifier, hosted endpoints, export and import, `datasets/`, and the UI in `ui/`. `catalog/README.md` is its API contract. |
| `custom/<name>/` | A custom service image: `Dockerfile`, `server/app.py`, `tests/`, `README.md`, and for a model `weights.yaml` and `WHY.md`. |
| `custom/_bases/` | Base images (`bases.yaml`) and `common/`, the `aisvc` package that every custom server is built on. |
| `clients/aicore/` | The `aicore` client package that apps install. |
| `tests/` | Unit tests for `tools/`, `aicore` and `aisvc`. Catalog tests are in `catalog/tests/`. |
| `.claude/skills/register-model/` | The step-by-step procedure for adding a model. Use it when adding or changing a service. |

## Commands

Run these from the repo root. Every command below was run and works. `just` with no arguments lists every recipe.

```bash
cp .env.example .env     # set AI_ASSETS (the weights folder)
uv sync                  # Python venv, including the editable aicore and aisvc packages
just validate            # schema, uniqueness and contract checks, plus the dataset version samples
just render              # validate, then write the generated files
just vram                # least GPU memory per service; `just vram A B` checks a set
just test                # Python unit tests (about two minutes, no GPU, Docker or weights needed)
just conform [NAME]      # contract checks against running services; a service that is down is skipped
just catalog-ui          # npm ci and build the UI into catalog/ui/dist
cd catalog/ui && npm run lint && npm test   # UI type check (tsc) and vitest
just catalog-serve       # catalog as a host process on :8700 (development)
just catalog-up          # catalog as a container on the ai-net network
just status              # every service: state, health, port and lease holders
```

No Python linter or formatter is configured. The UI's `lint` script is a TypeScript type check.

`just up NAME...`, `just build NAME`, `just push NAME` and `just fetch-weights NAME` pull images, build images,
push to the registry or download weights. Ask before you run them.

## Conventions

- **Manifests.** Service names are kebab case, so `qwen2.5` becomes `qwen2-5`. Each kind of service has its own
  port range: 81xx audio models, 82xx chat and vision models, 83xx embeddings, 84xx other models (usually
  oip-v2), 85xx utilities. Comments in a manifest record how a GPU figure was measured and when. A flag you
  could not confirm for the pinned engine gets `# TODO(verify): <what to check>`.
- **Contracts.** Every service follows exactly one contract, chosen in this order: `openai` when an OpenAI route
  fits; `oip-v2` when any input is not text; `openai-inspired` when every input is text. Every manifest that is not
  `openai` says why in `contract_reason`. Every service serves `GET /health` (503 while it loads). See
  `contracts/README.md`.
- **Custom servers** expose `server/app.py:build_app()`, which uvicorn runs with `--factory` on port 8000 inside the
  container. They use the `aisvc` app factories (`create_openai_app`, `create_openai_inspired_app`,
  `create_oip_app`), raise `ValueError` for bad input (400), and batch with `aisvc.MicroBatcher` or
  `DynamicBatcher`. Audio lengths given to a batcher are in milliseconds. Weights are mounted read-only at
  `WEIGHTS_DIR` (`/models`) and pinned by sha256 in `weights.yaml`. A custom service that runs a model also has a
  `WHY.md` saying why no stock engine can serve it.
- **Python.** Type hints throughout. Each module starts with a docstring that states its rules, and a CLI tool's
  docstring also gives its usage lines. Tools run as scripts (`uv run python tools/x.py`), not with `python -m`.
- **Tests.** Plain pytest functions with no GPU, Docker or weights needed. A test that needs real weights is marked
  `@pytest.mark.weights` and skips when they are missing. Custom service tests live in `custom/<name>/tests/` and
  run the server in process through `TestClient` with the shared conformance checks.
- **Commit messages** are one plain sentence in the imperative ("Keep the old tarball when a bundle fails while it
  is written"), with no type prefix.

## Gotchas

- **Never edit generated files.** `compose/generated/compose.yml`, `clients/aicore/src/aicore/endpoints.generated.yaml`
  and `ui/catalog.html` come from `just render` and are gitignored. Change the manifest instead.
- **Tests pin the docs and the registry.** `tests/test_readme.py` checks that the service tables in `README.md`
  match the manifests, column for column. `tests/test_schema.py` pins the list of service names and aliases.
  `catalog/tests/test_readme_routes.py` checks that every catalog API route is in `catalog/README.md`. Adding,
  renaming or changing a service, or adding a route, means updating these.
- **Some rules are copied on purpose.** The aicore base paths, the aisvc routes and the verifier's probe routes
  repeat `contracts/api_types.yaml`, because each ships without the others. `tests/test_rules_kept_in_two_places.py`
  catches drift. `server/rows.py` and `server/turns.py` are copied into several images and must stay identical
  (`SHARED` in `tests/test_images.py`). Change every copy together.
- **A new `openai-inspired` API type** needs a spec section in `contracts/README.md`, an entry in
  `contracts/api_types.yaml` and a check in `contracts/conformance/checks.py` before any manifest uses it.
- **Old base images.** kokoro-tts, voxcpm2, silero-vad, text-normalizer and monkey-custom pin `base_version: "0.1.1"`.
  That base ships aisvc 0.1.0, so aisvc changes don't reach them. kokoro-tts and voxcpm2 must not change.
- **One GPU service at a time.** It gets the free GPU memory less 1 GiB, at most `util_limit`. `just up` stops any
  other running GPU service; CPU services are never stopped. When the catalog is running, `tools/stack.py` goes
  through the catalog's API, and `AI_STACK_DIRECT=1` skips it.
- **Batch limits only reach GPU services.** `AISVC_MAX_BATCH` and `AISVC_MAX_BATCH_LEN` come from the catalog's
  batch profile, and only GPU custom services get them. CPU services use the defaults in their images.
- **A rule-based service's `README.md` is its model card.** For a custom service with no weights (text-normalizer,
  wpe-dereverb) the catalog serves that file as the card (`catalog/cards.py`), so edits there show in the UI.
- **Don't delete the dataset store.** `run/catalog/catalog.db` can be rebuilt. `${DATASET_STORE}/datasets.db` and
  the audio store cannot.
- **No `:latest` tags, no weights in images or git, no auth.** Every port binds to `127.0.0.1`.
