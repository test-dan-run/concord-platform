# Hosted endpoint: a model a provider already serves online

A hosted endpoint is registered in the Model Catalog only (`catalog/README.md`, "Hosted
endpoints"). There is no manifest, port, VRAM, image or pinned test to update. Apps call it
through the catalog's gateway like any platform model; the gateway adds the stored API key, so apps
never see it.

## Guardrails

- **Data leaves this network.** Creating one requires `"data_leaves_network": true`. Never set it on
  the user's behalf: ask, and say what will be sent (audio, text) and to which host.
- **The key is the user's secret.** Take it only from the user. Never echo it in your report, a
  command you show, a file or a commit. Pass it in the request body only; the API never returns it
  (records carry `key_hint`, its last four characters).
- The catalog must be running (`curl -s localhost:8700/api/health`) and have its master key
  (`~/.config/aiplatform/catalog.key`, created by `just catalog-up`). Without the key, creating one
  returns 503 naming the file.

## 1. Pick the contract

The provider's API must already speak one of these unchanged (the gateway forwards bodies as they are):

| The provider serves | `contract` / `api` | Base URL |
|---|---|---|
| Speech to text | `openai` / `audio.transcriptions` | includes the version path, e.g. `https://api.example.com/v1` |
| Chat (audio LLM or text) | `openai` / `chat.completions` | same |
| Text to speech | `openai` / `audio.speech` | same |
| Embeddings | `openai` / `embeddings` | same |
| Inverse text normalisation | `openai-inspired` / `text.itn` | same |
| VAD, diarization, speech enhancement (OIP v2) | `oip-v2` / `infer` | the server root, e.g. `https://inference.example.com` |

If it fits none (a different request shape, an ITN API with a top-level `language` or array
options), **stop** and tell the user: it cannot be registered until the provider serves the contract.
An `oip-v2` endpoint's signature is read from the provider's metadata (`GET /v2/models/<name>`) at
creation; it decides the probe as in SKILL.md step 2.

## 2. Choose the fields

- `id`: the catalog name apps use. Lower case letters, digits, `.`, `_`, `-`, at most 64; it must not
  be a platform service name or alias (409). Make it say it is hosted, e.g. `hosted-asr-whisper-1`.
- `model`: the provider's model name (what goes in request bodies; `model_name(id)` returns it).
- `auth_type`: `bearer` (`Authorization: Bearer <key>`), `header` (+ `header_name`, e.g. `api-key`)
  or `none`.
- Limits: `timeout_s` (default 120), `max_concurrency` (4), `max_upload_mb` (25). Use the provider's
  documented limits when known.
- `tags`, `display_name`, `parent_id` as for any registration.

## 3. Test, then create

```bash
# a JSON body file you write with the fields above; keep it out of the repo and delete it afterwards
curl -s -X POST localhost:8700/api/hosted/test -H 'content-type: application/json' -d @body.json
```

The test checks TLS, that the key is accepted, and that the endpoint speaks the contract (`GET
/models` lists the model, else one minimal request; OIP: ready and metadata). It stores nothing.
Fix any `fail` line before creating. Then, with `"data_leaves_network": true` once the user agreed:

```bash
curl -s -X POST localhost:8700/api/hosted -H 'content-type: application/json' -d @body.json
curl -s localhost:8700/api/models/<id>   # state "hosted"; capabilities.status pending -> verified | failed
```

Creation queues verification at once: no lease and no GPU. It runs the contract's conformance checks
(without the platform health route) and the same capability probe as a platform service of that API,
sending the check clip or phrases to the provider. A chat model that rejects audio is verified as a
text model. Hosted models are re-verified every 24 hours, because a provider can change the model
behind the same name.

The Add model page (`/add?source=hosted`) does the same in the UI; offer it when the user would rather
paste the key there than into the conversation.

## 4. Changing or removing it

- `PATCH /api/hosted/<id>` with any field. Changing contract, api, base URL or model makes a new build
  and verifies it again; limits and the key do not. A new `key` is tested before it replaces the old one.
- `DELETE /api/hosted/<id>` removes it, its encrypted key and its registration.

## 5. Report

The id, contract, base URL, provider model name and limits; the connection test and verification
lines; that data sent to it leaves the network; and how to call it:

```python
from aicore import client, model_name   # AICORE_CATALOG must be set: hosted models exist only in the catalog
c = client("<id>")
c.audio.transcriptions.create(model=model_name("<id>"), file=open("call.wav", "rb"))
```

Never include the key.
