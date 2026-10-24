# API contracts

Every service, model or utility, follows exactly one contract. The manifest declares it
(`contract`, `api`), `tools/schema.py` enforces it, `contracts/conformance/` tests it, and
`aicore.client(name)` returns the matching client.

## 1. Selection order

Pick the **first** contract that applies:

| # | Contract | When | Examples |
|---|---|---|---|
| 1 | `openai` | An OpenAI API route fits the task | chat, completions, embeddings, transcription (Whisper), speech (Kokoro) |
| 2 | `oip-v2` | No OpenAI route fits **and** any input is non-textual (audio arrays, images, tensors, tabular features) | VAD, diarisation, image classification, keyword spotting; utilities such as resampling, image resizing, feature extraction |
| 3 | `openai-inspired` | No OpenAI route fits **and** every input is text | text normaliser / punctuation, language ID on text, PII redaction |

Rules:

- Non-text input with no OpenAI route always means `oip-v2`, for utilities as well as models.
  An `openai-inspired` API never takes binary, array or file input.
- Every non-`openai` manifest states why in `contract_reason`
  (e.g. `"No OpenAI route for VAD; input is a float audio array"`).
- Reuse an existing `openai-inspired` API type when one fits (`text.transform` for any text-in/text-out
  utility, `text.itn` for spoken-form to written-form conversion). A new type needs a spec section below, an entry in `api_types.yaml`, and a conformance suite
  in `conformance/checks.py`, all before any manifest uses it.

`api_types.yaml` is the machine-readable registry the schema validates against.

## 2. Universal requirement: `GET /health`

Every service, whatever its contract, serves `GET /health`: **200** when ready, **503** while
loading (or after a failed load, with the error in the body). The platform uses it for compose
healthchecks, `just up` gating and the catalog status dots. Engines point `health_path` in
`engines/engines.yaml` at their native equivalent (vLLM and SGLang both use `/health`).
Custom services get this from `aisvc`.

---

## 3. `openai`

Follow the official OpenAI API shapes for the declared `api`. Every service also serves
`GET /v1/models`, listing the service `name` as a model id. Request bodies use `model: <name>`
(`aicore.model_name(alias)` returns it).

| `api` | Route | Request | Response |
|---|---|---|---|
| `chat.completions` | `POST /v1/chat/completions` | `model`, `messages[]`, `max_tokens`, `temperature`, ... | `chat.completion` object with `choices[].message` |
| `completions` | `POST /v1/completions` | `model`, `prompt`, ... | `text_completion` with `choices[].text` |
| `embeddings` | `POST /v1/embeddings` | `model`, `input` (string or list) | `list` of `{object: "embedding", index, embedding: float[]}` |
| `audio.transcriptions` | `POST /v1/audio/transcriptions` (multipart) | `file`, `model`, `language?`, `prompt?`, `response_format` (`json`/`text`/`verbose_json`) | `{"text": ...}` |
| `audio.speech` | `POST /v1/audio/speech` | `model`, `input`, `voice`, `response_format`, `speed` | audio bytes with the matching `content-type` (`audio/wav`, `audio/mpeg`, ...) |

Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`. `aisvc` custom servers
return 400 for invalid requests, 404 `model_not_found` for a wrong `model`, and 503 `not_ready`
while loading. Streaming is engine-dependent; `aisvc` servers do not stream.

**Conformance:** `/health` 200; `/v1/models` lists the name; chat returns a non-empty completion
(content or reasoning); transcription of `fixtures/tone_1s_16k.wav` (32 KB) returns a `text`
string; embeddings return vectors of one consistent dimension; speech (`response_format: wav`)
returns a non-empty RIFF body with an `audio/wav` content type.

---

## 4. `oip-v2`: KServe Open Inference Protocol V2

Source followed: the KServe *Open Inference Protocol (V2)* REST spec,
<https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol>
(same text as <https://github.com/kserve/open-inference-protocol>), checked 2026-10-17.
JSON tensor encoding is required; the binary tensor extension is not implemented.

| Route | Response (200) |
|---|---|
| `GET /v2/health/live` | `{"live": true}` |
| `GET /v2/health/ready` | `{"ready": true}`; **503** with `{"ready": false}` while loading |
| `GET /v2` | server metadata `{"name", "version", "extensions": []}` |
| `GET /v2/models/{name}[/versions/{v}]` | model metadata `{"name", "versions", "platform", "inputs", "outputs"}`; each tensor `{name, datatype, shape}` with `-1` for variable dims |
| `GET /v2/models/{name}[/versions/{v}]/ready` | `{"name", "ready": true}`; **503** while loading |
| `POST /v2/models/{name}[/versions/{v}]/infer` | inference response (below) |

Inference request:

```json
{"id": "optional", "parameters": {"threshold": 0.6},
 "inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [16000], "data": [0.0, 0.01, ...]},
            {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
 "outputs": [{"name": "SEGMENTS"}]}
```

- `data` is row-major and either flat or nested to match `shape`.
- Datatypes: `BOOL`, `UINT8`, `UINT16`, `UINT32`, `UINT64`, `INT8`, `INT16`, `INT32`, `INT64`, `FP16`,
  `FP32`, `FP64`, `BYTES` (JSON strings).
- `outputs` is optional; without it every output is returned.

Inference response:

```json
{"model_name": "silero-vad", "model_version": "6.2.3", "id": "optional (echoed)",
 "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [1, 2], "data": [0.96, 2.53]}]}
```

Errors: `{"error": "<message>"}`, with **400** for a malformed request or a tensor that does not
match the signature (unknown or missing input, wrong datatype, incompatible shape, data count ≠
shape product), **404** for an unknown model or version, **503** when not ready, and **500** for
a server-side failure. (The spec defines the error shape and "typically 400"; the other codes
are this platform's convention.)

**Signature.** The manifest's `signature` declares inputs and outputs (`name`, `datatype`,
`shape`, optional `example` value). The server's model-metadata response must match it, and the
conformance suite checks that. `aicore.OIPClient` uses it to validate numpy inputs before
sending.

**Conformance:** `/health` 200; live/ready endpoints; server metadata keys; model metadata equals
the manifest signature; an infer request built from the signature (each input's `example`, or
zeros with `-1` → 16000) returns every declared output with the declared datatype, a compatible
shape, and the right number of values, and echoes `id`; sending the first input with the wrong
datatype returns 400 with `{"error": "..."}`.

---

## 5. `openai-inspired`

For text-input services with no OpenAI route. Conventions:

- Routes under `/v1/` with resource-style nouns: `POST /v1/{resource}` (e.g. `/v1/text/transform`).
- `GET /v1/models` lists the service name in OpenAI format.
- Request: JSON with `model`, `input` (a string or a non-empty list of strings), and optional
  `options` (an object of **scalar**, service-specific settings). Unknown top-level fields are
  rejected. Services validate their own `options` and reject unknown ones with 400 `invalid_options`.
- Response: JSON with `object`, `model`, `created` (unix seconds), and a `data` list of
  `{object, index, ...}` items in input order, mirroring the OpenAI embeddings response, plus
  `usage` when meaningful.
- Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`: 400 `invalid_request` /
  `invalid_options`, 404 `model_not_found`, 503 `not_ready`, 500 `server_error`.

### API type `text.transform`

Text in, text out, one output per input. Reuse it for every text-to-text utility (normalisation,
punctuation, redaction, transliteration, ...).

```
POST /v1/text/transform
{"model": "text-normalizer", "input": ["hello world how are you"], "options": {}}

200 {"object": "list", "model": "text-normalizer", "created": 1760700000,
     "data": [{"object": "text.transform", "index": 0, "output": "Hello world how are you."}],
     "usage": {"input_chars": 23, "output_chars": 24}}
```

A string `input` yields a single `data` item; a list yields one item per element, in order.

**Conformance:** `/health` 200; `/v1/models` lists the name; string and list inputs return a
`data` list of matching length with `object: "text.transform"`, sequential `index` and string
`output`; a malformed request (`input: 123`, or `input` missing) returns an OpenAI-shaped error.

### API type `text.itn`

Inverse text normalisation (ITN): spoken-form text, as ASR writes it, in; written form out, one
output per input (`three thirty p m` → `3:30pm`). Adapted to the conventions above from an external
ITN API (v1.0.0); the full schema is [`openapi/text.itn.yaml`](openapi/text.itn.yaml).

```
POST /v1/text/itn
{"model": "itn-en", "input": ["meet at three thirty p m on the twelfth of october"],
 "options": {"language": "en-SG", "return_spans": true, "classes": "TIME,DATE"}}

200 {"object": "list", "model": "itn-en", "created": 1760700000,
     "data": [{"object": "text.itn", "index": 0, "output": "meet at 3:30pm on 12 October",
               "spans": [{"input_start": 8, "input_end": 24, "output_start": 8, "output_end": 14, "class": "TIME"},
                         {"input_start": 28, "input_end": 50, "output_start": 18, "output_end": 28, "class": "DATE"}]}],
     "usage": {"input_chars": 50, "output_chars": 28}}
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `language` | string, a BCP-47 locale | the service's default locale | Output conventions such as date order and currency symbols. A locale the service does not support is a 400 `invalid_options`. |
| `return_spans` | boolean | `false` | Every `data` item gets `spans` (an empty list when nothing was converted). |
| `classes` | string, comma-separated semiotic classes | every class the service supports | Converts only these classes. A class the service does not support is a 400 `invalid_options`. |

- **Spans** are zero-based, half-open `[start, end)` offsets in Unicode code points into that item's
  input and output, sorted by `input_start` and non-overlapping. `class` is a semiotic class:
  `CARDINAL`, `ORDINAL`, `DECIMAL`, `FRACTION`, `MONEY`, `MEASURE`, `DATE`, `TIME`, `TELEPHONE`,
  `ELECTRONIC`. Services may add classes, so clients must accept values they do not know.
- **Limits:** at most 256 inputs per request, each at most 10,000 characters; more is a 400 `invalid_request`.
- Output is deterministic for a given build.

Changes from the source spec, so a provider that serves the source spec unchanged does not conform:
the route `/v1/itn` becomes `/v1/text/itn`; `language` moves from the top level into `options`;
`model` is required; `classes` is a comma-separated string, not an array; unknown options are
rejected, not ignored; `input` may also be a single string; items carry `object` and `output`
(was `text`); the response adds `object`, `created` and `usage`; an unsupported language (was 404) and
oversized input (was 413) are 400s; and the service serves `/health` and `/v1/models`.

**Conformance:** `/health` 200; `/v1/models` lists the name; string and list inputs return a `data`
list of matching length with `object: "text.itn"`, sequential `index`, string `output` and no
`spans`; `return_spans: true` adds valid spans to every item (in bounds, sorted, non-overlapping, a
non-empty `class`); `classes: "CARDINAL"` is accepted; the same request twice returns the same
output; an unknown option is a 400 `invalid_options`; a malformed request (`input: 123`, `input`
missing, `classes` as a list) returns an OpenAI-shaped error.

---

## 6. Running the suites

```bash
just conform NAME          # one service (name or alias), against http://{AI_HOST}:{port}
just conform               # every service in the endpoints map
```

Services that are not reachable, or still loading, are **skipped** with a message. The checks live
in `conformance/checks.py` as plain functions `check(client, meta)`, so custom services reuse them
in-process with FastAPI's `TestClient`:

```python
from conformance import checks_for, load_server, running, service_meta

META = service_meta("text-normalizer")            # endpoint entry built from the manifest
app = load_server(SERVICE_DIR, "app").build_app()
with running(app) as client:                      # lifespan started, model loaded
    for check in checks_for(META):
        check(client, META)
```
