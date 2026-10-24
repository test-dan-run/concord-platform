"""Conformance checks, one function per requirement.

Every check is `check(client, meta)`: `client` is an httpx.Client (or FastAPI TestClient, which is
one) whose base_url is the service root, and `meta` is the service's endpoint-map entry (name,
contract, api, health_path, signature, ...). `checks_for(meta)` picks the checks for a service.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

Check = Callable[[httpx.Client, dict[str, Any]], None]
FIXTURES = Path(__file__).with_name("fixtures")
TONE_WAV = FIXTURES / "tone_1s_16k.wav"


def _json(r: httpx.Response, status: int = 200) -> Any:
    assert r.status_code == status, f"{r.request.method} {r.request.url.path} -> {r.status_code} (want {status}): {r.text[:500]}"
    return r.json()


def _openai_error(r: httpx.Response) -> None:
    assert 400 <= r.status_code < 500, f"expected a 4xx, got {r.status_code}: {r.text[:300]}"
    err = r.json().get("error")
    assert isinstance(err, dict), f"error must be an object {{message, type, code}}, got {r.text[:300]}"
    for key in ("message", "type", "code"):
        assert key in err, f"error object lacks {key!r}: {err}"
    assert isinstance(err["message"], str) and err["message"]


# ----------------------------------------------------------------------------- universal

def health_ok(c: httpx.Client, meta: dict) -> None:
    r = c.get(meta.get("health_path", "/health"))
    assert r.status_code == 200, f"{meta.get('health_path', '/health')} -> {r.status_code}: {r.text[:300]}"


def models_lists_name(c: httpx.Client, meta: dict) -> None:
    data = _json(c.get("/v1/models"))
    assert data.get("object") == "list"
    ids = [m.get("id") for m in data.get("data", [])]
    assert meta["name"] in ids, f"/v1/models lists {ids}, expected {meta['name']!r}"


# ----------------------------------------------------------------------------- openai

def chat_completion(c: httpx.Client, meta: dict) -> None:
    body = {"model": meta["name"], "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "max_tokens": 64, "temperature": 0}
    data = _json(c.post("/v1/chat/completions", json=body, timeout=300))
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")
    assert text.strip(), f"empty completion: {data}"


def completion(c: httpx.Client, meta: dict) -> None:
    data = _json(c.post("/v1/completions", json={"model": meta["name"], "prompt": "Once upon a time",
                                                  "max_tokens": 16, "temperature": 0}, timeout=300))
    assert data["choices"][0]["text"], f"empty completion: {data}"


def embeddings_consistent(c: httpx.Client, meta: dict) -> None:
    data = _json(c.post("/v1/embeddings", json={"model": meta["name"], "input": ["hello world", "a second, longer sentence"]},
                        timeout=120))
    vecs = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    assert len(vecs) == 2, f"expected 2 embeddings, got {len(vecs)}"
    dims = {len(v) for v in vecs}
    assert len(dims) == 1 and dims.pop() > 0, f"inconsistent dimensions {[len(v) for v in vecs]}"
    assert all(isinstance(x, float) and math.isfinite(x) for v in vecs for x in v)


def transcription_text(c: httpx.Client, meta: dict) -> None:
    with open(TONE_WAV, "rb") as fh:
        r = c.post("/v1/audio/transcriptions", data={"model": meta["name"]},
                   files={"file": (TONE_WAV.name, fh, "audio/wav")}, timeout=300)
    data = _json(r)
    assert isinstance(data.get("text"), str), f"no text field: {data}"


def speech_audio(c: httpx.Client, meta: dict) -> None:
    body = {"model": meta["name"], "input": "Hello from the conformance suite.", "voice": "alloy", "response_format": "wav"}
    r = c.post("/v1/audio/speech", json=body, timeout=300)
    assert r.status_code == 200, f"/v1/audio/speech -> {r.status_code}: {r.text[:300]}"
    ctype = r.headers.get("content-type", "")
    assert ctype.split(";")[0] in ("audio/wav", "audio/x-wav", "audio/wave"), f"content-type {ctype!r} for wav"
    assert len(r.content) > 44 and r.content[:4] == b"RIFF", "body is not a non-empty WAV file"


OPENAI_API_CHECKS: dict[str, list[Check]] = {
    "chat.completions": [chat_completion],
    "completions": [completion],
    "embeddings": [embeddings_consistent],
    "audio.transcriptions": [transcription_text],
    "audio.speech": [speech_audio],
}


# ----------------------------------------------------------------------------- oip-v2

def _model_path(meta: dict) -> str:
    return f"/v2/models/{meta['name']}"


def oip_live_ready(c: httpx.Client, meta: dict) -> None:
    assert _json(c.get("/v2/health/live")).get("live") is True
    assert _json(c.get("/v2/health/ready")).get("ready") is True
    ready = _json(c.get(f"{_model_path(meta)}/ready"))
    assert ready.get("ready") is True and ready.get("name") == meta["name"], ready


def oip_server_metadata(c: httpx.Client, meta: dict) -> None:
    data = _json(c.get("/v2"))
    for key in ("name", "version", "extensions"):
        assert key in data, f"server metadata lacks {key!r}: {data}"


def _tensor_set(ts: list[dict]) -> dict[str, tuple[str, list[int]]]:
    return {t["name"]: (t["datatype"], list(t["shape"])) for t in ts}


def oip_metadata_matches_signature(c: httpx.Client, meta: dict) -> None:
    data = _json(c.get(_model_path(meta)))
    assert data.get("name") == meta["name"], data
    for side in ("inputs", "outputs"):
        got, want = _tensor_set(data.get(side, [])), _tensor_set(meta["signature"][side])
        assert got == want, f"model metadata {side} {got} != manifest signature {want}"


def _example_inputs(signature: dict) -> list[dict]:
    """Inputs built from the signature: `example` values if declared, else zeros with -1 -> 16000."""
    inputs = []
    for t in signature["inputs"]:
        if t.get("example") is not None and len(t["shape"]) > 1 and isinstance(t["example"][0], list):
            # nested example (batch-shaped inputs such as AUDIO [-1, -1]): its nesting gives the shape
            rows = [list(r) for r in t["example"]]
            shape = [len(rows), len(rows[0])]
            data = [x for r in rows for x in r]
        elif t.get("example") is not None:
            data = list(t["example"])
            shape = [len(data) if d == -1 else d for d in t["shape"]]
        else:
            shape = [16000 if d == -1 else d for d in t["shape"]]
            n = math.prod(shape)
            data = [""] * n if t["datatype"] == "BYTES" else [False] * n if t["datatype"] == "BOOL" else [0] * n
        inputs.append({"name": t["name"], "datatype": t["datatype"], "shape": shape, "data": data})
    return inputs


def oip_infer_from_signature(c: httpx.Client, meta: dict) -> None:
    sig = meta["signature"]
    data = _json(c.post(f"{_model_path(meta)}/infer", json={"id": "conformance-1", "inputs": _example_inputs(sig)},
                        timeout=300))
    assert data.get("model_name") == meta["name"], data
    assert data.get("id") == "conformance-1", "the request id must be echoed"
    outs = {o["name"]: o for o in data.get("outputs", [])}
    for spec in sig["outputs"]:
        assert spec["name"] in outs, f"missing output {spec['name']!r}; got {sorted(outs)}"
        o = outs[spec["name"]]
        assert o["datatype"] == spec["datatype"], f"{spec['name']}: datatype {o['datatype']} != {spec['datatype']}"
        assert len(o["shape"]) == len(spec["shape"]) and all(d == -1 or d == a for d, a in zip(spec["shape"], o["shape"])), \
            f"{spec['name']}: shape {o['shape']} incompatible with {spec['shape']}"
        flat = o["data"]
        while flat and isinstance(flat[0], list):
            flat = [x for row in flat for x in row]
        assert len(flat) == math.prod(o["shape"]), f"{spec['name']}: {len(flat)} values for shape {o['shape']}"


def oip_wrong_datatype_400(c: httpx.Client, meta: dict) -> None:
    inputs = _example_inputs(meta["signature"])
    first = inputs[0]
    first["datatype"] = "INT32" if first["datatype"] != "INT32" else "FP32"
    first["data"] = [0] * len(first["data"])
    r = c.post(f"{_model_path(meta)}/infer", json={"inputs": inputs})
    assert r.status_code == 400, f"wrong datatype -> {r.status_code}, want 400: {r.text[:300]}"
    err = r.json().get("error")
    assert isinstance(err, str) and err, f"error must be {{'error': '<message>'}}, got {r.text[:300]}"


# ----------------------------------------------------------------------------- openai-inspired

def text_transform_string(c: httpx.Client, meta: dict) -> None:
    data = _json(c.post("/v1/text/transform", json={"model": meta["name"], "input": "hello world  how are you", "options": {}}))
    assert data.get("object") == "list" and data.get("model") == meta["name"], data
    assert isinstance(data.get("created"), int)
    assert len(data["data"]) == 1
    item = data["data"][0]
    assert item.get("object") == "text.transform" and item.get("index") == 0 and isinstance(item.get("output"), str), item


def text_transform_list(c: httpx.Client, meta: dict) -> None:
    texts = ["first one", "second  one", "third"]
    data = _json(c.post("/v1/text/transform", json={"model": meta["name"], "input": texts}))
    assert [d["index"] for d in data["data"]] == [0, 1, 2], data
    assert all(isinstance(d["output"], str) for d in data["data"])


def text_transform_malformed(c: httpx.Client, meta: dict) -> None:
    _openai_error(c.post("/v1/text/transform", json={"model": meta["name"], "input": 123}))
    _openai_error(c.post("/v1/text/transform", json={"model": meta["name"]}))


ITN_SAMPLE = "meet at three thirty p m on the twelfth of october"


def _itn(c: httpx.Client, meta: dict, input: Any, **options: Any) -> httpx.Response:
    return c.post("/v1/text/itn", json={"model": meta["name"], "input": input, "options": options})


def _check_spans(item: dict, text: str) -> None:
    spans = item.get("spans")
    assert isinstance(spans, list), f"return_spans: true but item has no spans list: {item}"
    out, last = item["output"], 0
    for s in spans:
        for k in ("input_start", "input_end", "output_start", "output_end"):
            assert isinstance(s.get(k), int), f"span {k} must be an int: {s}"
        assert last <= s["input_start"] <= s["input_end"] <= len(text), f"span outside the input or out of order: {s}"
        assert 0 <= s["output_start"] <= s["output_end"] <= len(out), f"span outside the output: {s}"
        assert isinstance(s.get("class"), str) and s["class"], f"span class must be a non-empty string: {s}"
        last = s["input_end"]


def text_itn_string(c: httpx.Client, meta: dict) -> None:
    data = _json(_itn(c, meta, ITN_SAMPLE))
    assert data.get("object") == "list" and data.get("model") == meta["name"], data
    assert isinstance(data.get("created"), int)
    assert len(data["data"]) == 1
    item = data["data"][0]
    assert item.get("object") == "text.itn" and item.get("index") == 0 and isinstance(item.get("output"), str), item
    assert "spans" not in item, f"spans returned without return_spans: {item}"


def text_itn_list(c: httpx.Client, meta: dict) -> None:
    texts = [ITN_SAMPLE, "twenty one", "hello"]
    data = _json(_itn(c, meta, texts))
    assert [d["index"] for d in data["data"]] == [0, 1, 2], data
    assert all(d.get("object") == "text.itn" and isinstance(d["output"], str) for d in data["data"])


def text_itn_spans(c: httpx.Client, meta: dict) -> None:
    texts = [ITN_SAMPLE, "twenty one"]
    data = _json(_itn(c, meta, texts, return_spans=True))
    for item, text in zip(sorted(data["data"], key=lambda d: d["index"]), texts, strict=True):
        _check_spans(item, text)


def text_itn_classes(c: httpx.Client, meta: dict) -> None:
    _json(_itn(c, meta, ITN_SAMPLE, classes="CARDINAL"))


def text_itn_deterministic(c: httpx.Client, meta: dict) -> None:
    first = _json(_itn(c, meta, ITN_SAMPLE, return_spans=True))["data"]
    second = _json(_itn(c, meta, ITN_SAMPLE, return_spans=True))["data"]
    assert first == second, f"same request, different output: {first} vs {second}"


def text_itn_unknown_option(c: httpx.Client, meta: dict) -> None:
    r = _itn(c, meta, ITN_SAMPLE, no_such_option=True)
    _openai_error(r)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options", r.text[:300]


def text_itn_malformed(c: httpx.Client, meta: dict) -> None:
    _openai_error(c.post("/v1/text/itn", json={"model": meta["name"], "input": 123}))
    _openai_error(c.post("/v1/text/itn", json={"model": meta["name"]}))
    _openai_error(_itn(c, meta, ITN_SAMPLE, classes=["TIME", "DATE"]))


INSPIRED_API_CHECKS: dict[str, list[Check]] = {
    "text.transform": [text_transform_string, text_transform_list, text_transform_malformed],
    "text.itn": [text_itn_string, text_itn_list, text_itn_spans, text_itn_classes, text_itn_deterministic,
                 text_itn_unknown_option, text_itn_malformed],
}


# ----------------------------------------------------------------------------- selection

def checks_for(meta: dict) -> list[Check]:
    contract, api = meta["contract"], meta["api"]
    if contract == "openai":
        return [health_ok, models_lists_name, *OPENAI_API_CHECKS[api]]
    if contract == "oip-v2":
        return [health_ok, oip_live_ready, oip_server_metadata, oip_metadata_matches_signature,
                oip_infer_from_signature, oip_wrong_datatype_400]
    if contract == "openai-inspired":
        return [health_ok, models_lists_name, *INSPIRED_API_CHECKS[api]]
    raise ValueError(f"no conformance suite for contract {contract!r}")
