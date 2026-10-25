"""Hosted endpoints: models a provider already serves online, registered with a URL, model name and API key.

The gateway and the verifier both call a provider through this module, so URL building and auth live in
one place. Keys are encrypted with Fernet under a master key kept outside PLATFORM_DIR
(Settings.master_key_file); only the encrypted bytes and the last four characters are stored. A key is
never returned by the API and never written to logs, errors, events or export bundles: provider text that
quotes it is redacted first. Only the gateway's stream of a provider reply passes through unchanged.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken

from .db import DB, now_iso
from .scheduler import SchedulerError

# (contract, api) a hosted endpoint may speak, and how the UI names it.
CONTRACTS: dict[tuple[str, str], str] = {
    ("openai", "audio.transcriptions"): "ASR",
    ("openai", "chat.completions"): "audio LLM or text",
    ("openai", "audio.speech"): "TTS",
    ("openai", "embeddings"): "embeddings",
    ("openai-inspired", "text.itn"): "inverse text normalisation",
    ("oip-v2", "infer"): "VAD, diarisation, speech enhancement",
}
AUTH_TYPES = ("bearer", "header", "none")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
HEADER_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")
TEST_TIMEOUT_S = 30.0


class KeyUnavailable(SchedulerError):
    def __init__(self, message: str):
        super().__init__(503, message)


@dataclass
class Hosted:
    id: str
    contract: str
    api: str
    base_url: str
    model: str  # the provider's model name: what goes in request bodies
    auth_type: str  # bearer | header | none
    header_name: str | None
    key_enc: bytes | None
    key_hint: str | None
    timeout_s: float
    max_concurrency: int
    max_upload_mb: float
    signature: dict[str, Any] | None
    data_ack_at: str
    created_at: str

    @property
    def build_key(self) -> str:
        return build_key(self.contract, self.api, self.base_url, self.model)

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).hostname or self.base_url

    def public(self) -> dict[str, Any]:
        """What records show: no key, only its last four characters."""
        return {"base_url": self.base_url, "model": self.model, "host": self.host, "auth_type": self.auth_type,
                "header_name": self.header_name, "key_hint": self.key_hint, "timeout_s": self.timeout_s,
                "max_concurrency": self.max_concurrency, "max_upload_mb": self.max_upload_mb,
                "resolved_url": resolved_url(self), "data_ack_at": self.data_ack_at, "created_at": self.created_at}


def build_key(contract: str, api: str, base_url: str, model: str) -> str:
    """A hosted model's build: contract, api, base URL and provider model name. Limits and the key are not in it."""
    raw = json.dumps([contract, api, normalise_base(base_url), model], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def normalise_base(url: str) -> str:
    return url.strip().rstrip("/")


def same_host(a: str, b: str) -> bool:
    """Whether two URLs have the same scheme, host and port (the scheme's default port when none is given)."""
    def origin(url: str) -> tuple[str, str | None, int | None] | None:
        u = urlsplit(url.strip())
        try:
            return u.scheme, u.hostname, u.port or {"https": 443, "http": 80}.get(u.scheme)
        except ValueError:  # a port that is not a number
            return None

    first = origin(a)
    return first is not None and first[1] is not None and first == origin(b)


def provider_url(h: Hosted, path: str) -> str:
    """The provider URL for a path under the gateway's /m/{id}/ prefix.

    OpenAI-style base URLs include the version path (https://api.example.com/v1), so a leading `v1/` is
    dropped before joining. For OIP, `v2/models/<catalog id>` becomes `v2/models/<provider model name>`;
    a path that already names the provider model (aicore's OIPClient) is joined unchanged.
    """
    path = path.lstrip("/")
    base = normalise_base(h.base_url)
    if h.contract == "oip-v2":
        prefix = f"v2/models/{h.id}"
        if path == prefix or path.startswith(prefix + "/"):
            path = f"v2/models/{h.model}" + path[len(prefix):]
    elif path == "v1" or path.startswith("v1/"):
        path = path[3:]
    return f"{base}/{path}" if path else base


def resolved_url(h: Hosted) -> str:
    """Where an inference call ends up, for the UI."""
    if h.contract == "oip-v2":
        return provider_url(h, f"v2/models/{h.id}/infer")
    from .verifier.probes import ROUTES

    return provider_url(h, ROUTES.get(h.api, "v1"))


def auth_headers(h: Hosted, key: str | None) -> dict[str, str]:
    if h.auth_type == "none" or not key:
        return {}
    if h.auth_type == "header" and h.header_name:
        return {h.header_name: key}
    return {"Authorization": f"Bearer {key}"}


def strip_auth(headers: list[tuple[str, str]], h: Hosted) -> list[tuple[str, str]]:
    """Drop whatever auth the caller sent: the stored key is the only credential that reaches the provider."""
    drop = {"authorization", "proxy-authorization"} | ({h.header_name.lower()} if h.header_name else set())
    return [(k, v) for k, v in headers if k.lower() not in drop]


def redact(text: str, key: str | None) -> str:
    return text.replace(key, "••••") if key and len(key) >= 4 else text


def redact_all(value: Any, key: str | None) -> Any:
    """value with the key redacted from every string in it: a verification report or record before it is stored."""
    if isinstance(value, str):
        return redact(value, key)
    if isinstance(value, dict):
        return {redact_all(k, key): redact_all(v, key) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_all(v, key) for v in value]
    return value


class Keys:
    """Fernet under the master key file. Missing file: the catalog runs, hosted calls get 503."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fernet: Fernet | None = None
        self._lock = threading.Lock()

    def _get(self) -> Fernet:
        with self._lock:
            if self._fernet is None:
                try:
                    raw = self.path.read_bytes().strip()
                except OSError:
                    raise KeyUnavailable(f"the catalog's master key file {self.path} is missing; `just catalog-up` "
                                         "creates it (or set CATALOG_MASTER_KEY_FILE)") from None
                try:
                    self._fernet = Fernet(raw)
                except ValueError:
                    raise KeyUnavailable(f"the catalog's master key file {self.path} does not hold a Fernet key") from None
            return self._fernet

    @property
    def available(self) -> bool:
        try:
            self._get()
            return True
        except KeyUnavailable:
            return False

    def encrypt(self, key: str) -> bytes:
        return self._get().encrypt(key.encode())

    def decrypt(self, token: bytes) -> str:
        try:
            return self._get().decrypt(token).decode()
        except InvalidToken:
            raise KeyUnavailable("the stored key cannot be decrypted with the current master key; replace the key") from None


class HostedStore:
    """The hosted table, with the set of ids cached for the scheduler and the gateway."""

    def __init__(self, db: DB, keys: Keys):
        self.db = db
        self.keys = keys
        self.transport: httpx.BaseTransport | None = None  # tests: a fake provider for the verifier and tests
        self._ids: set[str] = {r["id"] for r in db.all("SELECT id FROM hosted")}
        self._lock = threading.Lock()

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._ids)

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._ids

    def get(self, model_id: str) -> Hosted | None:
        row = self.db.one("SELECT * FROM hosted WHERE id = ?", (model_id,))
        return _from_row(row) if row else None

    def all(self) -> list[Hosted]:
        return [_from_row(r) for r in self.db.all("SELECT * FROM hosted ORDER BY id")]

    def key(self, h: Hosted) -> str | None:
        return self.keys.decrypt(h.key_enc) if h.key_enc else None

    def insert(self, h: Hosted, key: str | None) -> None:
        enc, hint = self._seal(key)
        self.db.execute(
            "INSERT INTO hosted (id, contract, api, base_url, model, auth_type, header_name, key_enc, key_hint, "
            "timeout_s, max_concurrency, max_upload_mb, signature, data_ack_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (h.id, h.contract, h.api, normalise_base(h.base_url), h.model, h.auth_type, h.header_name, enc, hint,
             h.timeout_s, h.max_concurrency, h.max_upload_mb, json.dumps(h.signature) if h.signature else None,
             h.data_ack_at, h.created_at))
        with self._lock:
            self._ids.add(h.id)

    def update(self, h: Hosted, key: str | None = None) -> None:
        """Writes every field; a new key (when given) replaces the stored one."""
        enc, hint = (h.key_enc, h.key_hint) if key is None else self._seal(key)
        if h.auth_type == "none":
            enc, hint = None, None
        self.db.execute(
            "UPDATE hosted SET contract = ?, api = ?, base_url = ?, model = ?, auth_type = ?, header_name = ?, "
            "key_enc = ?, key_hint = ?, timeout_s = ?, max_concurrency = ?, max_upload_mb = ?, signature = ? WHERE id = ?",
            (h.contract, h.api, normalise_base(h.base_url), h.model, h.auth_type, h.header_name, enc, hint, h.timeout_s,
             h.max_concurrency, h.max_upload_mb, json.dumps(h.signature) if h.signature else None, h.id))

    def delete(self, model_id: str) -> None:
        self.db.execute("DELETE FROM hosted WHERE id = ?", (model_id,))
        with self._lock:
            self._ids.discard(model_id)

    def _seal(self, key: str | None) -> tuple[bytes | None, str | None]:
        if not key:
            return None, None
        return self.keys.encrypt(key), key[-4:]

    def client(self, h: Hosted, key: str | None, timeout: float | None = None) -> httpx.Client:
        """A sync client whose relative paths go to the provider with the stored key (verifier, connection test)."""
        return httpx.Client(base_url="http://hosted", transport=HostedTransport(h, key, self.transport),
                            timeout=timeout or h.timeout_s)


def _from_row(r: dict[str, Any]) -> Hosted:
    return Hosted(r["id"], r["contract"], r["api"], r["base_url"], r["model"], r["auth_type"], r["header_name"],
                  r["key_enc"], r["key_hint"], r["timeout_s"], r["max_concurrency"], r["max_upload_mb"],
                  json.loads(r["signature"]) if r["signature"] else None, r["data_ack_at"], r["created_at"])


class HostedTransport(httpx.BaseTransport):
    """Rewrites `http://hosted/<path>` to the provider URL and adds the stored key."""

    def __init__(self, h: Hosted, key: str | None, inner: httpx.BaseTransport | None = None):
        self.h, self.key = h, key
        self.inner = inner or httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = httpx.URL(provider_url(self.h, request.url.path)).copy_with(query=request.url.query or None)
        headers = strip_auth([(k, v) for k, v in request.headers.items() if k.lower() != "host"], self.h)
        headers += list(auth_headers(self.h, self.key).items())
        out = httpx.Request(request.method, url, headers=headers, content=request.read(),
                            extensions=request.extensions)
        return self.inner.handle_request(out)

    def close(self) -> None:
        self.inner.close()


# ----------------------------------------------------------------------------- connection test

def _silent_wav(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


def _minimal(c: httpx.Client, h: Hosted) -> httpx.Response:
    """One small request for the api, for providers without GET /models."""
    if h.api == "audio.transcriptions":
        return c.post("/v1/audio/transcriptions", data={"model": h.model},
                      files={"file": ("silence.wav", _silent_wav(), "audio/wav")})
    if h.api == "chat.completions":
        return c.post("/v1/chat/completions", json={"model": h.model, "max_tokens": 1,
                                                    "messages": [{"role": "user", "content": "Hi"}]})
    if h.api == "embeddings":
        return c.post("/v1/embeddings", json={"model": h.model, "input": "hello"})
    if h.api == "audio.speech":
        return c.post("/v1/audio/speech", json={"model": h.model, "input": "Hi", "voice": "alloy"})
    return c.post("/v1/text/itn", json={"model": h.model, "input": "twenty one", "options": {}})


def _short(text: str, key: str | None) -> str:
    text = redact(text.strip(), key)
    return text if len(text) < 200 else text[:200] + "..."


def connection_test(store: HostedStore, h: Hosted, key: str | None) -> dict[str, Any]:
    """TLS, key and contract with one or two small requests; stores nothing. OIP also returns the signature."""
    lines: list[dict[str, str]] = []
    signature = None

    def add(name: str, status: str, reason: str) -> None:
        lines.append({"name": name, "status": status, "reason": reason})

    def auth_ok(r: httpx.Response) -> bool:
        if r.status_code in (401, 403):
            add("key", "fail", f"the provider refused the key ({r.status_code}): {_short(r.text, key)}")
            return False
        add("key", "pass", "Key accepted" if h.auth_type != "none" else "No key needed")
        return True

    tls = urlsplit(h.base_url).scheme == "https"
    label = f"{h.contract} / {h.api}"
    try:
        with store.client(h, key, timeout=min(h.timeout_s, TEST_TIMEOUT_S)) as c:
            if h.contract == "oip-v2":
                r = c.get(f"/v2/models/{h.model}/ready")
                add("tls", "pass" if tls else "warn", "Reached the host over TLS" if tls else "Reached the host without TLS (http)")
                if not auth_ok(r):
                    return {"ok": False, "lines": lines, "signature": None}
                meta = c.get(f"/v2/models/{h.model}")
                if r.status_code != 200 or meta.status_code != 200:
                    bad = r if r.status_code != 200 else meta
                    add("contract", "fail", f"{bad.request.url.path} answered {bad.status_code}: {_short(bad.text, key)}")
                else:
                    data = meta.json()
                    signature = {"inputs": [{k: t[k] for k in ("name", "datatype", "shape") if k in t} for t in data.get("inputs", [])],
                                 "outputs": [{k: t[k] for k in ("name", "datatype", "shape") if k in t} for t in data.get("outputs", [])]}
                    add("contract", "pass", f"Speaks {label}: {h.model} is ready")
            else:
                r = c.get("/v1/models")
                add("tls", "pass" if tls else "warn", "Reached the host over TLS" if tls else "Reached the host without TLS (http)")
                if not auth_ok(r):
                    return {"ok": False, "lines": lines, "signature": None}
                listed = None
                if r.status_code == 200:
                    try:
                        listed = [m.get("id") for m in r.json().get("data", [])]
                    except (ValueError, AttributeError, TypeError):
                        listed = None
                if listed is not None and h.model in listed and h.api != "text.itn":
                    add("contract", "pass", f"Speaks {label}: /models lists {h.model}")
                elif listed is not None and h.model not in listed:
                    add("contract", "fail", f"/models lists {len(listed)} model(s) but not {h.model!r}")
                else:
                    m = _minimal(c, h)
                    if m.status_code in (401, 403):
                        add("contract", "fail", f"the provider refused the key ({m.status_code}): {_short(m.text, key)}")
                    elif m.status_code >= 400:
                        add("contract", "fail", f"a minimal {h.api} request answered {m.status_code}: {_short(m.text, key)}")
                    else:
                        add("contract", "pass", f"Speaks {label}: a minimal request was accepted")
    except httpx.HTTPError as exc:
        add("tls", "fail", f"could not reach {h.host}: {type(exc).__name__}: {_short(str(exc), key)}")
    ok = bool(lines) and all(x["status"] != "fail" for x in lines) and any(x["name"] == "contract" for x in lines)
    return {"ok": ok, "lines": lines, "signature": signature, "tested_at": now_iso()}
