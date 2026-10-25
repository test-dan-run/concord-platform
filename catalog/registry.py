"""Registry: model records = platform manifest + registration + scheduler state + capability record.

Hosted endpoints (catalog/hosted.py) are records too: the hosted table instead of a manifest, state
`hosted`, no GPU, image or cold start. Display name and tags live in the same registration table, so tag
search works the same for both kinds.
"""

from __future__ import annotations

import json
from typing import Any

from tools.schema import BASE_PATHS, Manifest, Platform

from . import hosted as hosted_mod
from .db import DB, from_iso, iso, now_iso
from .hosted import CONTRACTS, Hosted, HostedStore
from .scheduler import Scheduler, SchedulerError
from .settings import Settings
from .verifier import Verifier


def _loads(v: str | None) -> Any:
    return None if v is None else json.loads(v)


class Registry:
    def __init__(self, db: DB, scheduler: Scheduler, verifier: Verifier, settings: Settings, hosted: HostedStore):
        self.db = db
        self.scheduler = scheduler
        self.verifier = verifier
        self.settings = settings
        self.hosted = hosted
        self._last_test: dict[str, dict[str, Any]] = {}  # hosted id -> its last connection test

    # ------------------------------------------------------------------------- registration

    def _manifest(self, model_id: str) -> tuple[Platform, Manifest]:
        p = self.scheduler.platform()
        try:
            return p, p.get(model_id)
        except KeyError as exc:
            raise SchedulerError(404, exc.args[0]) from None

    def _parent(self, parent_id: str) -> str:
        return parent_id if parent_id in self.hosted else self._manifest(parent_id)[1].name

    def register(self, model_id: str, display_name: str | None = None, tags: list[str] | None = None,
                 parent_id: str | None = None) -> dict[str, Any]:
        if model_id in self.hosted:  # always registered: only the fields change
            return self.update(model_id, _given(display_name=display_name, tags=tags, parent_id=parent_id))
        _, s = self._manifest(model_id)  # the schema already refused a model over util_limit
        if parent_id:
            parent_id = self._parent(parent_id)
        row = self.db.one("SELECT * FROM registration WHERE model_id = ?", (s.name,))
        if row is None:
            self.db.execute("INSERT INTO registration (model_id, display_name, tags, parent_id, registered_at) "
                            "VALUES (?, ?, ?, ?, ?)", (s.name, display_name, json.dumps(_tags(tags)), parent_id, now_iso()))
        else:
            self._update(s.name, **_given(display_name=display_name, tags=tags, parent_id=parent_id))
        rec = self.record(s.name)
        if rec["capabilities"] is None or rec["capabilities"]["status"] == "stale":
            self.verifier.enqueue(s.name)
            rec = self.record(s.name)
        return rec

    def _update(self, model_id: str, **fields: Any) -> None:
        """Write every field given. None is NULL, and None tags are no tags."""
        sets, args = [], []
        for k, v in fields.items():
            sets.append(f"{k} = ?")
            args.append(json.dumps(_tags(v)) if k == "tags" else v)
        if sets:
            self.db.execute(f"UPDATE registration SET {', '.join(sets)} WHERE model_id = ?", (*args, model_id))

    def name(self, model_id: str) -> str:
        """The model id for an id or alias, hosted or platform; 404 when it is neither."""
        return model_id if model_id in self.hosted else self._manifest(model_id)[1].name

    def update(self, model_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        name = self.name(model_id)
        if not self.db.one("SELECT 1 FROM registration WHERE model_id = ?", (name,)):
            raise SchedulerError(404, f"{name} is not registered")
        allowed = {k: fields.get(k) for k in ("display_name", "tags", "parent_id") if k in fields}
        if allowed.get("parent_id"):
            allowed["parent_id"] = self._parent(allowed["parent_id"])
        self._update(name, **allowed)
        return self.record(name)

    def unregister(self, model_id: str) -> None:
        if model_id in self.hosted:  # a hosted model exists only as a registration: remove it whole
            self.delete_hosted(model_id)
            return
        _, s = self._manifest(model_id)
        self.db.execute("DELETE FROM registration WHERE model_id = ?", (s.name,))

    def verify(self, model_id: str) -> dict[str, Any]:
        name = self.name(model_id)
        self.verifier.enqueue(name)
        return self.record(name)

    # ------------------------------------------------------------------------- hosted endpoints

    def _hosted(self, model_id: str) -> Hosted:
        h = self.hosted.get(model_id)
        if h is None:
            raise SchedulerError(404, f"no hosted endpoint {model_id!r}")
        return h

    def _check_hosted(self, h: Hosted, key: str | None) -> None:
        if (h.contract, h.api) not in CONTRACTS:
            raise SchedulerError(422, f"hosted endpoints speak {', '.join(f'{c} / {a}' for c, a in CONTRACTS)}; "
                                      f"not {h.contract} / {h.api}")
        if not h.base_url.startswith(("https://", "http://")):
            raise SchedulerError(422, "base_url must start with https:// (or http://)")
        if not h.model.strip():
            raise SchedulerError(422, "model (the provider's model name) must not be empty")
        if h.auth_type not in hosted_mod.AUTH_TYPES:
            raise SchedulerError(422, f"auth_type must be one of {', '.join(hosted_mod.AUTH_TYPES)}")
        if h.auth_type == "header" and not hosted_mod.HEADER_RE.match(h.header_name or ""):
            raise SchedulerError(422, "a custom header needs header_name (letters, digits and dashes)")
        if h.auth_type != "none" and key is not None and not key.strip():
            raise SchedulerError(422, "the API key must not be empty")
        if h.timeout_s <= 0 or h.max_concurrency < 1 or h.max_upload_mb <= 0:
            raise SchedulerError(422, "timeout_s, max_concurrency and max_upload_mb must be positive")

    def _spec(self, model_id: str, body: dict[str, Any], old: Hosted | None = None) -> Hosted:
        def get(k: str, default: Any = None) -> Any:
            v = body.get(k)
            return v if v is not None else (getattr(old, k) if old else default)

        auth = get("auth_type", "bearer")
        return Hosted(model_id, get("contract", ""), get("api", ""), hosted_mod.normalise_base(get("base_url", "")),
                      str(get("model", "")).strip(), auth, (get("header_name") or None) if auth == "header" else None,
                      old.key_enc if old else None, old.key_hint if old else None, float(get("timeout_s", 120.0)),
                      int(get("max_concurrency", 4)), float(get("max_upload_mb", 25.0)), old.signature if old else None,
                      old.data_ack_at if old else now_iso(), old.created_at if old else now_iso())

    def test_hosted(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /api/hosted/test: a connection test before saving. With `id` of an existing hosted model and no
        key, the stored key is used (replacing a key tests the new one), but only on the host it was stored for."""
        old = self.hosted.get(body.get("id") or "")
        h = self._spec(body.get("id") or "test", body, old)
        key = body.get("key")
        self._check_hosted(h, key)
        if key is None and old is not None and old.key_enc and h.auth_type != "none":
            if not hosted_mod.same_host(h.base_url, old.base_url):  # the catalog has no API auth: never send it elsewhere
                raise SchedulerError(422, "enter the key to test a different host")
            key = self.hosted.key(old)
        if h.auth_type != "none" and not key:
            raise SchedulerError(422, "the API key is required")
        return hosted_mod.connection_test(self.hosted, h, key)

    def test_model(self, model_id: str) -> dict[str, Any]:
        """POST /api/models/{id}/test: the connection test with the stored key."""
        if model_id not in self.hosted:
            raise SchedulerError(409, f"{model_id} is not a hosted endpoint; platform services are tested by verification")
        h = self._hosted(model_id)
        result = hosted_mod.connection_test(self.hosted, h, self.hosted.key(h))
        self.db.execute("UPDATE hosted SET signature = COALESCE(?, signature) WHERE id = ?",
                        (json.dumps(result["signature"]) if result.get("signature") else None, h.id))
        self._last_test[h.id] = result
        return result

    def create_hosted(self, body: dict[str, Any]) -> dict[str, Any]:
        model_id = str(body.get("id") or "").strip()
        if not hosted_mod.ID_RE.match(model_id):
            raise SchedulerError(422, "the catalog name must be lower case letters, digits, '.', '_' or '-' (at most 64)")
        if not body.get("data_leaves_network"):
            raise SchedulerError(422, "confirm that data sent to this model leaves this network (data_leaves_network)")
        p = self.scheduler.platform()
        taken = {s.name for s in p.services} | {s.alias for s in p.services if s.alias}
        if model_id in taken or model_id in self.hosted:
            raise SchedulerError(409, f"{model_id!r} is already a model id or alias in the catalog")
        h = self._spec(model_id, body)
        key = body.get("key")
        self._check_hosted(h, key)
        if h.auth_type != "none" and not key:
            raise SchedulerError(422, "the API key is required")
        parent = self._parent(body["parent_id"]) if body.get("parent_id") else None  # 404 before anything is stored
        if not self.hosted.keys.available:
            self.hosted.keys.encrypt("")  # raises KeyUnavailable: 503, naming the master key file
        if h.contract == "oip-v2":  # usable_as() needs the signature: take it from the provider's metadata
            test = hosted_mod.connection_test(self.hosted, h, key)
            if not test.get("signature"):
                raise SchedulerError(422, _no_metadata(h, test))
            h.signature = test["signature"]
        self.hosted.insert(h, key if h.auth_type != "none" else None)
        tags = body.get("tags")
        self.db.execute("INSERT OR REPLACE INTO registration (model_id, display_name, tags, parent_id, registered_at) "
                        "VALUES (?, ?, ?, ?, ?)", (model_id, body.get("display_name") or None, json.dumps(_tags(tags)),
                                                   parent, now_iso()))
        self.verifier.enqueue(model_id)
        return self.record(model_id)

    def update_hosted(self, model_id: str, body: dict[str, Any]) -> dict[str, Any]:
        old = self._hosted(model_id)
        fields = {k: body[k] for k in ("display_name", "tags", "parent_id") if k in body}
        if fields.get("parent_id"):  # an unknown parent is a 404 before anything is saved
            fields["parent_id"] = self._parent(fields["parent_id"])
        h = self._spec(model_id, body, old)
        key = body.get("key")
        self._check_hosted(h, key)
        if key:  # a new key is tested before it replaces the old one
            test = hosted_mod.connection_test(self.hosted, h, key)
            if not test["ok"]:
                why = "; ".join(x["reason"] for x in test["lines"] if x["status"] == "fail")
                raise SchedulerError(422, f"the new key did not pass the connection test: {why}")
        elif h.auth_type != "none" and not old.key_enc:
            raise SchedulerError(422, "the API key is required")
        if h.contract != old.contract:  # an OIP signature does not carry over to another contract
            h.signature = None
        if h.contract == "oip-v2" and (h.build_key != old.build_key or not h.signature):
            test = hosted_mod.connection_test(self.hosted, h, key or self.hosted.key(old))
            h.signature = test.get("signature") or h.signature
            if not h.signature:  # like create: usable_as() needs the signature
                raise SchedulerError(422, _no_metadata(h, test))
        self.hosted.update(h, key)
        if h.build_key != old.build_key:  # a new build: verify it; limits and keys keep the build
            self.verifier.enqueue(model_id)
        return self.update(model_id, fields)  # the registration fields given, as PATCH /api/models/{id} writes them

    def delete_hosted(self, model_id: str) -> None:
        self._hosted(model_id)
        self.hosted.delete(model_id)
        self.db.execute("DELETE FROM registration WHERE model_id = ?", (model_id,))
        self._last_test.pop(model_id, None)

    # ------------------------------------------------------------------------- records

    def capabilities(self, model_id: str, build_key: str | None) -> dict[str, Any] | None:
        queued = self.verifier.status(model_id)
        row = self.db.one("SELECT * FROM capability WHERE build_key = ?", (build_key,)) if build_key else None
        stale = False
        if row is None:
            row = self.db.one("SELECT * FROM capability WHERE model_id = ? ORDER BY updated_at DESC LIMIT 1", (model_id,))
            stale = row is not None
        if row is None:
            return {"build_key": build_key, "status": queued, "usable_as": None, "tasks": [], "timing_level": None,
                    "output_detail": [], "protocol": None, "rtf": None, "verified_at": None, "report": None} if queued else None
        record = _loads(row["record"]) or {}
        return {
            "build_key": row["build_key"],
            "status": queued or ("stale" if stale else row["status"]),
            "usable_as": record.get("usable_as"),
            "tasks": record.get("tasks") or [],
            "timing_level": record.get("timing_level"),
            "output_detail": record.get("output_detail") or [],
            "protocol": record.get("protocol"),
            "rtf": row["rtf"],
            "verified_at": row["verified_at"],
            "report": _loads(row["report"]),
        }

    def record(self, model_id: str, p: Platform | None = None, usage: dict[str, dict] | None = None,
               regs: dict[str, dict] | None = None) -> dict[str, Any]:
        if model_id in self.hosted:
            h = self.hosted.get(model_id)
            if h is not None:
                return self.hosted_record(h, usage, regs)
        p = p or self.scheduler.platform()
        try:
            s = p.get(model_id)
        except KeyError as exc:
            raise SchedulerError(404, exc.args[0]) from None
        reg = (regs or {}).get(s.name) if regs is not None else self.db.one(
            "SELECT * FROM registration WHERE model_id = ?", (s.name,))
        use = (usage or {}).get(s.name) if usage is not None else self.db.one(
            "SELECT * FROM usage WHERE model_id = ?", (s.name,))
        view = self.scheduler.model_view(s.name)
        build = view["build"]
        bk = build.build_key if build else None
        cold, measured = self.scheduler.cold_start(p, s.name)
        frac = s.vram_fraction(p.config.gpu_total_gb)
        last_used = max(filter(None, [view["last_used"], (use or {}).get("last_used")]), default=None)
        return {
            "id": s.name,
            "aliases": [s.alias] if s.alias else [],
            "registered": reg is not None,
            "display_name": (reg or {}).get("display_name") or s.name,
            "tags": json.loads(reg["tags"]) if reg else [],
            "parent_id": (reg or {}).get("parent_id"),
            "registered_at": (reg or {}).get("registered_at"),
            "description": s.description,
            "tasks": list(s.tasks),
            "kind": s.kind,
            "engine": s.engine,
            "contract": s.contract,
            "api": s.api,
            "signature": s.signature.model_dump(exclude_none=True) if s.signature else None,
            "gpu": None if s.gpu is None else {"fraction": round(frac, 4), "gb": round(frac * p.config.gpu_total_gb, 2)},
            "state": view["state"],
            "state_detail": view["state_detail"],
            "leased_by": view["leased_by"],
            "in_flight": view["in_flight"],
            "build_key": bk,
            "build_pending": build.pending if build else "not yet inspected",
            "image": p.image(s),
            "capabilities": self.capabilities(s.name, bk),
            "cold_start_s": round(cold, 1),
            "cold_start_measured": measured,
            "endpoint": {"gateway_url": f"{self.settings.public_url}/m/{s.name}", "base_path": BASE_PATHS[s.contract]},
            "last_used": iso(last_used),
            "calls": (use or {}).get("calls", 0),
            "served_name": s.name,
            "hosted": None,
        }

    def hosted_record(self, h: Hosted, usage: dict[str, dict] | None = None,
                      regs: dict[str, dict] | None = None) -> dict[str, Any]:
        reg = (regs or {}).get(h.id) if regs is not None else self.db.one(
            "SELECT * FROM registration WHERE model_id = ?", (h.id,))
        use = (usage or {}).get(h.id) if usage is not None else self.db.one(
            "SELECT * FROM usage WHERE model_id = ?", (h.id,))
        view = self.scheduler.model_view(h.id)
        last_used = max(filter(None, [view["last_used"], (use or {}).get("last_used")]), default=None)
        caps = self.capabilities(h.id, h.build_key)
        public = h.public()
        verified = (caps or {}).get("verified_at")
        public["next_check_at"] = iso(from_iso(verified) + self.settings.hosted_reverify_s) if verified else None
        public["last_test"] = self._last_test.get(h.id)
        detail = None if self.hosted.keys.available or h.auth_type == "none" else \
            f"key unavailable: master key file {self.hosted.keys.path} is missing"
        return {
            "id": h.id, "aliases": [], "registered": True,
            "display_name": (reg or {}).get("display_name") or h.id,
            "tags": json.loads(reg["tags"]) if reg else [],
            "parent_id": (reg or {}).get("parent_id"),
            "registered_at": (reg or {}).get("registered_at") or h.created_at,
            "description": f"{CONTRACTS.get((h.contract, h.api), h.api).capitalize()} served by {h.host}",
            "tasks": [], "kind": "hosted", "engine": "hosted", "contract": h.contract, "api": h.api,
            "signature": h.signature, "gpu": None, "state": "hosted", "state_detail": detail,
            "leased_by": view["leased_by"], "in_flight": view["in_flight"],
            "build_key": h.build_key, "build_pending": None, "image": None,
            "capabilities": caps, "cold_start_s": 0.0, "cold_start_measured": True,
            "endpoint": {"gateway_url": f"{self.settings.public_url}/m/{h.id}", "base_path": BASE_PATHS[h.contract]},
            "last_used": iso(last_used), "calls": (use or {}).get("calls", 0),
            "errors": (use or {}).get("errors", 0),
            "served_name": h.model,
            "hosted": public,
        }

    def records(self, task: list[str] | None = None, tag: list[str] | None = None, state: str | None = None,
                include_all: bool = False) -> list[dict[str, Any]]:
        p = self.scheduler.platform()
        regs = {r["model_id"]: r for r in self.db.all("SELECT * FROM registration")}
        usage = {r["model_id"]: r for r in self.db.all("SELECT * FROM usage")}
        out = []
        recs = [self.record(s.name, p, usage, regs) for s in p.services if include_all or s.name in regs]
        recs += [self.hosted_record(h, usage, regs) for h in self.hosted.all()]
        for rec in sorted(recs, key=lambda r: r["id"]):
            seen = set(rec["tasks"]) | set((rec["capabilities"] or {}).get("tasks") or [])
            if (rec["capabilities"] or {}).get("usable_as") == "chat-audio" and "asr" in seen:
                seen.add("chat-audio")
            if task and not seen & set(task):
                continue
            if tag and not set(tag) <= set(rec["tags"]):
                continue
            if state and rec["state"] != state:
                continue
            out.append(rec)
        return out


def _no_metadata(h: Hosted, test: dict[str, Any]) -> str:
    why = "; ".join(x["reason"] for x in test["lines"] if x["status"] == "fail") or "no signature"
    return f"cannot read {h.model}'s OIP metadata from the provider: {why}"


def _given(**fields: Any) -> dict[str, Any]:
    """The fields that are not None: registering again leaves a field it was not given as it was."""
    return {k: v for k, v in fields.items() if v is not None}


def _tags(tags: list[str] | None) -> list[str]:
    seen: list[str] = []
    for t in tags or []:
        t = str(t).strip()
        if t and t not in seen:
            seen.append(t)
    return seen
