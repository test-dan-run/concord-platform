"""Batch profiles: how a model's peak GPU memory grows with batch size and input length, and the largest
batch that fits a given amount of memory.

A custom audio image is measured (`just batch-profile NAME`, POST /api/models/{id}/batch-profile): the
profiler leases the model (holder `catalog:batch`), lifts the server's batch limits, resets its peak, and
sends batches of N = 1, 2, 4, ... items at three lengths (short, typical and the longest the server takes),
up to the first out of memory point. It fits

    peak ≈ base + a·N·L + b·N·L²        (GiB; L in seconds of audio)

by least squares (a, b >= 0; the L² term is attention) and stores it with the build key, since a rebuilt
image can use memory differently. The recommendation for V GiB is the largest N with peak <= 0.9·V. The
next start of that build gets the limits for the memory it was given, as AISVC_MAX_BATCH (N at the short
length) and AISVC_MAX_BATCH_LEN (the padded length, items x longest, in milliseconds of audio).

Peak memory is read two ways: nvidia-smi sampled during each request (device used minus what was used
before the model started; correct because one GPU model runs at a time), and the image's own torch counters
from /platform/memory. The torch reading is kept when there is one, since the device also holds what torch
cached for earlier batches. CTranslate2 and ONNX images have only the first, and it counts only above the
highest one so far, because those runtimes keep the memory a bigger batch took.

vLLM services batch by themselves, so nothing is measured: their startup log gives the KV cache size, and
the recommendation is a --max-num-seqs value, KV tokens / tokens per sequence.

A person can override either limit for a model (PUT /api/models/{id}/batch-override). The override is kept
per model, not per build, so it outlasts a rebuild, and a limit it leaves out still comes from the profile.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

from tools import render
from tools.schema import CUSTOM_ENGINE, Manifest, Platform

from .db import DB, dumps, now_iso
from .scheduler import Scheduler, SchedulerError

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger("catalog.batching")
HOLDER = "catalog:batch"
HEADROOM = 0.9  # a recommendation keeps the predicted peak under this share of the memory
MAX_N = 256  # largest batch ever recommended (the fit is not trusted far past the measured points)
# Most audio in one request: OIP sends samples as JSON numbers, so 64 x 180 s (11,520 s) took the aligner server's
# host memory with it on 2026-10-25, while 32 x 180 s worked. Recommendations and the sweep stay under this.
MAX_REQUEST_S = 6000.0
# Log lines read for a vLLM profile: the whole container log. The KV cache lines come once, at startup, and vLLM
# then logs a line per request, so a short tail misses them on a model that has served a while.
LOG_TAIL = 1_000_000
SWEEP_MAX_N = 128
# Most concurrent requests the sweep sends an OpenAI transcription server: aisvc's route is a sync function that
# Starlette runs in a pool of 40 threads, so more requests at once wait there and do not make a bigger batch.
OPENAI_ASR_MAX_N = 32
DEFAULT_LENGTHS_S = (2.0, 15.0, 60.0)
SLOW_S = 120.0  # a batch that takes longer ends its length's sweep
SAMPLE_S = 0.2  # nvidia-smi sampling interval during a request
CONTEXT_GIB = 0.6  # CUDA context and libraries, when only torch counters are known
LIFT = {"max_items": 4096, "max_len": None}  # server limits while measuring: every test batch runs whole
SR = 16000
BUDGETS_GB = (16.0, 24.0, 32.0)
TABLE_LENGTHS_S = (5.0, 20.0, 60.0)
TABLE_TOKENS = (1024, 4096, 8192)


# ----------------------------------------------------------------------------- the profile


@dataclass
class Point:
    n: int
    length_s: float
    peak_gib: float
    seconds: float


@dataclass
class Profile:
    kind: str  # "audio" (measured) | "kv" (vLLM startup log)
    base: float = 0.0
    a: float = 0.0
    b: float = 0.0
    max_item_s: float | None = None
    points: list[Point] = field(default_factory=list)
    oom: list[dict[str, Any]] = field(default_factory=list)  # where each length's sweep stopped, and why
    method: str = ""  # how peak memory was read
    kv_tokens: int | None = None  # kv: the KV cache at the given memory
    kv_gib: float | None = None
    fixed_gib: float | None = None  # kv: weights, activations and CUDA graphs (given memory - KV cache)
    max_model_len: int | None = None
    max_concurrency: float | None = None  # vLLM's own figure at max_model_len
    given_gib: float | None = None  # the memory the measured start was given

    def peak(self, n: int, length: float) -> float:
        """Predicted peak GiB: audio N items of `length` s; kv N sequences of `length` tokens."""
        if self.kind == "kv":
            assert self.fixed_gib is not None and self.kv_gib and self.kv_tokens
            return self.fixed_gib + n * length * self.kv_gib / self.kv_tokens
        return self.base + self.a * n * length + self.b * n * length * length

    def recommend(self, vram_gib: float, length: float) -> int:
        """Largest batch whose peak fits: audio within HEADROOM x V; kv within the KV cache V leaves (vLLM sizes
        its cache itself, from the memory it is given). 0 = not even one item."""
        if self.kind == "kv":
            assert self.fixed_gib is not None and self.kv_gib and self.kv_tokens
            tokens = (vram_gib - self.fixed_gib) * self.kv_tokens / self.kv_gib
            return max(0, min(MAX_N, int(tokens // max(1, length))))
        per = self.a * length + self.b * length * length
        room = HEADROOM * vram_gib - self.base
        if room < per or room <= 0:
            return 0
        cap = min(MAX_N, max(1, int(MAX_REQUEST_S // max(length, 1e-9))))
        return cap if per <= 0 else min(cap, int(room / per))

    def need(self, n: int, length: float) -> float:
        """GiB of memory a batch needs: the peak with the headroom (audio) or the vLLM budget (kv)."""
        return self.peak(n, length) / (1.0 if self.kind == "kv" else HEADROOM)

    def start_env(self, vram_gib: float) -> dict[str, str]:
        """AISVC_MAX_BATCH and AISVC_MAX_BATCH_LEN for a start given `vram_gib` (audio only)."""
        if self.kind != "audio":
            return {}
        short = min((p.length_s for p in self.points), default=DEFAULT_LENGTHS_S[0])
        longest = self.max_item_s or max((p.length_s for p in self.points), default=DEFAULT_LENGTHS_S[-1])
        room = HEADROOM * vram_gib - self.base
        n = self.recommend(vram_gib, short)
        if n < 1 or room <= 0:
            return {}
        padded_s = room / (self.a + self.b * longest) if self.a + self.b * longest > 0 else MAX_N * longest
        padded_s = min(padded_s, MAX_REQUEST_S)
        return {"AISVC_MAX_BATCH": str(n), "AISVC_MAX_BATCH_LEN": str(max(int(longest * 1000), int(padded_s * 1000)))}

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Profile:
        return cls(**{**d, "points": [Point(**p) for p in d.get("points") or []]})


def fit(points: list[Point]) -> tuple[float, float, float]:
    """(base, a, b) of peak ≈ base + a·N·L + b·N·L², least squares with a, b >= 0."""
    if not points:
        raise ValueError("no measured points")
    x1 = np.array([p.n * p.length_s for p in points], float)
    x2 = np.array([p.n * p.length_s ** 2 for p in points], float)
    y = np.array([p.peak_gib for p in points], float)
    best: tuple[float, tuple[float, float, float]] | None = None
    for cols in ((x1, x2), (x1,), (x2,), ()):  # the full model first, then with a coefficient held at 0
        A = np.column_stack([np.ones_like(y), *cols])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if any(c < 0 for c in coef[1:]):
            continue
        full = [coef[0], 0.0, 0.0]
        if len(cols) == 2:
            full[1], full[2] = coef[1], coef[2]
        elif len(cols) == 1:
            full[1 if cols[0] is x1 else 2] = coef[1]
        err = float(np.sum((A @ coef - y) ** 2))
        if best is None or err < best[0] - 1e-12:
            best = (err, (float(full[0]), float(full[1]), float(full[2])))
        if len(cols) == 2:
            break  # the unconstrained fit is already non-negative
    assert best is not None  # the constant-only fit never has a negative coefficient
    return best[1]


def table(profile: Profile, full_gib: float | None) -> dict[str, Any]:
    """Recommended batch for each table length against 16, 24 and 32 GB and the full budget."""
    budgets = [*BUDGETS_GB, *([round(full_gib, 2)] if full_gib else [])]
    lengths = TABLE_TOKENS if profile.kind == "kv" else tuple(
        x for x in TABLE_LENGTHS_S if profile.max_item_s is None or x <= profile.max_item_s) or (profile.max_item_s,)
    return {"unit": "tokens" if profile.kind == "kv" else "s", "budgets_gb": budgets, "full_gb": full_gib,
            "rows": [{"length": L, "batch": [profile.recommend(v, L) for v in budgets]} for L in lengths]}


# ----------------------------------------------------------------------------- vLLM


_KV_GIB = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_CONCURRENCY = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")


def kv_profile(log_text: str, given_gib: float) -> Profile:
    """The KV cache lines vLLM logs at startup -> a kv profile. Raises ValueError when they are missing."""
    gib, tokens = _KV_GIB.findall(log_text), _KV_TOKENS.findall(log_text)
    if not gib or not tokens:
        raise ValueError("the vLLM log has no 'Available KV cache memory' and 'GPU KV cache size' lines")
    kv_gib, kv_tokens = float(gib[-1]), int(tokens[-1].replace(",", ""))
    conc = _CONCURRENCY.findall(log_text)
    return Profile(kind="kv", kv_gib=kv_gib, kv_tokens=kv_tokens, fixed_gib=round(given_gib - kv_gib, 3),
                   given_gib=round(given_gib, 3), method="vLLM startup log",
                   max_model_len=int(conc[-1][0].replace(",", "")) if conc else None,
                   max_concurrency=float(conc[-1][1]) if conc else None)


# ----------------------------------------------------------------------------- measuring


class Oom(Exception):
    pass


def mode(meta: dict[str, Any]) -> str | None:
    """How a batch is sent: "oip-batch" (one request, AUDIO [-1,-1] + LENGTHS), "oip-single" (concurrent
    requests, AUDIO [-1]), "openai-asr" (concurrent transcriptions), or None (not measurable)."""
    if meta.get("contract") == "openai" and meta.get("api") == "audio.transcriptions":
        return "openai-asr"
    if meta.get("contract") != "oip-v2":
        return None
    inputs = {t["name"]: t for t in (meta.get("signature") or {}).get("inputs", [])}
    audio = inputs.get("AUDIO")
    if not audio or audio.get("datatype") != "FP32":
        return None
    if len(audio.get("shape") or []) == 2 and "LENGTHS" in inputs:
        return "oip-batch"
    return "oip-single" if len(audio.get("shape") or []) == 1 else None


def speech(seconds: float) -> np.ndarray:
    """The verification clip, repeated to `seconds` at 16 kHz."""
    from .verifier import probes

    base, _, _ = probes.clip()
    n = max(1, int(seconds * SR))
    return np.resize(base.astype(np.float32), n)


def _wav(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _check(exc: Exception) -> None:
    from aisvc.batch import is_oom

    if is_oom(exc) or "out of memory" in str(exc).lower():
        raise Oom(str(exc)[:300]) from exc
    raise exc


def sender(meta: dict[str, Any], url: str, http: httpx.Client) -> Callable[[int, float], None]:
    """send(n, seconds): one batch of n clips; raises Oom on an out of memory error."""
    how = mode(meta)
    if how is None:
        raise ValueError(f"{meta.get('contract')}/{meta.get('api')} has no batch measurement")
    if how == "openai-asr":
        def one(clip: bytes) -> None:
            r = http.post(f"{url}/v1/audio/transcriptions", files={"file": ("clip.wav", clip, "audio/wav")},
                          data={"model": meta["name"], "response_format": "json"})
            if r.status_code >= 400:
                _check(RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}"))

        def send(n: int, seconds: float) -> None:
            clip = _wav(speech(seconds))
            with ThreadPoolExecutor(n) as ex:
                for f in [ex.submit(one, clip) for _ in range(n)]:
                    f.result()
        return send

    from aicore.oip import OIPClient

    client = OIPClient(url, meta["name"], meta.get("signature"), timeout=SLOW_S * 5)  # its own client, on url
    extra = {t["name"]: np.asarray(t["example"]) for t in meta["signature"]["inputs"]
             if t["name"] not in ("AUDIO", "LENGTHS", "SAMPLE_RATE") and "example" in t}

    def infer(inputs: dict[str, np.ndarray]) -> None:
        try:
            client.infer({**inputs, "SAMPLE_RATE": np.array([SR], np.int32), **extra})
        except Exception as exc:  # noqa: BLE001
            _check(exc)

    if how == "oip-batch":
        def send(n: int, seconds: float) -> None:
            a = speech(seconds)
            infer({"AUDIO": np.tile(a, (n, 1)), "LENGTHS": np.full(n, len(a), np.int32)})
        return send

    def send(n: int, seconds: float) -> None:
        a = speech(seconds)
        with ThreadPoolExecutor(n) as ex:
            for f in [ex.submit(infer, {"AUDIO": a}) for _ in range(n)]:
                f.result()
    return send


class Memory:
    """Peak memory of one request: nvidia-smi samples (device used - `baseline`) and the image's torch counters."""

    def __init__(self, url: str, http: httpx.Client, gpu: Callable[[], tuple[float, float] | None],
                 baseline_gib: float | None):
        self.url, self.http, self.gpu, self.baseline = url, http, gpu, baseline_gib
        self.high: float | None = None  # the most the device has had in use: allocators keep what they took

    def _used(self) -> float | None:
        g = self.gpu()
        return None if g is None else g[0] - g[1]

    def _stats(self, reset: bool = False) -> dict[str, Any]:
        try:
            r = self.http.post(f"{self.url}/platform/memory/reset") if reset else self.http.get(f"{self.url}/platform/memory")
            return r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}

    @staticmethod
    def backoffs(stats: dict[str, Any]) -> int:
        return sum(int(b.get("backoffs") or 0) for b in (stats.get("batch") or {}).values())

    def measure(self, fn: Callable[[], None]) -> tuple[float, str] | None:
        """Run fn; (peak GiB, how it was read), or None when it cannot be told (a device reading under an earlier
        high, with no torch counters). Raises Oom on a lowered server limit, like an error would.

        The torch reading is exact: the reset empties the image's CUDA cache first. A device reading counts only
        above the highest one so far, since CTranslate2 and ONNX Runtime keep the memory a bigger batch took."""
        before = self._stats(reset=True)
        idle = self._used()
        samples: list[float] = []
        done = threading.Event()

        def sample() -> None:
            while not done.wait(SAMPLE_S):
                u = self._used()
                if u is not None:
                    samples.append(u)

        t = threading.Thread(target=sample, daemon=True)
        t.start()
        try:
            fn()
        finally:
            done.set()
            t.join()
        last = self._used()
        if last is not None:
            samples.append(last)
        after = self._stats()
        if self.backoffs(after) > self.backoffs(before):
            raise Oom("the server ran out of memory and split the batch")
        reads: list[tuple[float, str]] = []
        if self.baseline is not None and samples:
            top = max(samples)
            if self.high is None or top > self.high + 0.05:
                reads.append((top - self.baseline, "nvidia-smi"))
            self.high = max(top, self.high or top)
        if after.get("cuda") and before.get("cuda"):
            extra = after["peak_reserved_gib"] - before.get("reserved_gib", 0.0)
            if self.baseline is not None and idle is not None:
                reads.append((idle - self.baseline + extra, "torch"))
            else:
                reads.append((after["peak_reserved_gib"] + CONTEXT_GIB, "torch (context estimated)"))
        if not reads:
            if self.baseline is None:
                raise ValueError("no way to read the model's memory: the image has no torch counters and the "
                                 "memory used before it started is unknown (restart it from the catalog)")
            return None
        torch_reads = [r for r in reads if r[1].startswith("torch")]
        peak, how = max(torch_reads or reads)  # torch's own count, when there is one, is not inflated by caches
        return round(peak, 3), how


def lengths_for(s: Manifest) -> tuple[list[float], float | None]:
    hints = s.batch
    longest = hints.max_item_s if hints else None
    lengths = list(hints.lengths_s) if hints and hints.lengths_s else [
        x for x in DEFAULT_LENGTHS_S if longest is None or x < longest] + ([longest] if longest else [])
    return sorted(set(lengths)), longest


def sweep(send: Callable[[int, float], None], memory: Memory, lengths: list[float], *, budget_gib: float | None,
          max_n: int = SWEEP_MAX_N, stop: threading.Event | None = None,
          log_fn: Callable[[str], None] = log.info,
          relift: Callable[[], None] | None = None) -> tuple[list[Point], list[dict[str, Any]], str]:
    """Batches of 1, 2, 4, ... at each length until the first out of memory point, a slow batch, the budget or
    max_n. Returns (points, where each length stopped, how memory was read). `relift` runs before each length
    after the first: a split at one length lowers the server's limit until it is lifted again."""
    points: list[Point] = []
    stops: list[dict[str, Any]] = []
    how = ""
    for i, length in enumerate(lengths):
        if i and relift is not None:
            relift()
        n, why = 1, "max batch"
        here: list[Point] = []
        while n <= max_n:
            if stop is not None and stop.is_set():
                return points, stops, how
            if n > 1 and n * length > MAX_REQUEST_S:
                why = f"a batch of {n} would be {n * length:.0f} s of audio in one request, over {MAX_REQUEST_S:.0f} s"
                break
            if budget_gib is not None and len(here) >= 2:
                # Doubling N about doubles what the batch adds to the first point. Past the budget, WSL2 does not
                # fail: Windows moves GPU memory to system RAM and the batch takes minutes, so stop before it.
                guess = here[0].peak_gib + (here[-1].peak_gib - here[0].peak_gib) * n / here[-1].n
                if guess > 0.95 * budget_gib:
                    why = f"a batch of {n} would need about {guess:.1f} GiB, near the {budget_gib:.1f} GiB budget"
                    break
            t0 = time.perf_counter()
            try:
                read = memory.measure(lambda n=n, length=length: send(n, length))
            except Oom as exc:
                why = f"out of memory at {n}: {exc}"
                break
            secs = time.perf_counter() - t0
            if read is not None:  # None: no reading for this batch; a bigger one will tell
                peak, how = read
                points.append(Point(n=n, length_s=length, peak_gib=peak, seconds=round(secs, 3)))
                here.append(points[-1])
                log_fn(f"batch {n} x {length:g} s: peak {peak:.2f} GiB in {secs:.1f} s")
                if budget_gib is not None and peak > 0.95 * budget_gib:
                    why = f"peak {peak:.1f} GiB near the {budget_gib:.1f} GiB budget at {n}"
                    break
            if secs > SLOW_S:  # with or without a reading
                why = f"{secs:.0f} s for a batch of {n}"
                break
            n *= 2
        stops.append({"length_s": length, "stopped": why})
    return points, stops, how


# ----------------------------------------------------------------------------- the profiler


class BatchProfiler:
    """Measures one model at a time on a background thread and stores profiles per build key."""

    def __init__(self, db: DB, scheduler: Scheduler, runtime: Runtime):
        self.db, self.scheduler, self.runtime = db, scheduler, runtime
        self.pending: dict[str, str] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="batch-profiler").start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    def enqueue(self, model: str) -> str:
        with self._lock:
            if model not in self.pending:
                self.pending[model] = "pending"
                self._order.append(model)
            state = self.pending[model]
        self._wake.set()
        return state

    def status(self, model: str) -> str | None:
        with self._lock:
            return self.pending.get(model)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                model = self._order.popleft() if self._order else None
                if model:
                    self.pending[model] = "running"
            if model is None:
                self._wake.wait(10.0)
                self._wake.clear()
                continue
            try:
                self.run(model)
            except Exception as exc:  # noqa: BLE001
                log.exception("batch profile of %s failed", model)
                self._fail(model, None, f"crashed: {exc}")
            finally:
                with self._lock:
                    self.pending.pop(model, None)

    # -- storage

    def stored(self, build_key: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM batch_profile WHERE build_key = ?", (build_key,))
        if row is None:
            return None
        return {**row, "profile": json.loads(row["profile"]) if row["profile"] else None}

    def profile(self, build_key: str | None) -> Profile | None:
        row = self.stored(build_key) if build_key else None
        return Profile.from_json(row["profile"]) if row and row["status"] == "measured" and row["profile"] else None

    def _save(self, model: str, build_key: str, status: str, profile: Profile | None, error: str | None) -> None:
        self.db.execute("INSERT INTO batch_profile (build_key, model_id, status, profile, error, measured_at) "
                        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(build_key) DO UPDATE SET status = excluded.status, "
                        "profile = excluded.profile, error = excluded.error, measured_at = excluded.measured_at",
                        (build_key, model, status, dumps(profile.to_json()) if profile else None, error, now_iso()))

    def _fail(self, model: str, build_key: str | None, error: str) -> None:
        if build_key is None:
            build_key = self.scheduler.build_key(model)
        if build_key:
            self._save(model, build_key, "failed", None, error)
        log.warning("batch profile of %s: %s", model, error)

    def override(self, model: str) -> dict[str, Any] | None:
        return self.db.one("SELECT max_batch, max_len_s, set_at FROM batch_override WHERE model_id = ?", (model,))

    def set_override(self, model: str, max_batch: int | None, max_len_s: float | None) -> dict[str, Any]:
        """Set the limits by hand; one left out still comes from the profile. Raises ValueError when out of range."""
        if max_batch is None and max_len_s is None:
            raise ValueError("give max_batch, max_len_s or both (DELETE the override to use the measured limits)")
        if max_batch is not None and not 1 <= max_batch <= LIFT["max_items"]:
            raise ValueError(f"max_batch must be from 1 to {LIFT['max_items']}")
        if max_len_s is not None and not 0.001 <= max_len_s <= MAX_REQUEST_S:  # the image takes whole ms, at least 1
            raise ValueError(f"max_len_s must be from 0.001 to {MAX_REQUEST_S:g} s")
        self.db.execute("INSERT INTO batch_override (model_id, max_batch, max_len_s, set_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(model_id) DO UPDATE SET max_batch = excluded.max_batch, "
                        "max_len_s = excluded.max_len_s, set_at = excluded.set_at",
                        (model, max_batch, max_len_s, now_iso()))
        return self.override(model) or {}

    def clear_override(self, model: str) -> None:
        self.db.execute("DELETE FROM batch_override WHERE model_id = ?", (model,))

    def limits(self, model: str, build_key: str | None, vram_gib: float) -> dict[str, Any]:
        """The limits a start of `model` given `vram_gib` gets, and where each comes from: "override" (set by
        hand), "measured" (the build's profile) or "default" (the image's own, shown as None)."""
        p = self.profile(build_key)
        env = p.start_env(vram_gib) if p else {}
        o = self.override(model) or {}
        measured = {"max_batch": int(env["AISVC_MAX_BATCH"]) if "AISVC_MAX_BATCH" in env else None,
                    "max_len_s": int(env["AISVC_MAX_BATCH_LEN"]) / 1000 if "AISVC_MAX_BATCH_LEN" in env else None}
        out: dict[str, Any] = {"vram_gb": round(vram_gib, 2), "source": {}, "measured": measured}
        for key in ("max_batch", "max_len_s"):
            if o.get(key) is not None:
                out[key], out["source"][key] = o[key], "override"
            else:
                out[key], out["source"][key] = measured[key], "measured" if measured[key] is not None else "default"
        return out

    def start_env(self, name: str, build_key: str | None, vram_gib: float) -> dict[str, str]:
        """Batch limits for a start of `name` given `vram_gib` (the scheduler calls this; {} = image defaults)."""
        lim = self.limits(name, build_key, vram_gib)
        env = {}
        if lim["max_batch"] is not None:
            env["AISVC_MAX_BATCH"] = str(int(lim["max_batch"]))
        if lim["max_len_s"] is not None:
            env["AISVC_MAX_BATCH_LEN"] = str(round(lim["max_len_s"] * 1000))
        return env

    def in_use(self, p: Platform, s: Manifest) -> dict[str, Any] | None:
        """The limits the running service started with (None: started outside the catalog) and the ones it
        reports now, per batcher. None when it is not running."""
        sched = self.scheduler
        run = sched.run_view(s.name)
        if run is None or not run.running:
            return None
        given, batch, measured = run.given, run.batch, run.gpu_used_before is not None
        total = sched.gpu[0] if sched.gpu else p.config.gpu_total_gb
        started = None if batch is None else {
            "max_batch": int(batch["AISVC_MAX_BATCH"]) if "AISVC_MAX_BATCH" in batch else None,
            "max_len_s": int(batch["AISVC_MAX_BATCH_LEN"]) / 1000 if "AISVC_MAX_BATCH_LEN" in batch else None}
        # The memory the start was given, as tools/vram_check.start_memory works it out: the share of the device
        # measured then (gpu_used_before is set), floored to 0.01 GiB, else util_limit x gpu_total_gb.
        gib = None if not given else (math.floor(given * total * 100) / 100 if measured
                                      else round(given * p.config.gpu_total_gb, 2))
        out: dict[str, Any] = {"vram_gb": gib, "started_with": started, "batchers": []}
        if s.engine != CUSTOM_ENGINE:
            return out
        try:
            r = httpx.get(f"{sched.upstream_url(s.name)}/platform/memory", timeout=2.0)
            stats = (r.json().get("batch") or {}) if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError, AttributeError):
            stats = {}
        for name, b in stats.items():
            items, length = b.get("configured_max_items"), b.get("configured_max_len")
            if items is None or (items >= LIFT["max_items"] and length is None):
                continue  # a batcher of whole requests (ASR servers group files, then batch their chunks)
            out["batchers"].append({
                "name": name, "max_batch": items, "max_len_s": length / 1000 if length else None,
                "backoffs": int(b.get("backoffs") or 0),
                "lowered_len_s": b["max_len"] / 1000 if b.get("lowered") and b.get("max_len") else None})
        return out

    # -- one measurement

    def run(self, model: str) -> None:
        sched = self.scheduler
        p = sched.platform()
        s = p.get(model)
        if not s.gpu:
            self._fail(model, None, "runs on the CPU: no GPU memory to measure")
            return
        meta = render.endpoint_entry(p, s)
        if s.engine != "vllm" and mode(meta) is None:
            self._fail(model, None, f"{s.contract}/{s.api} has no batch measurement")
            return
        try:
            with sched.hold(HOLDER, model, self._stop, wait_s=30 * 60) as held:  # gives up 30 min after the request
                if held is None:
                    return
                run = sched.run_view(model)
                given, baseline = run.given, run.gpu_used_before
                total = p.config.gpu_total_gb
                gpu = sched.gpu
                if gpu:
                    total = gpu[0]
                given_gib = (given or p.config.util_limit) * total
                if s.engine == "vllm":
                    if given is None:  # its fixed memory is what it was given minus the KV cache
                        raise ValueError("started outside the catalog (or before the catalog restarted), so the "
                                         "memory it was given is unknown: restart it from the catalog to measure")
                    prof = kv_profile(self.runtime.logs(p, model, LOG_TAIL), given_gib)
                else:
                    prof = self._measure(p, s, meta, held.rid, baseline, given_gib)
                prof.given_gib = round(given_gib, 3)
                self._save(model, held.build_key, "measured", prof, None)
                log.info("batch profile of %s (%s): %s", model, held.build_key, json.dumps(prof.to_json())[:300])
        except (SchedulerError, ValueError, httpx.HTTPError) as exc:  # the lease's errors are Scheduler.hold's
            self._fail(model, None, str(exc))

    def _measure(self, p: Platform, s: Manifest, meta: dict[str, Any], rid: str, baseline: float | None,
                 given_gib: float) -> Profile:
        """The sweep under lease `rid`, which Scheduler.hold keeps alive."""
        url = self.scheduler.upstream_url(s.name)
        lengths, longest = lengths_for(s)
        max_n = s.batch.max_n if s.batch and s.batch.max_n else SWEEP_MAX_N
        if mode(meta) == "openai-asr":
            max_n = min(max_n, OPENAI_ASR_MAX_N)
        with httpx.Client(timeout=SLOW_S * 5) as http:
            lifted = http.post(f"{url}/platform/batch-limits", json=LIFT).status_code == 200

            def relift() -> None:  # an out of memory split keeps the server's limit lowered until this
                http.post(f"{url}/platform/batch-limits", json=LIFT)

            try:
                memory = Memory(url, http, lambda: self.runtime.gpu_memory(p), baseline)
                points, stops, how = sweep(sender(meta, url, http), memory, lengths, budget_gib=given_gib,
                                           max_n=max_n, stop=self._stop, log_fn=lambda m: log.info("%s: %s", s.name, m),
                                           relift=relift if lifted else None)
            finally:
                if lifted:
                    http.post(f"{url}/platform/batch-limits", json={"restore": True})
        if not points:
            raise ValueError("no batch ran: " + "; ".join(x["stopped"] for x in stops))
        base, a, b = fit(points)
        return Profile(kind="audio", base=round(base, 4), a=round(a, 6), b=round(b, 8), max_item_s=longest,
                       points=points, oom=stops, method=how + ("" if lifted else "; server limits not lifted (base 0.1.1)"))

    # -- the API view

    def view(self, model: str, build_key: str | None, *, vram_gb: float | None, item_s: float | None,
             item_tokens: int | None, n: int | None, full_gib: float | None) -> dict[str, Any]:
        row = self.stored(build_key) if build_key else None
        out: dict[str, Any] = {"model": model, "build_key": build_key, "measuring": self.status(model),
                               "status": row["status"] if row else "none", "error": row["error"] if row else None,
                               "measured_at": row["measured_at"] if row else None}
        prof = self.profile(build_key)
        if prof is None:
            return out
        out["profile"] = prof.to_json()
        out["table"] = table(prof, full_gib)
        unit = "tokens" if prof.kind == "kv" else "s"
        # a length in the other unit is no length for this profile (aicore and `just batch` default to 20 s)
        item = item_tokens if prof.kind == "kv" else item_s
        if item is None and (item_s if prof.kind == "kv" else item_tokens) is not None:
            out["note"] = f"this profile counts items in {unit}: give item_{'tokens' if unit == 'tokens' else 's'}"
        elif item is not None and prof.kind == "audio" and prof.max_item_s and item > prof.max_item_s:
            # the server splits a longer item into pieces of its own, so one item of `item` s is not one row
            out["note"] = f"items over {prof.max_item_s:g} s are split by the server: give item_s of at most that"
            item = None
        if item is not None:
            v = vram_gb or full_gib
            if v:
                out["recommendation"] = {"vram_gb": v, "item": item, "unit": unit, "batch": prof.recommend(v, item),
                                         "peak_gib": round(prof.peak(max(1, prof.recommend(v, item)), item), 3)}
            if n is not None:
                out["need"] = {"batch": n, "item": item, "unit": unit, "vram_gb": round(prof.need(n, item), 2)}
        return out


def full_budget_gib(p: Platform, sched: Scheduler) -> float:
    total = sched.gpu[0] if sched.gpu else p.config.gpu_total_gb
    return round(p.config.util_limit * total, 2)

