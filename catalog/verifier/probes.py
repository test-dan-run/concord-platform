"""Verification of one live build: the platform's conformance checks, then a capability probe.

| Model | Probe on the check clip | Result |
|---|---|---|
| openai / audio.transcriptions | verbose_json + word granularity, then segment, then json | ASR; timing = word, segment or text; WER |
| openai / chat.completions with an audio task | input_audio part + "transcribe" | ASR (text only) if the answer matches the clip |
| oip-v2 with AUDIO + SAMPLE_RATE in, SEGMENTS out | infer | SEGMENTS -> VAD; + SPEAKERS -> diarisation |
| oip-v2 with AUDIO + SAMPLE_RATE in, AUDIO out | infer on the clip with babble at 5 dB SNR (a synthetic room response for dereverberation) | speech enhancement; SI-SDR against the clean clip |
| either of the two above with AUDIO [-1, -1] + LENGTHS (a batch signature) | infer with a batch of one row | as above |
| openai-inspired / text.itn | spoken-form phrases with return_spans | ITN when each output has the expected digits |
| anything else | conformance only | verified on the protocol; no audio capability |

Every line is pass / warn / fail / info with a plain-language reason (Concord's check, moved here).
"""

from __future__ import annotations

import base64
import importlib
import json
import re
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from tools.schema import PLATFORM_DIR

from . import metrics

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WER_WARN = 0.5  # transcripts this far from the clip's text are flagged
WER_FAIL_CHAT = 0.8  # chat answers this far away mean the audio part was ignored
AUDIO_TASKS = {"speech-recognition", "audio-understanding", "audio-transcription", "audio"}
CHAT_PROMPT = ("Transcribe the speech in this audio exactly as spoken. Reply with the transcript only: no quotes, "
               "no labels, no commentary.")
TIMEOUT_S = 600.0
SNR_DB = 5.0  # babble level of the enhancement probe's noisy input
BABBLE_COPIES = 4
LENGTH_TOLERANCE = 160  # samples: 10 ms at 16 kHz
CLIP_FRACTION = 0.001  # more than 0.1% of samples at full scale is clipping
DEREVERB_TASKS = {"dereverberation"}
ITN_PHRASES = [("twenty one", "21"), ("three thirty p m", "3:30"), ("twelve", "12")]
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class Line:
    name: str
    status: str  # pass | warn | fail | info
    reason: str


@dataclass
class Report:
    service: str
    usable_as: str | None
    tasks: list[str] = field(default_factory=list)
    timing_level: str | None = None
    output_detail: list[str] = field(default_factory=list)
    protocol: dict[str, Any] = field(default_factory=dict)
    lines: list[Line] = field(default_factory=list)
    sample: dict[str, Any] = field(default_factory=dict)
    latency_s: float | None = None
    conformance_ok: bool = True

    def add(self, name: str, status: str, reason: str) -> None:
        self.lines.append(Line(name, status, reason))

    @property
    def ok(self) -> bool:
        return self.conformance_ok and (self.usable_as is None or bool(self.tasks))

    def record(self, duration: float) -> dict[str, Any]:
        return {"usable_as": self.usable_as, "tasks": self.tasks, "timing_level": self.timing_level,
                "output_detail": self.output_detail, "protocol": self.protocol,
                "rtf": round(self.latency_s / duration, 4) if self.latency_s is not None else None}

    def report(self) -> dict[str, Any]:
        return {"ok": self.ok, "lines": [asdict(x) for x in self.lines], "sample": self.sample}


@cache
def clip() -> tuple[np.ndarray, dict[str, Any], bytes]:
    """(16 kHz mono float32 audio, truth, the WAV bytes) of the 10.56 s two-speaker check clip."""
    path = FIXTURES / "check_clip.wav"
    raw = path.read_bytes()
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    truth = json.loads((FIXTURES / "check_clip.json").read_text())
    return audio, truth, raw


def audio_form(signature: dict[str, Any] | None) -> str:
    """How an oip-v2 signature takes audio: "batch" (AUDIO [-1, -1] zero-padded rows + LENGTHS [-1]), else "single"
    (one clip, AUDIO [-1]). The platform's batch enhancers and diarizers also take the single form, but the probe
    sends the declared one."""
    ins = {t["name"]: t for t in (signature or {}).get("inputs", [])}
    return "batch" if "LENGTHS" in ins and len(ins.get("AUDIO", {}).get("shape") or []) == 2 else "single"


def usable_as(meta: dict[str, Any]) -> str | None:
    """The probe that applies. For oip-v2 it goes by tensor names, so the single-clip and the batch signatures
    (AUDIO [-1] or AUDIO [-1, -1] + LENGTHS) of an enhancer, VAD or diarizer are both detected."""
    if meta["contract"] == "openai" and meta["api"] == "audio.transcriptions":
        return "asr"
    if meta["contract"] == "openai" and meta["api"] == "chat.completions" and AUDIO_TASKS & set(meta.get("tasks") or []):
        return "chat-audio"
    if meta["contract"] == "oip-v2":
        sig = meta.get("signature") or {}
        ins = {t["name"] for t in sig.get("inputs", [])}
        outs = {t["name"] for t in sig.get("outputs", [])}
        if {"AUDIO", "SAMPLE_RATE"} <= ins and "SEGMENTS" in outs:
            return "segmenter"
        if {"AUDIO", "SAMPLE_RATE"} <= ins and "AUDIO" in outs:
            return "enhancer"
    if meta["contract"] == "openai-inspired" and meta["api"] == "text.itn":
        return "itn"
    return None


def _short(text: str) -> str:
    return text if len(text) < 240 else text[:240] + "..."


@cache
def _conformance():
    contracts = str(PLATFORM_DIR / "contracts")
    if contracts not in sys.path:
        sys.path.insert(0, contracts)
    return importlib.import_module("conformance.checks")


def conformance(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    checks = _conformance().checks_for(meta)
    if meta.get("hosted"):  # a provider has no platform health route; reachability is the connection test's job
        checks = [c for c in checks if c.__name__ != "health_ok"]
    failed = 0
    for check in checks:
        try:
            check(http, meta)
        except AssertionError as exc:
            failed += 1
            rep.add(f"contract: {check.__name__}", "fail", _short(str(exc) or "assertion failed"))
        except httpx.HTTPError as exc:
            failed += 1
            rep.add(f"contract: {check.__name__}", "fail", f"{type(exc).__name__}: {_short(str(exc))}")
    rep.conformance_ok = failed == 0
    if not failed:
        rep.add("contract", "pass", f"{meta['contract']} / {meta['api']}: {len(checks)} conformance checks passed")


def _judge_transcript(rep: Report, text: str, truth: dict[str, Any], warn_only: bool) -> None:
    rep.sample["transcript"] = text
    if not text.strip():
        rep.add("transcript", "warn" if warn_only else "fail", "empty transcript for a clip with speech")
        if not warn_only:
            return
    w = metrics.wer(truth["transcript"], text)
    rep.sample["wer"] = round(w, 4)
    if w <= WER_WARN:
        rep.add("transcript", "pass", f"transcript matches the clip (WER {w:.1%})")
        rep.tasks.append("asr")
    elif warn_only or w <= WER_FAIL_CHAT:
        rep.add("transcript", "warn", f"transcript is far from the clip's text (WER {w:.1%}); check the model before "
                "trusting its scores")
        rep.tasks.append("asr")
    else:
        rep.add("transcript", "fail", f"audio part ignored: the answer is unrelated to the clip (WER {w:.0%})")


def probe_asr(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    audio, truth, wav = clip()
    body = None
    for timing in ("word", "segment", "text"):
        data: dict[str, Any] = {"model": meta["name"], "response_format": "json" if timing == "text" else "verbose_json"}
        if timing == "word":
            data["timestamp_granularities[]"] = "word"
        t0 = time.perf_counter()
        r = http.post("/v1/audio/transcriptions", data=data, files={"file": ("audio.wav", wav, "audio/wav")},
                      timeout=TIMEOUT_S)
        latency = time.perf_counter() - t0
        if r.status_code < 400:
            rep.protocol["response_format"] = data["response_format"]
            rep.latency_s = latency
            rep.add("request", "pass", f"{'verbose_json + word timestamps' if timing == 'word' else timing} request "
                    f"accepted ({latency:.2f} s)")
            body = r.json()
            break
        if 400 <= r.status_code < 500 and timing != "text":
            rep.add("request", "info", f"{timing} timing request rejected ({r.status_code}): {_short(r.text)}; trying a "
                    "simpler request")
            continue
        rep.add("request", "fail", f"transcription request failed ({r.status_code}): {_short(r.text)}")
        return
    if body is None:
        rep.add("request", "fail", "no transcription request was accepted")
        return
    words = [w for w in body.get("words") or [] if str(w.get("word", "")).strip()]
    segments = [s for s in body.get("segments") or [] if "start" in s and "end" in s]
    if words:
        rep.timing_level = "word"
        rep.output_detail = ["word_timestamps"] + (["segment_timestamps"] if segments else [])
        rep.add("timing", "pass", f"word times returned ({len(words)} words)")
    elif segments:
        rep.timing_level = "segment"
        rep.output_detail = ["segment_timestamps"]
        rep.add("timing", "pass", f"segment times only ({len(segments)} segments)")
    else:
        rep.timing_level = "text"
        rep.add("timing", "warn", "text only, no timestamps; per-word audio and error regions are unavailable")
    _judge_transcript(rep, (body.get("text") or "").strip(), truth, warn_only=True)


def probe_chat_audio(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    audio, truth, wav = clip()
    body = {"model": meta["name"], "temperature": 0.0, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": CHAT_PROMPT},
                {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}]}]}
    t0 = time.perf_counter()
    r = http.post("/v1/chat/completions", json=body, timeout=TIMEOUT_S)
    latency = time.perf_counter() - t0
    if 400 <= r.status_code < 500:
        rep.add("request", "fail", f"audio part rejected ({r.status_code}): the model can't take audio: {_short(r.text)}")
        return
    if r.status_code >= 500:
        rep.add("request", "fail", f"chat request failed ({r.status_code}): {_short(r.text)}")
        return
    rep.latency_s = latency
    rep.add("request", "pass", f"chat request with an input_audio part accepted ({latency:.2f} s)")
    try:
        content = r.json()["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, ValueError):
        content = ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    _judge_transcript(rep, _THINK.sub("", content).strip().strip('"').strip(), truth, warn_only=False)
    if "asr" in rep.tasks:
        rep.timing_level = "text"
        rep.add("timing", "warn", "text only: audio LLMs return no timestamps; pipelines take timing from the "
                "segmentation stage")


def _infer_clip(http: httpx.Client, meta: dict[str, Any], audio: np.ndarray) -> dict[str, np.ndarray]:
    """One 16 kHz clip through the service, in the signature's form (a batch signature gets one row)."""
    from aicore.oip import OIPClient

    c = OIPClient(str(http.base_url).rstrip("/"), meta["name"], meta.get("signature"), http=http)
    inputs = {"AUDIO": audio, "SAMPLE_RATE": np.array([16000], dtype=np.int32)}
    if audio_form(meta.get("signature")) == "batch":
        inputs |= {"AUDIO": audio[None, :], "LENGTHS": np.array([len(audio)], dtype=np.int32)}
    return c.infer(inputs)


def _form(meta: dict[str, Any]) -> str:
    return " (batch signature: AUDIO [-1, -1] + LENGTHS)" if audio_form(meta.get("signature")) == "batch" else ""


def probe_segmenter(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    audio, truth, _ = clip()
    outputs = {t["name"] for t in (meta.get("signature") or {}).get("outputs", [])}
    rep.add("signature", "pass", "AUDIO + SAMPLE_RATE in, SEGMENTS" + (" + SPEAKERS" if "SPEAKERS" in outputs else "")
            + " out" + _form(meta))
    t0 = time.perf_counter()
    try:
        out = _infer_clip(http, meta, audio)
    except Exception as exc:  # noqa: BLE001 - any failure is the report's answer
        rep.add("request", "fail", f"infer failed: {_short(str(exc))}")
        return
    rep.latency_s = time.perf_counter() - t0
    rep.add("request", "pass", f"infer accepted ({rep.latency_s:.2f} s)")
    if "SEGMENTS" not in out:
        rep.add("segments", "fail", "no SEGMENTS output returned")
        return
    dur = len(audio) / 16000
    segs = np.asarray(out["SEGMENTS"], dtype=float).reshape(-1, 2).round(4).tolist()
    speakers = np.asarray(out["SPEAKERS"]).reshape(-1).astype(int).tolist() if "SPEAKERS" in out else None
    rep.sample["segments"] = segs[:50]
    rep.sample["speakers"] = (speakers or [])[:50]
    if not segs:
        rep.add("segments", "fail", "no segments returned for a clip with speech")
        return
    if "ROWS" in out and np.asarray(out["ROWS"]).reshape(-1).tolist() != [0] * len(segs):
        rep.add("segments", "fail", f"ROWS must give row 0 for each of the {len(segs)} segments of a one-row batch")
        return
    bad = [s for s in segs if s[0] < -0.05 or s[1] > dur + 0.5 or s[1] < s[0]]
    if bad:
        rep.add("segments", "fail", f"{len(bad)} segment(s) outside the clip's bounds or reversed, e.g. {bad[0]}")
        return
    ref = [metrics.Turn(t["start"], t["end"], t["speaker"]) for t in truth["turns"]]
    rep.timing_level = "segment"
    rep.output_detail = ["segment_timestamps"]
    if speakers is None:
        rep.tasks.append("vad")
        missed, fa = metrics.vad_errors(ref, [(s, e) for s, e in segs], dur)
        rep.add("segments", "pass", f"speech-only segments: VAD. {len(segs)} segments; missed speech {missed:.2f} s, "
                f"false alarm {fa:.2f} s")
        return
    if len(speakers) != len(segs):
        rep.add("speakers", "fail", f"SPEAKERS has {len(speakers)} values for {len(segs)} segments")
        return
    rep.tasks.append("diar")
    rep.output_detail.append("segment_speakers")
    hyp = [metrics.Turn(s, e, f"spk{k}") for (s, e), k in zip(segs, speakers, strict=True)]
    d = metrics.der(ref, hyp, dur)
    rep.sample["der"] = None if d is None else round(d, 4)
    n_spk, want = len(set(speakers)), len(truth["speakers"])
    shown = "n/a" if d is None else f"{d:.1%}"
    if n_spk == want:
        rep.add("speakers", "pass", f"diarisation: found {n_spk} speakers, as expected (DER on the clip {shown})")
    else:
        rep.add("speakers", "warn", f"diarisation: found {n_spk} speaker(s) on a {want}-speaker clip (DER {shown})")


@cache
def degraded(kind: str) -> np.ndarray:
    """The check clip made worse in a fixed way: `noise` adds babble at SNR_DB, `reverb` a room response.

    The babble is BABBLE_COPIES reversed, time-shifted copies of the clip summed together, so the probe
    needs no extra audio file. The room response is exponentially decaying noise (RT60 0.5 s) after a
    direct path at sample 0, with as much energy as the direct path (0 dB direct-to-reverberant ratio),
    so the reverberant clip stays aligned with the clean one.
    """
    audio = clip()[0].astype(np.float64)
    n = len(audio)
    if kind == "reverb":
        rng = np.random.default_rng(7)
        t = np.arange(int(0.5 * 16000)) / 16000
        tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.5)
        tail[0] = 0.0
        rir = tail / np.sqrt(np.sum(tail ** 2))
        rir[0] = 1.0
        out = np.convolve(audio, rir)[:n]
    else:
        rev = audio[::-1]
        babble = sum(np.roll(rev, (k + 1) * n // (BABBLE_COPIES + 1)) for k in range(BABBLE_COPIES))
        gain = np.sqrt(np.mean(audio ** 2) / (np.mean(babble ** 2) * 10 ** (SNR_DB / 10)))
        out = audio + gain * babble
    peak = np.max(np.abs(out))
    if peak > 0.95:  # keep the input itself unclipped
        out *= 0.95 / peak
    return out.astype(np.float32)


def probe_enhancer(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    clean = clip()[0]
    kind = "reverb" if DEREVERB_TASKS & set(meta.get("tasks") or []) else "noise"
    noisy = degraded(kind)
    rep.add("signature", "pass", "AUDIO + SAMPLE_RATE in, AUDIO out" + _form(meta))
    t0 = time.perf_counter()
    try:
        out = _infer_clip(http, meta, noisy)
    except Exception as exc:  # noqa: BLE001 - any failure is the report's answer
        rep.add("request", "fail", f"infer failed: {_short(str(exc))}")
        return
    rep.latency_s = time.perf_counter() - t0
    what = "a synthetic room response" if kind == "reverb" else f"babble noise at {SNR_DB:.0f} dB SNR"
    rep.add("request", "pass", f"infer on the clip with {what} accepted ({rep.latency_s:.2f} s)")
    if "AUDIO" not in out:
        rep.add("audio", "fail", "no AUDIO output returned")
        return
    est = np.asarray(out["AUDIO"], dtype=np.float64)
    est = est[0, : int(np.asarray(out["LENGTHS"]).reshape(-1)[0])] if est.ndim == 2 and "LENGTHS" in out else est.reshape(-1)
    rate = int(np.asarray(out["SAMPLE_RATE"]).reshape(-1)[0]) if "SAMPLE_RATE" in out else None
    diff = len(est) - len(noisy)
    rep.sample["length_diff_samples"] = diff
    failed = False
    if rate is None:
        rep.add("sample rate", "info", "no SAMPLE_RATE output; taken to be the input rate")
    elif rate != 16000:
        rep.add("sample rate", "fail", f"output SAMPLE_RATE is {rate}, not the input's 16000")
        failed = True
    if abs(diff) > LENGTH_TOLERANCE:
        rep.add("length", "fail", f"output is {abs(diff) / 16:.1f} ms {'longer' if diff > 0 else 'shorter'} than the "
                "input (more than 10 ms)")
        failed = True
    if not np.all(np.isfinite(est)):
        rep.add("values", "fail", f"{int(np.sum(~np.isfinite(est)))} NaN or infinite sample(s) in the output")
        return
    clipped = float(np.mean(np.abs(est) >= 1.0))
    if clipped > CLIP_FRACTION:
        rep.add("clipping", "fail", f"{clipped:.2%} of output samples are at or past full scale")
        failed = True
    if failed:
        return
    before, after = metrics.si_sdr(clean, noisy), metrics.si_sdr(clean, est)
    rep.sample["si_sdr_input_db"] = round(before, 2)
    rep.sample["si_sdr_output_db"] = round(after, 2)
    rep.tasks.append("enhance")
    rep.output_detail = ["enhanced_audio"]
    if after > before:
        rep.add("si-sdr", "pass", f"SI-SDR against the clean clip improves from {before:.1f} dB to {after:.1f} dB")
    else:
        rep.add("si-sdr", "warn", f"SI-SDR does not improve on the degraded input ({before:.1f} dB in, {after:.1f} dB "
                "out); check the model before trusting it")


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def probe_itn(rep: Report, http: httpx.Client, meta: dict[str, Any]) -> None:
    phrases = [p for p, _ in ITN_PHRASES]
    t0 = time.perf_counter()
    r = http.post("/v1/text/itn", json={"model": meta["name"], "input": phrases, "options": {"return_spans": True}},
                  timeout=TIMEOUT_S)
    if r.status_code >= 400:
        rep.add("request", "fail", f"ITN request failed ({r.status_code}): {_short(r.text)}")
        return
    rep.latency_s = time.perf_counter() - t0
    rep.add("request", "pass", f"{len(phrases)} spoken-form phrases with return_spans accepted ({rep.latency_s:.2f} s)")
    try:
        items = sorted(r.json()["data"], key=lambda d: d["index"])
        outputs = [str(d["output"]) for d in items]
    except (KeyError, TypeError, ValueError):
        rep.add("outputs", "fail", f"response has no data[].output: {_short(r.text)}")
        return
    if len(outputs) != len(phrases):
        rep.add("outputs", "fail", f"{len(outputs)} outputs for {len(phrases)} inputs")
        return
    rep.sample["itn"] = [{"input": p, "output": o} for p, o in zip(phrases, outputs, strict=True)]
    hits = [_digits(want) in _digits(out) and _digits(out) != "" for (_, want), out in zip(ITN_PHRASES, outputs, strict=True)]
    shown = "; ".join(f"“{p}” → “{o}”" for p, o in zip(phrases, outputs, strict=True))
    classes = sorted({str(s.get("class")) for d in items for s in d.get("spans") or [] if s.get("class")})
    if not any(hits):
        rep.add("written form", "fail", f"no output has the expected digits: {_short(shown)}")
        return
    rep.tasks.append("itn")
    rep.output_detail = ["spans"] + [f"class:{c}" for c in classes]
    if all(hits):
        rep.add("written form", "pass", f"every output has the expected digits: {_short(shown)}")
    else:
        rep.add("written form", "warn", f"{sum(hits)} of {len(hits)} outputs have the expected digits: {_short(shown)}")


ROUTES = {"audio.transcriptions": "/v1/audio/transcriptions", "chat.completions": "/v1/chat/completions",
          "completions": "/v1/completions", "embeddings": "/v1/embeddings", "audio.speech": "/v1/audio/speech",
          "text.transform": "/v1/text/transform", "text.itn": "/v1/text/itn"}


def _probe(rep: Report, client: httpx.Client, meta: dict[str, Any]) -> None:
    """The capability probe for rep.usable_as."""
    if rep.usable_as == "asr":
        probe_asr(rep, client, meta)
    elif rep.usable_as == "chat-audio":
        first = len(rep.lines)
        probe_chat_audio(rep, client, meta)
        if not rep.tasks and meta.get("audio_optional"):  # hosted chat: a text model is still a model
            rep.lines[first:] = [x if x.status != "fail" else Line(x.name, "info", x.reason) for x in rep.lines[first:]]
            rep.add("audio", "info", "no audio capability: verified as a text model")
            rep.usable_as = None
            rep.latency_s = None
    elif rep.usable_as == "segmenter":
        probe_segmenter(rep, client, meta)
    elif rep.usable_as == "enhancer":
        probe_enhancer(rep, client, meta)
    elif rep.usable_as == "itn":
        probe_itn(rep, client, meta)


def verify(meta: dict[str, Any], base_url: str, http: httpx.Client | None = None) -> Report:
    """Run every check against the service at base_url (its root URL). meta is the endpoint-map entry."""
    rep = Report(meta["name"], usable_as(meta))
    route = f"/v2/models/{meta['name']}/infer" if meta["contract"] == "oip-v2" else ROUTES.get(meta["api"], "")
    rep.protocol = {"contract": meta["contract"], "api": meta["api"], "routes": [route] if route else []}
    with http or httpx.Client(base_url=base_url, timeout=TIMEOUT_S) as client:
        conformance(rep, client, meta)
        try:
            _probe(rep, client, meta)
        except Exception as exc:  # noqa: BLE001 - a crashed probe is one failed line; the lines before it stay
            rep.tasks.clear()
            rep.add("probe", "fail", f"the {rep.usable_as} probe crashed: {type(exc).__name__}: {_short(str(exc))}")
    if rep.usable_as and not rep.tasks:
        rep.add("result", "fail", f"no {rep.usable_as} capability detected")
    elif rep.tasks:
        names = {"asr": "ASR", "diar": "diarisation", "vad": "VAD", "enhance": "speech enhancement", "itn": "ITN"}
        rep.add("result", "pass" if rep.conformance_ok else "fail",
                "detected: " + ", ".join(names.get(t, t) for t in rep.tasks))
    elif rep.conformance_ok:
        rep.add("result", "pass", "protocol verified (no audio capability probe applies)")
    return rep
