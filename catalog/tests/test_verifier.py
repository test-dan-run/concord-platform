"""Verifier probes against in-process services built with aisvc (no Docker, no models)."""

import time

import numpy as np
import pytest
from aisvc import create_oip_app, create_openai_app
from catalog_helpers import step, wait_builds
from fastapi.testclient import TestClient

from catalog.verifier import metrics, probes

TRUTH = probes.clip()[1]
SEG_SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
                      {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "example": [16000]}],
           "outputs": [{"name": "SEGMENTS", "datatype": "FP32", "shape": [-1, 2]},
                       {"name": "SPEAKERS", "datatype": "INT32", "shape": [-1]}]}


def meta(name, contract, api, signature=None, tasks=("speech-recognition",)):
    return {"name": name, "contract": contract, "api": api, "signature": signature, "tasks": list(tasks),
            "health_path": "/health"}


def client(app):
    c = TestClient(app)
    c.__enter__()
    state = getattr(app.state, "svc", None)
    if state is not None:
        assert state.wait(10)
    return c


def asr_app(text, words=True):
    def handler(req):
        out = {"text": text}
        if getattr(req, "response_format", "json") == "verbose_json":
            out["segments"] = [{"id": 0, "start": 0.4, "end": 9.9, "text": text}]
            if words:
                out["words"] = [{"word": w, "start": 0.4 + i * 0.3, "end": 0.6 + i * 0.3} for i, w in enumerate(text.split())]
        return out

    return create_openai_app("fake-asr", "audio.transcriptions", handler, background_load=False)


def test_asr_with_word_times_is_verified():
    rep = probes.verify(meta("fake-asr", "openai", "audio.transcriptions"), "http://testserver",
                        client(asr_app(TRUTH["transcript"])))
    assert rep.ok and rep.tasks == ["asr"] and rep.timing_level == "word", rep.lines
    assert rep.output_detail == ["word_timestamps", "segment_timestamps"]
    assert rep.sample["wer"] == 0.0 and rep.latency_s is not None
    assert rep.record(10.56)["protocol"]["routes"] == ["/v1/audio/transcriptions"]


def test_asr_far_from_the_clip_is_a_warning_not_a_failure():
    rep = probes.verify(meta("fake-asr", "openai", "audio.transcriptions"), "http://testserver",
                        client(asr_app("completely unrelated words here")))
    assert rep.ok and rep.tasks == ["asr"]
    assert any(x.name == "transcript" and x.status == "warn" for x in rep.lines)


def seg_app(speakers=True):
    turns = TRUTH["turns"]
    names = sorted({t["speaker"] for t in turns})
    sig = SEG_SIG if speakers else {**SEG_SIG, "outputs": SEG_SIG["outputs"][:1]}

    def handler(inputs, params):
        segs = np.array([[t["start"], t["end"]] for t in turns], dtype=np.float32)
        out = {"SEGMENTS": segs}
        if speakers:
            out["SPEAKERS"] = np.array([names.index(t["speaker"]) for t in turns], dtype=np.int32)
        return out

    name = "fake-diar" if speakers else "fake-vad"
    return create_oip_app(name, sig, handler, background_load=False), sig


def test_diariser_is_told_apart_from_vad():
    app, sig = seg_app(speakers=True)
    rep = probes.verify(meta("fake-diar", "oip-v2", "infer", sig), "http://testserver", client(app))
    assert rep.ok and rep.tasks == ["diar"] and rep.output_detail == ["segment_timestamps", "segment_speakers"], rep.lines
    assert rep.sample["der"] == 0.0
    app, sig = seg_app(speakers=False)
    rep = probes.verify(meta("fake-vad", "oip-v2", "infer", sig), "http://testserver", client(app))
    assert rep.ok and rep.tasks == ["vad"] and rep.output_detail == ["segment_timestamps"], rep.lines


def test_text_chat_model_gets_protocol_checks_only():
    app = create_openai_app("fake-chat", "chat.completions", lambda req: "hello", background_load=False)
    rep = probes.verify(meta("fake-chat", "openai", "chat.completions", tasks=("chat",)), "http://testserver", client(app))
    assert rep.usable_as is None and rep.ok and rep.tasks == []


def test_chat_audio_that_ignores_the_audio_fails():
    app = create_openai_app("fake-chat", "chat.completions", lambda req: "I like turtles", background_load=False)
    rep = probes.verify(meta("fake-chat", "openai", "chat.completions"), "http://testserver", client(app))
    assert rep.usable_as == "chat-audio" and not rep.ok and "audio part ignored" in rep.lines[-2].reason


def test_metrics():
    assert metrics.wer("Hello, World!", "hello world") == 0.0
    assert metrics.wer("a b c d", "a x c") == 0.5
    ref = [metrics.Turn(0, 2, "a"), metrics.Turn(2, 4, "b")]
    assert metrics.der(ref, [metrics.Turn(0, 2, "x"), metrics.Turn(2, 4, "y")], 4) == 0.0
    assert metrics.der(ref, [metrics.Turn(0, 4, "x")], 4) == pytest.approx(0.5, abs=0.1)
    missed, fa = metrics.vad_errors(ref, [(0, 2)], 4)
    assert missed == pytest.approx(1.5, abs=0.05) and fa == 0.0


def test_verifier_records_a_capability_per_build(cat, monkeypatch):
    def fake_verify(meta, base_url, http=None):
        rep = probes.Report(meta["name"], "segmenter", tasks=["vad"], timing_level="segment",
                            output_detail=["segment_timestamps"], latency_s=0.5)
        rep.add("result", "pass", "detected: VAD")
        return rep

    monkeypatch.setattr(probes, "verify", fake_verify)
    wait_builds(cat, ["silero-vad"])
    cat.registry.register("default-vad", tags=["call-centre"])
    assert cat.registry.record("silero-vad")["capabilities"]["status"] == "pending"
    cat.verifier.start()  # its thread takes a lease and waits; this test drives the scheduler
    deadline = time.monotonic() + 10
    while cat.verifier.status("silero-vad") is not None:
        assert time.monotonic() < deadline
        step(cat, 1)
        time.sleep(0.02)
    cat.verifier.shutdown()
    rec = cat.registry.record("silero-vad")
    caps = rec["capabilities"]
    assert caps["status"] == "verified" and caps["tasks"] == ["vad"] and caps["build_key"] == rec["build_key"]
    assert caps["rtf"] == pytest.approx(0.5 / 10.56, abs=1e-3) and rec["leased_by"] == []
    assert [m["id"] for m in cat.registry.records(task=["vad"], tag=["call-centre"])] == ["silero-vad"]


SE_SIG = {"inputs": SEG_SIG["inputs"],
          "outputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1]},
                      {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1]}]}


def enhancer_app(make):
    """An OIP enhancer whose output is make(noisy input) -> (audio, sample rate)."""
    def handler(inputs, params):
        audio, rate = make(np.asarray(inputs["AUDIO"], dtype=np.float32))
        return {"AUDIO": np.asarray(audio, dtype=np.float32), "SAMPLE_RATE": np.array([rate], dtype=np.int32)}

    return create_oip_app("fake-se", SE_SIG, handler, background_load=False)


def verify_enhancer(make, tasks=("speech-enhancement",)):
    m = meta("fake-se", "oip-v2", "infer", SE_SIG, tasks=tasks)
    return probes.verify(m, "http://testserver", client(enhancer_app(make)))


def line(rep, name):
    return next(x for x in rep.lines if x.name == name)


def test_enhancer_that_cleans_the_clip_passes():
    clean = probes.clip()[0]
    rep = verify_enhancer(lambda x: (clean, 16000))
    assert rep.usable_as == "enhancer" and rep.ok and rep.tasks == ["enhance"], rep.lines
    assert rep.output_detail == ["enhanced_audio"] and line(rep, "si-sdr").status == "pass"
    assert rep.sample["si_sdr_output_db"] > rep.sample["si_sdr_input_db"] and rep.sample["length_diff_samples"] == 0
    assert rep.record(10.56)["protocol"]["routes"] == ["/v2/models/fake-se/infer"]


def test_enhancer_that_changes_nothing_is_a_warning():
    rep = verify_enhancer(lambda x: (x, 16000))
    assert rep.ok and rep.tasks == ["enhance"] and line(rep, "si-sdr").status == "warn", rep.lines
    assert rep.sample["si_sdr_output_db"] == rep.sample["si_sdr_input_db"]


@pytest.mark.parametrize("make,name", [
    (lambda x: (x[:-320], 16000), "length"),  # 20 ms short
    (lambda x: (np.where(np.arange(len(x)) % 100 == 0, np.nan, x), 16000), "values"),
    (lambda x: (x, 8000), "sample rate"),
])
def test_enhancer_output_faults_fail(make, name):
    rep = verify_enhancer(make)
    assert not rep.ok and line(rep, name).status == "fail" and rep.tasks == [], rep.lines


def test_dereverberation_gets_a_room_response_not_babble():
    seen = []
    rep = verify_enhancer(lambda x: (seen.append(x) or x, 16000), tasks=("dereverberation",))
    assert np.allclose(seen[-1], probes.degraded("reverb"), atol=1e-6) and "room response" in line(rep, "request").reason
    assert not np.allclose(probes.degraded("reverb"), probes.degraded("noise"))
    assert abs(metrics.si_sdr(probes.clip()[0], probes.degraded("noise")) - probes.SNR_DB) < 1.0


def test_itn_service_is_verified_by_its_digits():
    from factories import toy_itn_app

    m = meta("itn-toy", "openai-inspired", "text.itn", tasks=("inverse-text-normalization",))
    rep = probes.verify(m, "http://testserver", client(toy_itn_app()))
    assert rep.usable_as == "itn" and rep.ok and rep.tasks == ["itn"], rep.lines
    assert rep.output_detail == ["spans", "class:CARDINAL"] and line(rep, "written form").status == "pass"
    assert rep.sample["itn"][0] == {"input": "twenty one", "output": "21"}


def test_itn_service_that_returns_its_input_fails():
    from factories import toy_itn_app

    rep = probes.verify(meta("itn-toy", "openai-inspired", "text.itn"), "http://testserver",
                        client(toy_itn_app(lambda texts, options: [(t, []) for t in texts])))
    assert rep.usable_as == "itn" and not rep.ok and line(rep, "written form").status == "fail", rep.lines


def test_metrics_si_sdr():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    assert metrics.si_sdr(ref, 0.5 * ref) == float("inf")  # scale-invariant
    noisy = ref + 0.1 * rng.standard_normal(16000)
    assert metrics.si_sdr(ref, noisy) == pytest.approx(20.0, abs=0.5)
