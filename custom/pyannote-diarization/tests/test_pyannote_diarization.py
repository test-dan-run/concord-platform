"""pyannote-diarization: oip-v2 conformance with a mocked model, shared diarization tensors, real model when present."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
diar_mod = load_server(SERVICE_DIR, "diarizer")
turns_mod = load_server(SERVICE_DIR, "turns")
META = service_meta("pyannote-diarization")


class FakeDiarizer(diar_mod.PyannoteDiarizer):
    """Two speakers, returned out of time order with labels that are not in arrival order."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        self.model = self.pipeline = object()

    def diarize(self, audio, sample_rate, *args):
        self.calls.append((len(audio), sample_rate, args))
        if sample_rate != 16000:
            raise ValueError("SAMPLE_RATE must be 16000")
        if not np.any(audio):
            return turns_mod.turns_to_tensors([])
        return turns_mod.turns_to_tensors([(5.0, 7.5, "B"), (0.5, 4.0, "A"), (8.0, 9.0, "B"), (3.5, 5.5, "A")])


@pytest.fixture(scope="module")
def fake():
    return FakeDiarizer()


@pytest.fixture(scope="module")
def client(fake):
    with running(app_mod.build_app(fake)) as c:
        yield c


def infer(client, audio, sr=16000, **extra):
    """The single-clip request that callers sent before the batch signature."""
    return client.post("/v2/models/pyannote-diarization/infer", json={"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}], **extra})


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


def test_turns_sorted_and_speakers_in_arrival_order(client):
    out = {o["name"]: o for o in infer(client, np.ones(1600)).json()["outputs"]}
    assert sorted(out) == ["SEGMENTS", "SPEAKERS"]  # the single-clip answer has no ROWS
    assert out["SEGMENTS"]["shape"] == [4, 2] and out["SPEAKERS"]["shape"] == [4]
    assert out["SEGMENTS"]["data"][::2] == [0.5, 3.5, 5.0, 8.0]
    assert out["SPEAKERS"]["data"] == [0, 0, 1, 1]  # "A" spoke first -> 0


def infer_batch(client, clips, sr=16000, **extra):
    from aisvc.batch import pad

    audio, lengths = pad(clips)
    return client.post("/v2/models/pyannote-diarization/infer", json={"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [len(lengths)], "data": lengths.tolist()},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}], **extra})


def test_batch_rows_run_one_by_one_and_are_stacked_with_their_row_index(client, fake):
    del fake.calls[:]
    r = infer_batch(client, [np.ones(1600), np.zeros(3200), np.ones(4800)], parameters={"max_speakers": 3})
    out = {o["name"]: o for o in r.json()["outputs"]}
    assert [c[0] for c in fake.calls] == [1600, 3200, 4800]  # input order, every row with the request's parameters
    assert {c[2][0] for c in fake.calls} == {diar_mod.DiarParams(max_speakers=3)}
    assert out["SEGMENTS"]["shape"] == [8, 2] and out["ROWS"]["data"] == [0] * 4 + [2] * 4
    assert out["SPEAKERS"]["data"] == [0, 0, 1, 1] * 2  # speaker indices start again in every row


def test_aicore_client_with_the_batch_signature_still_sends_one_clip(client):
    """What Bench and Label do today: aicore.client(name).infer({"AUDIO": [-1], "SAMPLE_RATE": [1]})."""
    from aicore.oip import OIPClient

    c = OIPClient("http://testserver", META["name"], META["signature"], http=client)
    out = c.infer({"AUDIO": np.ones(1600, np.float32), "SAMPLE_RATE": np.array([16000], np.int32)})
    assert sorted(out) == ["SEGMENTS", "SPEAKERS"] and out["SEGMENTS"].shape == (4, 2) and out["SPEAKERS"].tolist() == [0, 0, 1, 1]


def test_batch_sizes_come_from_the_environment(monkeypatch):
    assert diar_mod.batch_size(diar_mod.EMBEDDING_BATCH, 32) == 32
    monkeypatch.setenv(diar_mod.EMBEDDING_BATCH, "128")
    assert diar_mod.batch_size(diar_mod.EMBEDDING_BATCH, 32) == 128
    monkeypatch.setenv(diar_mod.EMBEDDING_BATCH, "0")
    with pytest.raises(ValueError, match="PYANNOTE_EMBEDDING_BATCH"):
        diar_mod.batch_size(diar_mod.EMBEDDING_BATCH, 32)


def test_silence_gives_empty_tensors(client):
    out = {o["name"]: o for o in infer(client, np.zeros(1600)).json()["outputs"]}
    assert out["SEGMENTS"]["shape"] == [0, 2] and out["SPEAKERS"]["shape"] == [0]


def test_model_errors_are_400(client):
    r = infer(client, np.ones(800), sr=8000)
    assert r.status_code == 400 and "SAMPLE_RATE" in r.json()["error"]


def test_turns_to_tensors_unit():
    t = turns_mod.turns_to_tensors([(2.0, 3.0, "spk9"), (0.0, 1.0, "spk3"), (1.5, 2.5, "spk9")])
    assert t["SEGMENTS"].dtype == np.float32 and t["SPEAKERS"].dtype == np.int32
    assert t["SEGMENTS"].tolist() == [[0.0, 1.0], [1.5, 2.5], [2.0, 3.0]] and t["SPEAKERS"].tolist() == [0, 1, 1]


WEIGHTS = Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets"))) / "pyannote-speaker-diarization-3.1"


@pytest.fixture(scope="module")
def real():
    if importlib.util.find_spec("pyannote") is None:
        pytest.skip("pyannote.audio not installed (runs inside the image or a matching venv)")
    if not (WEIGHTS / "config.yaml").is_file():
        pytest.skip(f"no config.yaml in {WEIGHTS}")
    d = diar_mod.PyannoteDiarizer(WEIGHTS)
    d.load()
    return d


def test_parameters_are_validated_and_passed(client, fake):
    r = infer(client, np.ones(1600), parameters={"num_speakers": 2, "exclusive": True})
    assert r.status_code == 200 and fake.calls[-1][2][0] == diar_mod.DiarParams(num_speakers=2, exclusive=True)
    for bad in ({"num_speakers": 0}, {"speakers": 2}, {"exclusive": "yes"}):
        r = infer(client, np.ones(1600), parameters=bad)
        assert r.status_code == 400, bad


def check_clip() -> np.ndarray:
    """The catalog's 10.56 s two-speaker check clip (16 kHz)."""
    import wave

    with wave.open(str(SERVICE_DIR.parents[1] / "catalog/fixtures/check_clip.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


@pytest.mark.weights
def test_real_model_batch_sizes_do_not_change_the_result(real):
    """The pipeline's own batches are the service's batch settings: each window and (window, speaker) pair is
    computed on its own, so the batch size changes speed and memory, not the result."""
    a = np.concatenate([check_clip()] * 3)  # 32 s: 23 windows, so batches of 32 and of 4 differ
    pl = real.pipeline
    sizes = pl.segmentation_batch_size, pl.embedding_batch_size
    try:
        big = real.diarize(a, 16000, diar_mod.DiarParams())
        pl.segmentation_batch_size = pl.embedding_batch_size = 4
        small = real.diarize(a, 16000, diar_mod.DiarParams())
    finally:
        pl.segmentation_batch_size, pl.embedding_batch_size = sizes
    assert big["SPEAKERS"].tolist() == small["SPEAKERS"].tolist() and len(set(big["SPEAKERS"].tolist())) == 2
    assert np.allclose(big["SEGMENTS"], small["SEGMENTS"], atol=0.02)


@pytest.mark.weights
def test_real_model_rows_through_the_app(real):
    a = check_clip()
    with running(app_mod.build_app(real)) as c:
        out = {o["name"]: o for o in infer_batch(c, [a, a[: 16000 * 6], a[:800]]).json()["outputs"]}
    rows = np.asarray(out["ROWS"]["data"])
    segs = np.asarray(out["SEGMENTS"]["data"], np.float32).reshape(-1, 2)
    speakers = np.asarray(out["SPEAKERS"]["data"])
    for i, clip in enumerate([a, a[: 16000 * 6]]):
        alone = real.diarize(clip, 16000, diar_mod.DiarParams())
        assert np.allclose(segs[rows == i], alone["SEGMENTS"]) and speakers[rows == i].tolist() == alone["SPEAKERS"].tolist()
    assert not np.any(rows == 2)  # 50 ms: nothing to diarize


@pytest.mark.weights
def test_real_model(real):
    out = real.diarize(np.zeros(16000 * 3, np.float32), 16000, diar_mod.DiarParams())
    assert out["SEGMENTS"].shape == (0, 2)
    wav = os.environ.get("DIARIZATION_TEST_WAV")  # e.g. a two-speaker 16 kHz clip
    if wav:
        import soundfile as sf

        audio, sr = sf.read(wav, dtype="float32")
        for p in (diar_mod.DiarParams(), diar_mod.DiarParams(exclusive=True, max_speakers=4)):
            out = real.diarize(audio, sr, p)
            assert len(set(out["SPEAKERS"].tolist())) >= 2 and out["SPEAKERS"][0] == 0
        out8k = real.diarize(audio[::2], sr // 2, diar_mod.DiarParams())  # resampled by pyannote
        assert len(set(out8k["SPEAKERS"].tolist())) >= 2
