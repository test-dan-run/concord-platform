"""nemotron-3-diarization: oip-v2 conformance with a mocked model, shared diarization tensors, real model when present."""

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
META = service_meta("nemotron-3-diarization")


class FakeDiarizer(diar_mod.NemotronDiarizer):
    """Two speakers, returned out of time order with labels that are not in arrival order; the turns are shifted
    by the clip's length in seconds, so rows can be told apart."""

    def __init__(self):
        super().__init__("/nonexistent")
        self.calls = []

    def load(self):
        self.model = object()

    def diarize_batch(self, clips):
        self.calls.append([len(x) for x in clips])
        return [turns_mod.turns_to_tensors([]) if not np.any(x) else turns_mod.turns_to_tensors(
            [(d + 5.0, d + 7.5, "B"), (d + 0.5, d + 4.0, "A"), (d + 8.0, d + 9.0, "B"), (d + 3.5, d + 5.5, "A")])
            for x in clips for d in [len(x) / 16000 - 0.1]]


@pytest.fixture(scope="module")
def fake():
    return FakeDiarizer()


@pytest.fixture(scope="module")
def client(fake):
    with running(app_mod.build_app(fake)) as c:
        yield c


def infer(client, audio, sr=16000, **extra):
    """The single-clip request that callers sent before the batch signature."""
    return client.post("/v2/models/nemotron-3-diarization/infer", json={"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": [len(audio)], "data": list(map(float, audio))},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}], **extra})


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def test_turns_sorted_and_speakers_in_arrival_order(client):
    out = {o["name"]: o for o in infer(client, np.ones(1600)).json()["outputs"]}
    assert sorted(out) == ["SEGMENTS", "SPEAKERS"]  # the single-clip answer has no ROWS
    assert out["SEGMENTS"]["shape"] == [4, 2] and out["SPEAKERS"]["shape"] == [4]
    assert out["SEGMENTS"]["data"][::2] == [0.5, 3.5, 5.0, 8.0]
    assert out["SPEAKERS"]["data"] == [0, 0, 1, 1]  # "A" spoke first -> 0


def infer_batch(client, clips, sr=16000):
    from aisvc.batch import pad

    audio, lengths = pad(clips)
    return client.post("/v2/models/nemotron-3-diarization/infer", json={"inputs": [
        {"name": "AUDIO", "datatype": "FP32", "shape": list(audio.shape), "data": audio.reshape(-1).tolist()},
        {"name": "LENGTHS", "datatype": "INT32", "shape": [len(lengths)], "data": lengths.tolist()},
        {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [sr]}]})


def test_batch_rows_are_stacked_with_their_row_index(client):
    out = {o["name"]: o for o in infer_batch(client, [np.ones(1600), np.zeros(3200), np.ones(4800)]).json()["outputs"]}
    assert out["SEGMENTS"]["shape"] == [8, 2] and out["ROWS"]["data"] == [0] * 4 + [2] * 4
    assert out["SPEAKERS"]["data"] == [0, 0, 1, 1] * 2  # speaker indices start again in every row
    assert np.allclose(out["SEGMENTS"]["data"][::2], [0.5, 3.5, 5.0, 8.0] + [x + 0.2 for x in (0.5, 3.5, 5.0, 8.0)])


def test_rows_run_in_batches_within_the_limit_in_input_order(monkeypatch):
    monkeypatch.setenv("AISVC_MAX_BATCH", "2")
    model = FakeDiarizer()
    lengths = [16000 * 3, 1600, 16000 * 2, 8000, 0]
    with running(app_mod.build_app(model)) as c:
        out = {o["name"]: o for o in infer_batch(c, [np.ones(n) for n in lengths]).json()["outputs"]}
    assert model.calls[-3:] == [[0, 1600], [8000, 32000], [48000]]  # sorted by length, at most 2 per batch
    assert out["ROWS"]["data"] == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4  # input order; the empty row has no segments
    starts = np.asarray(out["SEGMENTS"]["data"], np.float32).reshape(-1, 2)[::4, 0]
    assert np.allclose(starts, [n / 16000 + 0.4 for n in lengths[:4]])


def test_stack_rows_unit():
    t = turns_mod.turns_to_tensors
    out = turns_mod.stack_rows([t([(0.0, 1.0, "x")]), t([]), t([(0.5, 2.0, "y"), (0.0, 0.4, "z")])])
    assert np.allclose(out["SEGMENTS"], [[0.0, 1.0], [0.0, 0.4], [0.5, 2.0]]) and out["SPEAKERS"].tolist() == [0, 0, 1]
    assert out["ROWS"].tolist() == [0, 2, 2] and out["ROWS"].dtype == out["SPEAKERS"].dtype == np.int32
    empty = turns_mod.stack_rows([])
    assert empty["SEGMENTS"].shape == (0, 2) and empty["SPEAKERS"].shape == empty["ROWS"].shape == (0,)


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


WEIGHTS = Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets"))) / "nemotron-3-diarization"


@pytest.fixture(scope="module")
def real():
    if importlib.util.find_spec("nemo") is None:
        pytest.skip("nemo not installed (runs inside the image or a matching venv)")
    if not (WEIGHTS / "Nemotron-3-Diarization.nemo").is_file():
        pytest.skip(f"no Nemotron-3-Diarization.nemo in {WEIGHTS}")
    d = diar_mod.NemotronDiarizer(WEIGHTS)
    d.load()
    return d


def test_parameters_are_rejected(client):
    r = infer(client, np.ones(1600), parameters={"num_speakers": 2})
    assert r.status_code == 400 and "takes none" in r.json()["error"]


def test_parse_segment():
    assert diar_mod.parse_segment("0.260 2.900 speaker_0") == (0.26, 2.9, 0)
    assert diar_mod.parse_segment("17.240 18.810 speaker_7") == (17.24, 18.81, 7)


def check_clip() -> np.ndarray:
    """The catalog's 10.56 s two-speaker check clip (16 kHz)."""
    import wave

    with wave.open(str(SERVICE_DIR.parents[1] / "catalog/fixtures/check_clip.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


@pytest.mark.weights
def test_real_model_batch_matches_one_by_one(real):
    """A padded batch must give each row the segments it gets alone (frames are 80 ms)."""
    a = check_clip()
    clips = [a, a[: 16000 * 6], a[16000 * 3:], np.concatenate([a, a]), a[:800]]
    batched = real.diarize_batch(clips)
    for clip, b in zip(clips, batched, strict=True):
        one = real.diarize_batch([clip])[0]
        assert one["SPEAKERS"].tolist() == b["SPEAKERS"].tolist(), (len(clip), one, b)
        assert one["SEGMENTS"].shape == b["SEGMENTS"].shape and np.allclose(one["SEGMENTS"], b["SEGMENTS"], atol=0.081)
    assert len(set(batched[0]["SPEAKERS"].tolist())) == 2 and batched[-1]["SEGMENTS"].shape == (0, 2)


@pytest.mark.weights
def test_real_model(real):
    out = real.diarize(np.zeros(16000 * 3, np.float32), 16000)
    assert out["SEGMENTS"].shape[1] == 2 and out["SPEAKERS"].dtype == np.int32
    with pytest.raises(ValueError, match="16000"):
        real.diarize(np.zeros(8000, np.float32), 8000)
    wav = os.environ.get("DIARIZATION_TEST_WAV")
    if wav:
        import soundfile as sf

        audio, sr = sf.read(wav, dtype="float32")
        out = real.diarize(audio, sr)
        assert len(set(out["SPEAKERS"].tolist())) >= 2 and out["SPEAKERS"][0] == 0
