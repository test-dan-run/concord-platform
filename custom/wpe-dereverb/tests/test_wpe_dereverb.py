"""wpe-dereverb: oip-v2 conformance and dereverberation on a synthetic reverberant signal."""

from pathlib import Path

import numpy as np
import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
wpe_mod = load_server(SERVICE_DIR, "dereverb")
META = service_meta("wpe-dereverb")

pytest.importorskip("nara_wpe")
pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def client():
    with running(app_mod.build_app()) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


def test_server_signature_matches_manifest():
    assert app_mod.SIGNATURE == {k: [{f: t[f] for f in ("name", "datatype", "shape")} for t in v]
                                 for k, v in META["signature"].items()}


BURST, GAP = 3200, 8000  # 200 ms noise bursts every 0.5 s


def reverberant(sr=16000, n_bursts=12, rt60=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = GAP * (n_bursts + 1)
    dry = np.zeros(n)
    for k in range(n_bursts):
        start = GAP // 2 + k * GAP
        dry[start:start + BURST] += rng.normal(0, 1, BURST) * np.hanning(BURST)
    t = np.arange(int(sr * rt60)) / sr
    rir = rng.normal(0, 1, len(t)) * np.exp(-6.9 * t / rt60)
    rir[0] = 8.0
    wet = np.convolve(dry, rir)[:n]
    return dry.astype(np.float32), (wet / np.abs(wet).max()).astype(np.float32)


def tail_energy(x: np.ndarray, n_bursts=12) -> float:
    """Energy from 50 ms after each burst ends to just before the next: only the reverberant tail is there."""
    out = 0.0
    for k in range(n_bursts):
        end = GAP // 2 + k * GAP + BURST
        out += float(np.sum(x[end + 800: end + GAP - BURST - 200] ** 2))
    return out


def test_wpe_reduces_the_reverberant_tail():
    # Single-channel WPE is a modest suppressor (about 14% less tail energy here); M4's bake-off decides
    # whether that is dry enough for cloning.
    _, wet = reverberant()
    out = wpe_mod.dereverb(wet, 16000, wpe_mod.WpeParams())
    assert out.shape == wet.shape and out.dtype == np.float32
    assert tail_energy(out) < 0.92 * tail_energy(wet)


def test_short_clip_is_returned_unchanged():
    x = np.linspace(-1, 1, 100, dtype=np.float32)
    assert np.allclose(wpe_mod.dereverb(x, 16000, wpe_mod.WpeParams()), x)


def test_bad_parameters_are_400(client):
    b = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [4], "data": [0.0] * 4},
                    {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [16000]}],
         "parameters": {"taps": 100}}
    assert client.post("/v2/models/wpe-dereverb/infer", json=b).status_code == 400
