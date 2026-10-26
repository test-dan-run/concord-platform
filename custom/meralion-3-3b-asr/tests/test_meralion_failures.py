"""meralion-3-3b-asr: a file that fails in a shared batch fails only its own request, and an out of memory error on
one file never lowers the limits of the batcher that groups requests."""

import io
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from conformance import load_server, running

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
asr_mod = load_server(SERVICE_DIR, "meralion")


class FailingMeralion(asr_mod.Meralion3Asr):
    """Fails any batch that holds a chunk of `bad` samples; names each chunk's length otherwise."""

    def __init__(self, error: Exception):
        super().__init__("/nonexistent")
        self.error, self.bad = error, 16000 * 3

    def load(self):
        self.model = object()

    def transcribe_batch(self, pieces):
        if any(len(x) == self.bad for x in pieces):
            raise self.error
        return [f"chunk{len(x)}" for x in pieces]


def wav(seconds: float) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.full(int(seconds * 16000), 0.1, np.float32), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def post_together(client, payloads):
    out, start = [None] * len(payloads), threading.Barrier(len(payloads))

    def call(i):
        start.wait()
        out[i] = client.post("/v1/audio/transcriptions", data={"model": "meralion-3-3b-asr"},
                             files={"file": ("a.wav", payloads[i], "audio/wav")})

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(payloads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out


def test_one_failing_file_fails_only_its_own_request(monkeypatch):
    error = MemoryError("CUDA out of memory")  # a RuntimeError takes the same rerun path (see the rerun_memory tests)
    pytest.importorskip("soundfile")
    monkeypatch.setitem(sys.modules, "torch", None)  # is_oom then goes by the error alone
    monkeypatch.setenv("AISVC_BATCH_WAIT_MS", "500")
    model = FailingMeralion(error)
    with running(app_mod.build_app(model)) as c:
        rs = post_together(c, [wav(1.0), wav(2.0), wav(3.0)])
        assert [r.status_code for r in rs] == [200, 200, 500]
        assert [r.json()["text"] for r in rs[:2]] == ["chunk16000", "chunk32000"]
        assert type(error).__name__ in rs[2].json()["error"]["message"]
        assert c.get("/platform/memory").json()["batch"]["requests"]["lowered"] is False
