import importlib
import sys

import numpy as np
import openai
import pytest
from aisvc import create_oip_app
from conformance import load_server, running
from factories import toy_itn_app

import aicore
from aicore import AicoreError, TextServiceClient, UnknownServiceError, client, endpoint, model_name
from tools import render
from tools.schema import PLATFORM_DIR, load_platform


@pytest.fixture(autouse=True)
def endpoints_file(tmp_path, monkeypatch):
    """Point aicore at a freshly rendered map so tests do not depend on `just render`."""
    f = tmp_path / "endpoints.yaml"
    f.write_text(render.dump_yaml(render.build_endpoints(load_platform())))
    monkeypatch.setenv("AICORE_ENDPOINTS", str(f))
    monkeypatch.delenv("AI_HOST", raising=False)
    monkeypatch.delenv("AICORE_CATALOG", raising=False)
    return f


def test_resolves_every_name_and_alias():
    p = load_platform()
    for s in p.services:
        for key in filter(None, (s.name, s.alias)):
            ep = endpoint(key)
            assert ep["name"] == s.name and model_name(key) == s.name
            assert ep["url"] == f"http://localhost:{s.port}"
    assert endpoint("default-vad")["base_url"] == "http://localhost:8401/v2"
    assert endpoint("default-chat")["base_url"] == "http://localhost:8201/v1"


def test_client_type_per_contract():
    chat = client("default-chat")
    assert isinstance(chat, openai.OpenAI) and str(chat.base_url).rstrip("/") == "http://localhost:8201/v1"
    assert isinstance(client("default-normalizer"), TextServiceClient)
    assert isinstance(client("default-vad"), aicore.OIPClient)
    assert client("default-vad").signature["outputs"][0]["name"] == "SEGMENTS"


def test_ai_host_controls_host_and_port(monkeypatch):
    monkeypatch.setenv("AI_HOST", "127.0.0.1")
    assert endpoint("default-chat")["url"] == "http://127.0.0.1:8201"
    monkeypatch.setenv("AI_HOST", "{name}")  # app container on ai-net: service name + internal port
    assert endpoint("default-chat")["url"] == "http://qwen3-8b:8000"
    assert endpoint("default-tts")["url"] == "http://kokoro-tts:8000"
    monkeypatch.setenv("AI_HOST", "gpu-box")
    assert endpoint("default-asr")["url"] == "http://gpu-box:8000"
    assert endpoint("default-asr", host="localhost")["url"] == "http://localhost:8101"


@pytest.mark.parametrize(("host", "url"), [
    ("::1", "http://[::1]:8201"),  # local: the published port
    ("[::1]", "http://[::1]:8201"),  # already in brackets, still local
    ("fd00::5", "http://[fd00::5]:8000"),  # anything else: the internal port
    ("[fd00::5]", "http://[fd00::5]:8000"),
])
def test_resolve_base_brackets_ipv6_literals(host, url):
    """An IPv6 AI_HOST goes in brackets, so the port stays a port; [::1] is the local host, like ::1."""
    entry = {"name": "qwen3-8b", "host_port": 8201, "internal_port": 8000}
    assert aicore.endpoints.resolve_base(entry, host) == url


def test_catalog_routes_every_contract_through_the_gateway(monkeypatch):
    monkeypatch.setenv("AICORE_CATALOG", "http://127.0.0.1:8700/")
    ep = endpoint("default-asr")
    assert ep["url"] == "http://127.0.0.1:8700/m/whisper-large-v3" and ep["via"] == "catalog"
    assert str(client("default-chat").base_url).rstrip("/") == "http://127.0.0.1:8700/m/qwen3-8b/v1"
    assert client("default-vad").base_url == "http://127.0.0.1:8700/m/silero-vad"
    assert endpoint("default-asr", host="localhost")["url"] == "http://localhost:8101"  # explicit host: direct


def test_fake_catalog_leases_and_index():
    from aicore.catalog import FakeCatalog, index_key, params_hash

    cat = FakeCatalog([{"id": "silero-vad", "build_key": "b1", "capabilities": {"tasks": ["vad"]}, "tags": ["x"]}])
    assert [m["id"] for m in cat.models(task="vad", tag="x")] == ["silero-vad"] and cat.models(task="asr") == []
    with cat.lease("bench:job-1", ["silero-vad"]) as lease:
        assert lease.active
    assert cat.leases() == []
    key = index_key("b1", "f" * 64, None, None, params_hash({"adapter": "OIPSegmenter"}))
    assert key["span_start"] == key["span_end"] == -1.0
    cat.register_app("bench", "http://bench")
    cat.index_record("bench", [{**key, "output_id": 7}])
    assert cat.index_lookup("label", [key, {**key, "params_hash": "0" * 16}]) == [
        {"owner": "bench", "output_id": 7, "owner_url": "http://bench"}, None]
    plan = cat.plan([{"name": "seg", "models": ["silero-vad"], "keys": {"silero-vad": [key]}}])
    assert plan["steps"][0]["skip"] and plan["loads"] == 0


def test_unknown_name_lists_available():
    with pytest.raises(UnknownServiceError) as exc:
        client("nope")
    msg = str(exc.value)
    assert "unknown service 'nope'" in msg and "default-chat" in msg and "silero-vad" in msg
    assert "Hosted models are only reachable through the catalog" in msg  # no AICORE_CATALOG set


@pytest.fixture
def fake_catalog(monkeypatch):
    """A catalog that knows one hosted ASR model, `hosted-asr`, served by the provider as `whisper-1`."""
    import httpx

    from aicore import endpoints as eps

    calls = []
    record = {"id": "hosted-asr", "served_name": "whisper-1", "kind": "hosted", "contract": "openai",
              "api": "audio.transcriptions", "engine": "hosted", "tasks": [], "signature": None,
              "description": "ASR served by api.example.com", "hosted": {"model": "whisper-1"}}

    def get(url, timeout=None):
        calls.append(url)
        if url.endswith("/api/models/hosted-asr"):
            return httpx.Response(200, json=record)
        return httpx.Response(404, json={"error": {"message": "unknown"}})

    monkeypatch.setattr(httpx, "get", get)
    eps._remote.clear()
    return calls


def test_hosted_models_resolve_through_the_catalog(monkeypatch, fake_catalog):
    monkeypatch.setenv("AICORE_CATALOG", "http://127.0.0.1:8700/")
    ep = endpoint("hosted-asr")
    assert ep["name"] == "whisper-1" and ep["id"] == "hosted-asr" and ep["via"] == "catalog"
    assert ep["url"] == "http://127.0.0.1:8700/m/hosted-asr" and ep["base_url"] == "http://127.0.0.1:8700/m/hosted-asr/v1"
    assert model_name("hosted-asr") == "whisper-1"
    c = client("hosted-asr")
    assert isinstance(c, openai.OpenAI) and str(c.base_url).rstrip("/") == "http://127.0.0.1:8700/m/hosted-asr/v1"
    assert len(fake_catalog) == 1  # cached for 60 s
    assert model_name("whisper-large-v3") == "whisper-large-v3" and len(fake_catalog) == 1  # local names never ask
    with pytest.raises(UnknownServiceError, match="does not know it either"):
        endpoint("nope")


def test_missing_endpoints_file_is_explained(monkeypatch, tmp_path):
    monkeypatch.setenv("AICORE_ENDPOINTS", str(tmp_path / "missing.yaml"))
    with pytest.raises(AicoreError, match="just render"):
        endpoint("default-chat")


# ----------------------------------------------------------------------------- OIP round trip

STUB_SIG = {
    "inputs": [
        {"name": "X", "datatype": "FP32", "shape": [-1, 3]},
        {"name": "K", "datatype": "INT64", "shape": [1]},
        {"name": "TAGS", "datatype": "BYTES", "shape": [-1]},
        {"name": "MASK", "datatype": "BOOL", "shape": [-1]},
    ],
    "outputs": [
        {"name": "Y", "datatype": "FP64", "shape": [-1, 3]},
        {"name": "N", "datatype": "INT32", "shape": [1]},
        {"name": "UPPER", "datatype": "BYTES", "shape": [-1]},
        {"name": "NOT_MASK", "datatype": "BOOL", "shape": [-1]},
    ],
}


def stub_handler(inputs, params):
    x, k = inputs["X"], inputs["K"]
    assert x.dtype == np.float32 and x.ndim == 2 and k.dtype == np.int64
    return {"Y": x.astype(np.float64) * k[0] + params.get("offset", 0), "N": np.array([x.shape[0]]),
            "UPPER": np.array([t.upper() for t in inputs["TAGS"]], dtype=object), "NOT_MASK": ~inputs["MASK"]}


@pytest.fixture(scope="module")
def oip_stub():
    with running(create_oip_app("stub", STUB_SIG, stub_handler)) as http:
        yield aicore.OIPClient("http://testserver", "stub", STUB_SIG, http=http)


def test_oip_round_trip(oip_stub):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = oip_stub.infer({"X": x, "K": np.array([3], dtype=np.int64), "TAGS": np.array(["a", "bc"]),
                          "MASK": np.array([True, False])}, parameters={"offset": 1})
    assert out["Y"].dtype == np.float64 and out["Y"].shape == (2, 3)
    np.testing.assert_allclose(out["Y"], x * 3 + 1)
    assert out["N"].dtype == np.int32 and out["N"].tolist() == [2]
    assert out["UPPER"].tolist() == ["A", "BC"]
    assert out["NOT_MASK"].dtype == np.bool_ and out["NOT_MASK"].tolist() == [False, True]


def test_oip_metadata_and_ready(oip_stub):
    assert oip_stub.ready() and oip_stub.live()
    meta = oip_stub.metadata()
    assert meta["name"] == "stub" and [t["name"] for t in meta["inputs"]] == ["X", "K", "TAGS", "MASK"]


def test_oip_requested_outputs_subset(oip_stub):
    out = oip_stub.infer({"X": np.zeros((1, 3), np.float32), "K": np.array([1]), "TAGS": np.array(["x"]),
                          "MASK": np.array([True])}, outputs=["N"])
    assert list(out) == ["N"]


def test_oip_client_validates_before_sending():
    class NoHTTP:
        def __getattr__(self, name):
            raise AssertionError("must not send a request")

    c = aicore.OIPClient("http://x", "stub", STUB_SIG, http=NoHTTP())
    good = {"X": np.zeros((1, 3), np.float32), "K": np.array([1]), "TAGS": np.array(["x"]), "MASK": np.array([True])}
    with pytest.raises(AicoreError, match="dtype float64 .* needs FP32"):
        c.encode(good | {"X": np.zeros((1, 3))})
    with pytest.raises(AicoreError, match="shape \\[1, 4\\] does not match"):
        c.encode(good | {"X": np.zeros((1, 4), np.float32)})
    with pytest.raises(AicoreError, match="missing \\['MASK'\\]"):
        c.encode({k: v for k, v in good.items() if k != "MASK"})


ALIGN_SIG = {"inputs": [{"name": "AUDIO", "datatype": "FP32", "shape": [-1, -1]},
                        {"name": "LENGTHS", "datatype": "INT32", "shape": [-1]},
                        {"name": "TEXT", "datatype": "BYTES", "shape": [-1]}],
             "outputs": [{"name": "WORDS", "datatype": "BYTES", "shape": [-1]}]}


def test_oip_client_sends_one_clip_to_a_batch_signature_as_it_is():
    """Callers written for a single-clip signature keep working when it becomes a padded batch (AUDIO [-1, -1] +
    LENGTHS): AUDIO [-1] without LENGTHS goes to the server unchanged, which answers it as one row."""
    c = aicore.OIPClient("http://x", "m", ALIGN_SIG, http=object())
    one = c.encode({"AUDIO": np.zeros(5, np.float32), "TEXT": np.array(["hi"])})
    assert [(t["name"], t["shape"]) for t in one] == [("AUDIO", [5]), ("TEXT", [1])]
    batch = c.encode({"AUDIO": np.zeros((2, 5), np.float32), "LENGTHS": np.array([5, 3], np.int32),
                      "TEXT": np.array(["a", "b"])})
    assert [t["shape"] for t in batch] == [[2, 5], [2], [2]]
    with pytest.raises(AicoreError, match="missing \\['LENGTHS'\\]"):  # a 2-D batch still needs its lengths
        c.encode({"AUDIO": np.zeros((2, 5), np.float32), "TEXT": np.array(["a", "b"])})
    with pytest.raises(AicoreError, match="shape \\[5\\] does not match"):  # with LENGTHS, AUDIO must be 2-D
        c.encode({"AUDIO": np.zeros(5, np.float32), "LENGTHS": np.array([5], np.int32), "TEXT": np.array(["a"])})


def test_oip_client_without_numpy_has_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.delitem(sys.modules, "aicore.oip", raising=False)
    with pytest.raises(ImportError, match=r"aicore\[oip\]"):
        importlib.import_module("aicore.oip")
    with pytest.raises(ImportError, match=r"aicore\[oip\]"):
        from aicore import OIPClient  # noqa: F401


# ----------------------------------------------------------------------------- text services

def test_text_service_client_against_normalizer():
    app = load_server(PLATFORM_DIR / "custom" / "text-normalizer", "app").build_app()
    with running(app) as http:
        c = client("default-normalizer", http=http)
        assert c.transform("how are you") == "How are you?"
        assert c.transform(["a b", "c"], terminal_punctuation=False) == ["A b", "C"]
        assert c.models() == ["text-normalizer"]
        with pytest.raises(aicore.ServiceError) as exc:
            c.transform("x", bogus=1)
        assert exc.value.status == 400 and exc.value.code == "invalid_options"


def test_text_service_client_itn():
    with running(toy_itn_app()) as http:
        c = TextServiceClient("http://test", "itn-toy", "text.itn", http=http)
        assert c.itn("meet at three") == "meet at 3"
        assert c.itn(["one", "two"], language="en-SG") == ["1", "2"]
        item = c.itn("twelve", classes=["CARDINAL"], return_spans=True)
        assert item["output"] == "12" and item["spans"][0]["class"] == "CARDINAL"
        with pytest.raises(aicore.ServiceError) as exc:
            c.itn("x", language="fr-FR")
        assert exc.value.code == "invalid_options"
        with pytest.raises(AicoreError, match="not text.transform"):
            c.transform("x")


def test_batch_size_from_the_catalog_or_the_default(monkeypatch):
    from aicore.batching import batch_size
    from aicore.catalog import FakeCatalog

    cat = FakeCatalog([{"id": "wavlm-sv", "batch": {"recommendation": {"batch": 66}}}, {"id": "other"}])
    assert batch_size("wavlm-sv", item_s=20, catalog=cat) == 66
    assert batch_size("other", catalog=cat, default=4) == 4
    assert cat.calls[0] == ("batch", ("wavlm-sv", None, 20, None, None))
    monkeypatch.delenv("AICORE_CATALOG", raising=False)
    assert batch_size("wavlm-sv") == 8  # no catalog: the default


def test_map_batches_and_map_concurrent_keep_order():
    import threading
    import time

    from aicore.batching import map_batches, map_concurrent

    sizes = []
    assert map_batches(lambda xs: (sizes.append(len(xs)), [x * 2 for x in xs])[1], list(range(7)), 3) == \
        [0, 2, 4, 6, 8, 10, 12]
    assert sizes == [3, 3, 1]
    with pytest.raises(ValueError):
        map_batches(lambda xs: xs[:1], [1, 2], 2)
    active, peak = 0, 0
    lock = threading.Lock()

    def slow(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return -x

    assert map_concurrent(slow, list(range(12)), 4) == [-x for x in range(12)] and peak == 4
