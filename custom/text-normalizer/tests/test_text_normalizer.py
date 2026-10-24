"""text-normalizer: text.transform conformance (in-process) and normalisation rules."""

from pathlib import Path

import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
norm = load_server(SERVICE_DIR, "normalize")
META = service_meta("text-normalizer")


@pytest.fixture(scope="module")
def client():
    with running(app_mod.build_app()) as c:
        yield c


@pytest.mark.parametrize("check", checks_for(META), ids=lambda c: c.__name__)
def test_conformance(client, check):
    check(client, META)


@pytest.mark.parametrize(("text", "expected"), [
    ("hello world how are you", "Hello world how are you."),
    ("how are you", "How are you?"),
    ("  hello   world.  how are   you ", "Hello world. How are you?"),
    ("ｈｅｌｌｏ　ｗｏｒｌｄ", "Hello world."),          # NFKC folds full-width forms and spaces
    ("already done!", "Already done!"),
    ("it costs 5", "It costs 5."),
    ("", ""),
    ("   ", ""),
    ("ends with emoji 🙂", "Ends with emoji 🙂"),       # only alphanumeric endings get punctuation
])
def test_rules(text, expected):
    assert norm.normalize(text) == expected


def test_options_disable_rules():
    opts = norm.Options(terminal_punctuation=False, capitalize=False)
    assert norm.normalize("hello  world", opts) == "hello world"


def test_options_over_http(client):
    r = client.post("/v1/text/transform", json={"model": "text-normalizer", "input": ["a b", "c"],
                                                "options": {"terminal_punctuation": False}})
    assert [d["output"] for d in r.json()["data"]] == ["A b", "C"]


def test_unknown_option_is_400(client):
    r = client.post("/v1/text/transform", json={"model": "text-normalizer", "input": "x", "options": {"nope": True}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options"


def test_wrong_model_is_404(client):
    r = client.post("/v1/text/transform", json={"model": "other", "input": "x"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
