"""monkey-custom: text.transform conformance and transcript-normer's documented behaviour, in process with the
real normaliser; punctuated mode with the real ONNX model when its weights are present."""

import os
from pathlib import Path

import pytest
from conformance import checks_for, load_server, running, service_meta

SERVICE_DIR = Path(__file__).resolve().parents[1]
app_mod = load_server(SERVICE_DIR, "app")
META = service_meta("monkey-custom")


class NoPunctuation(app_mod.Engine):
    """The real normaliser without the 210 MB punctuation model: canonical and verbatim only."""

    def __init__(self):
        super().__init__("/nonexistent")

    def load(self):
        from importlib import import_module

        normer = import_module(app_mod.__package__ + ".normer.normalizer")
        self.normalizer = normer.TranscriptNormalizer(punctuator=None)


@pytest.fixture(scope="module")
def client():
    with running(app_mod.build_app(NoPunctuation())) as c:
        yield c


def test_conformance(client):
    for check in checks_for(META):
        check(client, META)


def transform(c, input, **options):
    return c.post("/v1/text/transform", json={"model": "monkey-custom", "input": input, "options": options})


def outputs(r):
    assert r.status_code == 200, r.text
    return [d["output"] for d in r.json()["data"]]


@pytest.mark.parametrize("text,mode,want", [
    ("I went to the U S A last year lah, spent twenty dollars you know.", "canonical", "i went to the usa last year spent $20"),
    ("Give me five no wait ten apples lah", "verbatim", "Give me five no wait ten apples"),  # self-correction kept
    ("give me five no wait ten apples lah", "canonical", "give me 5 no wait 10 apples"),
    ("I I think so", "verbatim", "I I think so"),  # a false start, not an acronym
    ("flight to LA lah", "verbatim", "flight to LA"),
    ("the U.S.A. is big", "verbatim", "the USA is big"),
])
def test_documented_examples(client, text, mode, want):
    assert outputs(transform(client, text, mode=mode)) == [want]


def test_filler_and_particle_switches(client):
    assert outputs(transform(client, ["um okay lah"], mode="verbatim", remove_fillers=False)) == ["um okay"]
    assert outputs(transform(client, ["um okay lah"], mode="verbatim", remove_particles=False)) == ["okay lah"]


def test_bad_options_and_limits_are_400(client):
    r = transform(client, "x", mode="shouty")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options"
    r = transform(client, "x", lowercase=True)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_options"
    r = transform(client, ["x"] * 1001)
    assert r.status_code == 400 and "at most 1000" in r.json()["error"]["message"]


def test_missing_weights_keep_the_service_unready(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-weights monkey-custom"):
        app_mod.Engine(tmp_path).load()


WEIGHTS = Path(os.environ.get("MONKEY_CUSTOM_WEIGHTS", Path(os.path.expanduser(os.environ.get("AI_ASSETS", "~/ai-assets")))
                              / "custom-models/monkey-custom/b26fd1c"))


@pytest.mark.weights
@pytest.mark.skipif(not (WEIGHTS / "punct_cap_seg_en.onnx").is_file(), reason=f"no punctuation model in {WEIGHTS}")
def test_punctuated_mode_with_the_real_model():
    with running(app_mod.build_app(app_mod.Engine(WEIGHTS))) as c:
        got = outputs(transform(c, ["um okay lah see you tomorrow how are you doing",
                                    "give me five no wait ten apples lah"], mode="punctuated"))
    assert got[0] == "Okay, see you tomorrow, how are you doing?"
    assert [w.strip(",.?").lower() for w in got[1].split()] == "give me five no wait ten apples".split()  # wording kept
