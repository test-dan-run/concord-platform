"""fetch_weights: a failed download leaves no .part file and prints an error line instead of a traceback."""

import contextlib

import httpx
import pytest

from tools import bundle, fetch_weights, identity

URL = "https://huggingface.co/org/repo/resolve/abc123/w.bin"


class Response:
    headers = {"content-length": "10"}

    def raise_for_status(self):
        pass

    def iter_bytes(self, n):
        yield b"hello"
        raise httpx.ReadTimeout("peer went away")


class Gated(Response):
    def raise_for_status(self):
        request = httpx.Request("GET", URL)
        raise httpx.HTTPStatusError("Client error '401 Unauthorized'", request=request,
                                    response=httpx.Response(401, request=request))


def serve(monkeypatch, response):
    @contextlib.contextmanager
    def stream(*args, **kwargs):
        yield response

    monkeypatch.setattr(fetch_weights.httpx, "stream", stream)


@pytest.fixture
def dest(monkeypatch, tmp_path):
    spec = {"version": "v1", "files": [{"path": "w.bin", "url": URL, "sha256": "0" * 64, "size": 10}]}
    monkeypatch.setattr(fetch_weights, "weights_spec", lambda p, name: (spec, tmp_path))
    return tmp_path


def test_a_stream_failing_mid_way_leaves_no_part_file(monkeypatch, tmp_path):
    serve(monkeypatch, Response())
    with pytest.raises(httpx.ReadTimeout):
        fetch_weights.download(URL, tmp_path / "w.bin", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_main_reports_a_failed_download_and_exits_1(dest, monkeypatch, capsys):
    serve(monkeypatch, Response())
    assert fetch_weights.main(["silero-vad"]) == 1
    assert "error: w.bin: peer went away" in capsys.readouterr().err
    assert list(dest.iterdir()) == []


def test_main_gives_the_hf_token_hint_for_a_gated_file(dest, monkeypatch, capsys):
    serve(monkeypatch, Gated())
    assert fetch_weights.main(["silero-vad"]) == 1
    err = capsys.readouterr().err
    assert "error: w.bin: Client error '401 Unauthorized'" in err and "HF_TOKEN" in err


def test_download_leftovers_are_not_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"w")
    for name in ("model.safetensors.part", "x.incomplete", "y.lock"):
        (tmp_path / name).write_bytes(b"partial")
    assert [rel for rel, _ in identity.weight_files(tmp_path)] == ["model.safetensors"]
    files, _, _ = bundle.collect(tmp_path, "w/")
    assert sorted(files) == ["w/model.safetensors"]
