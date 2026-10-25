"""A tarball bundle that fails while it is written leaves the tarball that was there before, not a partial one."""

import pytest

from tools import bundle
from tools.schema import load_platform


def test_a_weights_file_that_cannot_be_read_keeps_the_old_tarball(tmp_path, monkeypatch):
    model = tmp_path / "ai-assets" / "whisper-large-v3"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    shard = model / "model.safetensors"
    shard.write_bytes(b"w" * 3000)
    monkeypatch.setenv("AI_ASSETS", str(tmp_path / "ai-assets"))
    monkeypatch.setattr(bundle, "docker_available", lambda: False)
    out = tmp_path / "dist"
    out.mkdir()
    old = out / "aiplatform-whisper-large-v3.tar"
    old.write_bytes(b"a good bundle from yesterday")
    p = load_platform()
    shard.chmod(0)
    try:
        with pytest.raises(PermissionError):
            bundle.make_bundle(p, p.resolve(["whisper-large-v3"]), out, False, with_weights=True, log=lambda m: None)
    finally:
        shard.chmod(0o644)
    assert old.read_bytes() == b"a good bundle from yesterday"
    assert sorted(f.name for f in out.iterdir()) == [old.name]  # no temporary folder or partial file left
