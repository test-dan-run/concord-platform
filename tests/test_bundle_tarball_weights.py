"""A tarball with weights takes them from where they are: no staged copy, real files, root-owned."""

import tarfile

from tools import bundle
from tools.schema import load_platform


def test_tarball_weights_are_streamed_as_real_files(tmp_path, monkeypatch):
    model = tmp_path / "ai-assets" / "whisper-large-v3"
    (model / "sub").mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"w" * 3000)
    (model / "sub" / "config.json").write_text("{}")
    (model / "tokenizer.json").symlink_to(model / "sub" / "config.json")
    monkeypatch.setenv("AI_ASSETS", str(tmp_path / "ai-assets"))
    monkeypatch.setattr(bundle, "docker_available", lambda: False)

    def no_copy(*args, **kw):
        raise AssertionError("the weights were staged on disk before going into the tarball")

    monkeypatch.setattr(bundle, "copy_weights", no_copy)
    p = load_platform()
    target = bundle.make_bundle(p, p.resolve(["whisper-large-v3"]), tmp_path / "dist", with_images=False,
                                with_weights=True, log=lambda m: None)
    root = "aiplatform-whisper-large-v3/assets/whisper-large-v3"
    with tarfile.open(target) as tar:
        weights = {m.name.removeprefix(root + "/"): m for m in tar.getmembers() if m.name.startswith(root + "/")}
        assert sorted(weights) == ["model.safetensors", "sub/config.json", "tokenizer.json"]
        assert all(m.isreg() and (m.uid, m.gid, m.uname, m.gname, m.mode) == (0, 0, "", "", 0o644)
                   for m in weights.values())
        assert tar.extractfile(weights["tokenizer.json"]).read() == b"{}"
        assert tar.extractfile(weights["model.safetensors"]).read() == b"w" * 3000
