"""render.write_outputs replaces each file in one step, so docker compose never reads a half-written compose.yml."""

import threading

from tools import render


def test_a_reader_never_sees_a_partial_file(monkeypatch, tmp_path):
    f = tmp_path / "compose.yml"
    texts = ("a" * 40336, "b" * 40000)  # about the size of the generated compose.yml
    f.write_text(texts[0])
    stop, torn, current = threading.Event(), [], [texts[0]]
    monkeypatch.setattr(render, "outputs", lambda p: {f: current[0]})

    def writer():
        while not stop.is_set():
            for text in texts:
                current[0] = text
                render.write_outputs(None)

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(3000):
            n = len(f.read_text())
            if n not in (len(texts[0]), len(texts[1])):
                torn.append(n)
    finally:
        stop.set()
        t.join()
    assert torn == []
    assert [p.name for p in tmp_path.iterdir()] == ["compose.yml"]  # no temp files left behind


def test_an_unchanged_file_is_not_rewritten(monkeypatch, tmp_path):
    f = tmp_path / "compose.yml"
    f.write_text("same")
    before = f.stat().st_ino, f.stat().st_mtime_ns
    monkeypatch.setattr(render, "outputs", lambda p: {f: "same"})
    assert render.write_outputs(None) == [f]
    assert (f.stat().st_ino, f.stat().st_mtime_ns) == before
