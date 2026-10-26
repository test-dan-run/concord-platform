"""A bundle download holds a writer thread and a pipe only while it is read, and says why a stream stopped."""

import gc
import logging
import os
import threading
import time

import pytest
from export_helpers import PAIR, api, finish, images  # noqa: F401  (fixtures)

from catalog import exporter as exporter_mod


@pytest.fixture
def job(api, cat, images, settings):
    job = finish(cat, api, api.post("/api/exports", json={"models": PAIR}).json()["id"])
    assert job["state"] == "done", job
    (settings.export_dir / job["name"] / "big.bin").write_bytes(b"x" * (4 << 20))  # more than a pipe holds
    return job


def writers():
    return [t.name for t in threading.enumerate() if t.name.startswith("download-")]


def settle(fds0):
    deadline = time.monotonic() + 5
    while (writers() or len(os.listdir("/proc/self/fd")) > fds0) and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.02)
    return writers(), len(os.listdir("/proc/self/fd"))


def test_a_download_that_never_starts_holds_no_thread_or_pipe(cat, job):
    fds0 = len(os.listdir("/proc/self/fd"))
    name, chunks = cat.exporter.download(job["id"])
    assert name == f"{job['name']}.tar"
    del chunks  # the client went away before the response started
    assert settle(fds0) == ([], fds0)


def test_a_download_left_half_read_stops_its_writer(cat, job):
    fds0 = len(os.listdir("/proc/self/fd"))
    _, chunks = cat.exporter.download(job["id"])
    assert next(chunks)
    assert writers()
    chunks.close()  # the client went away mid-stream
    assert settle(fds0) == ([], fds0)


def test_a_writer_error_is_logged(cat, job, monkeypatch, caplog):
    def gone(self, *a, **kw):
        raise FileNotFoundError(2, "No such file or directory", "big.bin")

    monkeypatch.setattr(exporter_mod.tarfile.TarFile, "add", gone)
    with caplog.at_level(logging.WARNING, logger="catalog.export"):
        _, chunks = cat.exporter.download(job["id"])
        assert b"".join(chunks) == b""
        deadline = time.monotonic() + 5
        while writers() and time.monotonic() < deadline:
            time.sleep(0.02)
    assert f"download of export {job['id']} stopped" in caplog.text and "No such file or directory" in caplog.text
