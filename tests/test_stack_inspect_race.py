"""Listing containers survives one that is removed between `docker ps` and `docker inspect`."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from catalog.runtime import ContainerInfo, DockerRuntime
from tools import stack

P = SimpleNamespace(config=SimpleNamespace(compose_project="aiplatform"))
VAD = {"Id": "a1b2c3d4e5f6a7b8", "Image": "sha256:vad", "State": {"Status": "running", "Health": {"Status": "healthy"}},
       "Config": {"Labels": {"com.docker.compose.service": "silero-vad", "aiplatform.gpu": "false",
                             "aiplatform.config_hash": "h1"}}}


def fake_docker(monkeypatch, inspect: subprocess.CompletedProcess) -> None:
    """docker ps lists two IDs; docker inspect answers with `inspect`. Behaves like subprocess.run with check."""

    def run(cmd, check=False, **kw):
        r = subprocess.CompletedProcess(cmd, 0, "a1b2c3d4e5f6\nf6e5d4c3b2a1\n", "") if cmd[1] == "ps" else inspect
        if check and r.returncode:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
        return r

    monkeypatch.setattr(stack.subprocess, "run", run)


def test_a_container_removed_after_docker_ps_is_left_out(monkeypatch):
    # What docker 29 does for `docker inspect <found> <gone>`: the found one on stdout, exit 1.
    fake_docker(monkeypatch, subprocess.CompletedProcess([], 1, json.dumps([VAD]), "error: no such object: f6e5d4c3b2a1\n"))
    assert stack.containers(P) == {"silero-vad": stack.Container("a1b2c3d4e5f6", "silero-vad", "running", "healthy", False)}
    assert DockerRuntime().containers(P) == {
        "silero-vad": ContainerInfo("silero-vad", "running", "healthy", False, "sha256:vad", "h1")}


def test_every_container_removed_after_docker_ps_is_an_empty_list(monkeypatch):
    fake_docker(monkeypatch, subprocess.CompletedProcess([], 1, "[]\n", "error: no such object: a1b2c3d4e5f6\n"
                                                                       "error: no such object: f6e5d4c3b2a1\n"))
    assert stack.containers(P) == {} and DockerRuntime().containers(P) == {}


def test_other_inspect_failures_still_raise(monkeypatch):
    fake_docker(monkeypatch, subprocess.CompletedProcess([], 1, "", "Cannot connect to the Docker daemon\n"))
    with pytest.raises(subprocess.CalledProcessError):
        stack.containers(P)
    with pytest.raises(subprocess.CalledProcessError):
        DockerRuntime().containers(P)
