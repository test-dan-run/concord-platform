"""Where model services run: Docker Compose on this host (DockerRuntime), or memory in tests (FakeRuntime).

The scheduler decides; the runtime only starts, stops and reports. Starting renders nothing by itself:
the catalog renders compose.yml whenever it reloads the manifests.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from tools import images, stack
from tools.schema import Manifest, Platform
from tools.vram_check import VRAM_VAR, measure_gpu, min_gib

log = logging.getLogger("catalog.runtime")


class StartError(Exception):
    pass


class Cancelled(Exception):
    """An export step stopped because the job was cancelled."""


class ImageError(Exception):
    """docker save / tag / push / load failed; the message is the CLI's error text."""


@dataclass
class ContainerInfo:
    name: str
    state: str  # running, exited, created, ...
    health: str  # healthy, unhealthy, starting, or "-" without a healthcheck
    gpu: bool
    image_id: str | None
    config_hash: str | None  # label aiplatform.config_hash: the config the container was started with


def budget_gib(s: Manifest, total_gib: float) -> float:
    """The least GPU memory a service loads in (gpu.util x the device, or gpu.vram_gb). At start it gets more:
    the whole GPU as measured free (tools/vram_check.py start_memory)."""
    return min_gib(s, total_gib)


class Runtime(Protocol):
    def containers(self, p: Platform) -> dict[str, ContainerInfo]: ...
    def image_id(self, ref: str) -> str | None: ...
    def start(self, p: Platform, s: Manifest, env: dict[str, str] | None = None) -> None: ...  # env: compose variables
    def stop(self, p: Platform, names: list[str]) -> None: ...
    def healthy(self, p: Platform, s: Manifest, url: str) -> bool: ...
    def failure_detail(self, p: Platform, name: str) -> str: ...
    def gpu_memory(self, p: Platform) -> tuple[float, float] | None: ...  # (total, free) GiB; None: cannot measure
    # Export (catalog/exporter.py): sizes, one `docker save` for all images, and registry pushes.
    def image_size(self, ref: str) -> int | None: ...
    def save_images(self, refs: list[str], path: Path, cancel: threading.Event, tick: Callable[[], None]) -> None: ...
    def push_image(self, ref: str, target: str, cancel: threading.Event) -> str | None: ...


def _wait(proc: subprocess.Popen, cancel: threading.Event, tick: Callable[[], None], what: str) -> None:
    while True:
        try:
            proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise Cancelled(what) from None
            tick()
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "") or ""
        raise ImageError(f"{what} exited {proc.returncode}: {err.strip()[-400:]}")


def docker_auth_env() -> dict[str, str]:
    """For a push from the container: a temporary DOCKER_CONFIG holding a copy of the host's login
    (CATALOG_DOCKER_AUTH, mounted read-only); the host process uses the user's own config."""
    src = os.environ.get("CATALOG_DOCKER_AUTH")
    env = dict(os.environ)
    if src and Path(src).is_file() and Path(src).stat().st_size:
        d = tempfile.mkdtemp(prefix="catalog-docker-")
        shutil.copyfile(src, Path(d) / "config.json")
        env["DOCKER_CONFIG"] = d
    return env


class DockerRuntime:
    def __init__(self) -> None:
        self._image_ids: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()
        self._gpu_warned = False

    def containers(self, p: Platform) -> dict[str, ContainerInfo]:
        ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=com.docker.compose.project={p.config.compose_project}"],
                             capture_output=True, text=True, check=True).stdout.split()
        if not ids:
            return {}
        out = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True, check=True).stdout
        result = {}
        for c in json.loads(out):
            labels = c["Config"]["Labels"] or {}
            name = labels.get("com.docker.compose.service") or labels.get("aiplatform.name")
            if not name:
                continue
            health = (c["State"].get("Health") or {}).get("Status", "-")
            result[name] = ContainerInfo(name, c["State"]["Status"], health, labels.get("aiplatform.gpu") == "true",
                                         c.get("Image"), labels.get("aiplatform.config_hash"))
        return result

    def image_id(self, ref: str) -> str | None:
        with self._lock:
            hit = self._image_ids.get(ref)
            if hit and time.monotonic() - hit[0] < 30:
                return hit[1]
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", ref], capture_output=True, text=True)
        value = r.stdout.strip() if r.returncode == 0 else None
        with self._lock:
            self._image_ids[ref] = (time.monotonic(), value)
        return value

    def start(self, p: Platform, s: Manifest, env: dict[str, str] | None = None) -> None:
        try:
            stack.ensure_network(p.config.network)
            stack.ensure_assets(p, [s])
            if s.is_custom and not images.image_exists(p.image(s)):
                images.build(p, s)
                with self._lock:
                    self._image_ids.pop(p.image(s), None)
            stack.compose(p, "up", "-d", "--no-deps", "--no-build", s.name, capture=True, env=env)
        except stack.StackError as exc:
            raise StartError(str(exc)) from None
        except subprocess.CalledProcessError as exc:
            raise StartError(f"docker compose up {s.name} exited {exc.returncode}: {(exc.stderr or '').strip()[-400:]}") from None

    def stop(self, p: Platform, names: list[str]) -> None:
        if names:
            stack.compose(p, "stop", *names, capture=True, check=False)

    def healthy(self, p: Platform, s: Manifest, url: str) -> bool:
        try:
            return httpx.get(url + p.health_path(s), timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def failure_detail(self, p: Platform, name: str) -> str:
        r = stack.compose(p, "logs", "--no-color", "--tail", "15", name, capture=True, check=False)
        lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
        return "\n".join(lines[-15:])

    def gpu_memory(self, p: Platform) -> tuple[float, float] | None:
        # Device-wide, so it includes processes outside the platform. The catalog container gets
        # nvidia-smi from the NVIDIA runtime (utility capability, catalog/compose.yml).
        gpu = measure_gpu()
        if gpu is None and not self._gpu_warned:
            self._gpu_warned = True
            log.warning("cannot read GPU memory with nvidia-smi: GPU models start with util_limit of the GPU")
        return gpu

    def image_size(self, ref: str) -> int | None:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Size}}", ref], capture_output=True, text=True)
        return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None

    def save_images(self, refs: list[str], path: Path, cancel: threading.Event, tick: Callable[[], None]) -> None:
        # One `docker save` for every image: each shared layer is written once.
        proc = subprocess.Popen(["docker", "save", "-o", str(path), *refs], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        _wait(proc, cancel, tick, "docker save")

    def push_image(self, ref: str, target: str, cancel: threading.Event) -> str | None:
        env = docker_auth_env()
        try:
            r = subprocess.run(["docker", "tag", ref, target], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                raise ImageError(f"docker tag exited {r.returncode}: {r.stderr.strip()[-400:]}")
            proc = subprocess.Popen(["docker", "push", target], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=env)
            _wait(proc, cancel, lambda: None, f"docker push {target}")
            out = proc.stdout.read() if proc.stdout else ""
        finally:
            if env.get("DOCKER_CONFIG") and env["DOCKER_CONFIG"] != os.environ.get("DOCKER_CONFIG"):
                shutil.rmtree(env["DOCKER_CONFIG"], ignore_errors=True)
        m = re.search(r"digest: (sha256:[0-9a-f]{64})", out)
        return m.group(1) if m else None


class FakeRuntime:
    """Containers in memory. `start_s` is how long a start takes; `fail` names services whose start fails.
    Set `gpu_total_gib` to measure GPU memory: a running GPU service takes what its start gave it (AISVC_VRAM_GB's
    compose variable; its minimum without one), `gpu_other_gib` is used by processes outside the platform.
    `envs` keeps the compose variables of each start."""

    def __init__(self, start_s: float = 0.0, fail: set[str] | None = None):
        self.start_s = start_s
        self.fail = set(fail or ())
        self.running: dict[str, float] = {}  # name -> healthy-at (monotonic)
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.images: dict[str, str] = {}
        self.layers: dict[str, list[tuple[str, int]]] = {}  # image ref -> (layer id, bytes): shared ids are saved once
        self.missing: set[str] = set()  # image refs that are not present locally
        self.pushed: list[tuple[str, str]] = []
        self.save_delay_s = 0.0
        self.gpu_total_gib: float | None = None
        self.gpu_other_gib = 0.0
        self.envs: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def containers(self, p: Platform) -> dict[str, ContainerInfo]:
        from tools.identity import config_hash

        known = {s.name: s for s in p.services}
        with self._lock:
            return {n: ContainerInfo(n, "running", "healthy" if time.monotonic() >= t else "starting",
                                     bool(known[n].gpu) if n in known else False, self.image_id(p.image(known[n])) if n in known else None,
                                     config_hash(p, known[n]) if n in known else None)
                    for n, t in self.running.items()}

    def image_id(self, ref: str) -> str | None:
        if ref in self.missing:
            return None
        return self.images.get(ref, "sha256:" + ref.encode().hex()[:24])

    def image_size(self, ref: str) -> int | None:
        if ref in self.missing:
            return None
        return sum(n for _, n in self.layers.get(ref, [("base", 1000)]))

    def save_images(self, refs: list[str], path: Path, cancel: threading.Event, tick: Callable[[], None]) -> None:
        unique = {lid: n for ref in refs for lid, n in self.layers.get(ref, [("base", 1000)])}
        deadline = time.monotonic() + self.save_delay_s
        with open(path, "wb") as fh:
            for lid, n in sorted(unique.items()):
                fh.write(lid.encode()[:n].ljust(n, b"\0"))
        while time.monotonic() < deadline:
            if cancel.wait(0.01):
                raise Cancelled("docker save")
            tick()
        if cancel.is_set():
            raise Cancelled("docker save")

    def push_image(self, ref: str, target: str, cancel: threading.Event) -> str | None:
        if cancel.is_set():
            raise Cancelled("docker push")
        self.pushed.append((ref, target))
        return "sha256:" + (ref + target).encode().hex()[:64].ljust(64, "0")

    def start(self, p: Platform, s: Manifest, env: dict[str, str] | None = None) -> None:
        if s.name in self.fail:
            raise StartError(f"{s.name} failed to start (fake)")
        with self._lock:
            self.running[s.name] = time.monotonic() + self.start_s
            self.started.append(s.name)
            self.envs[s.name] = dict(env or {})

    def stop(self, p: Platform, names: list[str]) -> None:
        with self._lock:
            for n in names:
                self.running.pop(n, None)
                self.stopped.append(n)

    def healthy(self, p: Platform, s: Manifest, url: str) -> bool:
        with self._lock:
            t = self.running.get(s.name)
        return t is not None and time.monotonic() >= t

    def failure_detail(self, p: Platform, name: str) -> str:
        return ""

    def gpu_memory(self, p: Platform) -> tuple[float, float] | None:
        if self.gpu_total_gib is None:
            return None
        known = {s.name: s for s in p.services}
        with self._lock:
            used = sum(float(self.envs.get(n, {}).get(VRAM_VAR) or budget_gib(known[n], self.gpu_total_gib))
                       for n in self.running if n in known and known[n].gpu)
        return self.gpu_total_gib, self.gpu_total_gib - used - self.gpu_other_gib
