"""wait_healthy without a terminal prints a progress line about every 30 s, whatever the polling cadence."""

from tools import stack


class Clock:
    """Stands in for the time module in stack: sleep moves the clock, and so does every docker poll."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_progress_is_printed_every_30_s_without_a_tty(monkeypatch, capsys):
    clock = Clock()

    def containers(p):
        clock.now += 0.15  # docker ps + docker inspect
        health = "healthy" if clock.now > 300 else "starting"
        return {"svc": stack.Container("id", "svc", "running", health, True)}

    monkeypatch.setattr(stack, "time", clock)
    monkeypatch.setattr(stack, "containers", containers)
    stack.wait_healthy(None, "svc", timeout=900)
    lines = capsys.readouterr().out.splitlines()
    progress = [line for line in lines if line.startswith("  svc: running/starting ")]
    assert len(progress) >= 9, lines
    assert progress[0].endswith("0m00s (timeout 15m)") and lines[-1].startswith("  svc: healthy after ")
