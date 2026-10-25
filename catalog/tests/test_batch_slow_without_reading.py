"""A batch that takes longer than SLOW_S ends its length's sweep, also when it gave no memory reading."""

from __future__ import annotations

import time

from catalog import batching


def test_a_slow_batch_with_no_reading_ends_the_length(monkeypatch):
    monkeypatch.setattr(batching, "SLOW_S", 0.05)

    class NoReading:
        def measure(self, fn):
            fn()
            return None  # a device reading under an earlier high, with no torch counters

    sent: list[int] = []

    def send(n, length):
        sent.append(n)
        time.sleep(0.1)

    pts, stops, _ = batching.sweep(send, NoReading(), [15.0], budget_gib=None, max_n=128)
    assert sent == [1] and pts == [] and stops[0]["stopped"].endswith("s for a batch of 1")
