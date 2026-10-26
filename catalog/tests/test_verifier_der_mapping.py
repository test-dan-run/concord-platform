"""DER maps speakers one to one however many there are: one hypothesis speaker never covers two reference speakers."""

import pytest

from catalog.verifier import metrics

T = metrics.Turn


def test_extra_tiny_speakers_do_not_hide_a_missed_speaker_change():
    ref = [T(0, 2, "a"), T(2, 4, "b")]
    # 8 hypothesis speakers: past the old brute-force cap of 7, where the missed change used to vanish (DER 0.02)
    hyp = [T(0, 4, "x")] + [T(1.0 + i * 0.05, 1.01 + i * 0.05, f"t{i}") for i in range(7)]
    assert metrics.der(ref, hyp, 4) == pytest.approx(0.5, abs=0.05)


def test_many_speakers_map_one_to_one():
    ref = [T(i, i + 1, f"r{i}") for i in range(9)]  # 9 x 9 speakers: too many mappings to try them all
    assert metrics.der(ref, [T(i, i + 1, f"h{i}") for i in range(9)], 9) == 0.0
    merged = [T(0, 2, "x")] + [T(i, i + 1, f"h{i}") for i in range(2, 9)] + [T(0.5, 0.52, "y")]
    assert metrics.der(ref, merged, 9) == pytest.approx(0.5 / 4.5, abs=0.01)  # r0 or r1 is confused: 0.5 s scored
