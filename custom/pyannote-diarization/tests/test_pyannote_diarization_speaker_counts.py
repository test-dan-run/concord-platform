"""pyannote-diarization: the speaker counts must be real integers; JSON true is not 1."""

from pathlib import Path

import pytest
from conformance import load_server

SERVICE_DIR = Path(__file__).resolve().parents[1]
diar_mod = load_server(SERVICE_DIR, "diarizer")


@pytest.mark.parametrize("key", ["num_speakers", "min_speakers", "max_speakers"])
@pytest.mark.parametrize("value", [True, False, 2.0, "2"])
def test_a_speaker_count_that_is_not_an_int_is_refused(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        diar_mod.DiarParams.from_parameters({key: value})


def test_whole_numbers_still_pass():
    p = diar_mod.DiarParams.from_parameters({"num_speakers": 2, "min_speakers": 1, "max_speakers": 3})
    assert (p.num_speakers, p.min_speakers, p.max_speakers) == (2, 1, 3)
