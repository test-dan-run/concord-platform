"""files.csv durations are finite numbers of 0 or more and sample rates whole numbers above 0, written plainly: a
NaN, an infinity, a negative duration, a rate of 0 or a number with "_" in it is a problem, so it can't reach the
database and break the dataset later. FakeDatasetCatalog checks with the same check_folder."""

import io
import json
import zipfile

import pytest
from catalog_helpers import new_dataset, publish, sha, upload, wav
from fastapi.testclient import TestClient

from catalog.app import create_app

A, B = wav(1), wav(2)


@pytest.fixture
def api(cat):
    app = create_app(cat, start=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        upload(c, A)
        upload(c, B)
        yield c
    app.state.datasets.db.close()


def _manifest(n: int) -> dict:
    return {"format": "concord-reference-package", "format_version": 3, "name": "Calls",
            "created_at": "2026-10-24T09:00:00Z", "label_types": [], "counts": {"files": n, "segments": 0},
            "audio_included": "none"}


def audio_only_zip(duration: str, sample_rate: str = "16000") -> bytes:
    """A version with audio only whose files.csv gives a.wav that duration and sample rate."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("manifest.json", json.dumps(_manifest(1)))
        z.writestr("files.csv", f"path,sha256,duration,sample_rate\na.wav,{sha(A)},{duration},{sample_rate}\n")
    return buf.getvalue()


BAD = [("nan", "16000"), ("-1", "16000"), ("1_0", "16000"), ("0.5", "0"), ("0.5", "16_000")]


@pytest.mark.parametrize(("duration", "sample_rate"), BAD)
def test_a_bad_number_in_files_csv_is_422(api, duration, sample_rate):
    did = new_dataset(api)
    r = publish(api, did, audio_only_zip(duration, sample_rate))
    assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_version", r.text
    field, value = ("duration", duration) if sample_rate == "16000" else ("sample_rate", sample_rate)
    assert r.json()["error"]["problems"] == [f"files.csv: {field} of a.wav is {value!r}"]
    assert api.get(f"/api/datasets/{did}").json()["versions"] == []


def test_a_duration_of_0_and_plain_numbers_stay_valid(api):
    """0 stays valid: the fake writes 0.0 for audio it can't probe."""
    for duration, sample_rate in (("0", "16000"), ("0.0", "8000"), ("1e3", "44100"), ("12.5", "48000")):
        did = new_dataset(api)
        assert publish(api, did, audio_only_zip(duration, sample_rate)).status_code == 201
        [f] = api.get(f"/api/datasets/{did}/versions/1/files").json()["files"]
        assert (f["duration"], f["sample_rate"]) == (float(duration), int(sample_rate))

