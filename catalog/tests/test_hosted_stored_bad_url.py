"""An endpoint saved with a base_url the catalog now refuses (before 422 checks existed) still lists: its host falls
back to the URL as stored instead of failing every GET /api/models."""

from catalog.hosted import Hosted


def test_the_host_of_an_unparseable_stored_url_is_the_url():
    h = Hosted(id="old", contract="openai", api="audio.transcriptions", base_url="https://[api.example.com/v1",
               model="whisper-1", auth_type="none", header_name=None, key_enc=None, key_hint=None, timeout_s=60.0,
               max_concurrency=4, max_upload_mb=25.0, signature=None, data_ack_at="", created_at="")
    assert h.host == "https://[api.example.com/v1"
    assert h.public()["host"] == "https://[api.example.com/v1"


def test_the_host_of_a_good_url_is_its_hostname():
    h = Hosted(id="new", contract="openai", api="audio.transcriptions", base_url="https://api.example.com:8443/v1",
               model="whisper-1", auth_type="none", header_name=None, key_enc=None, key_hint=None, timeout_s=60.0,
               max_concurrency=4, max_upload_mb=25.0, signature=None, data_ack_at="", created_at="")
    assert h.host == "api.example.com"
