"""The static catalog's text.transform snippet passes no service-specific option: each service has its own."""

from tools import render
from tools.schema import load_platform


def test_usage_snippet_for_text_transform_passes_no_options():
    p = load_platform()
    for s in p.services:
        if s.api == "text.transform":
            snippet = render.usage_snippet(s)
            assert ".transform(" in snippet and "terminal_punctuation" not in snippet, s.name
            assert f"{s.build.context}/README.md" in snippet
            compile(snippet, "<usage snippet>", "exec")
    assert render.usage_snippet(p.get("monkey-custom")).count("=") == 1  # only `svc = client(...)`
