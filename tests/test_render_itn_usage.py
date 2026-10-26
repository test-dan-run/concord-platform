"""The static catalog's usage snippet for a text.itn service calls .itn(), the method aicore has for it."""

from tools import render
from tools.schema import load_platform


def test_usage_snippet_for_a_text_itn_service_calls_itn():
    itn = load_platform().get("text-normalizer").model_copy(update={"api": "text.itn"})
    snippet = render.usage_snippet(itn)
    assert ".itn(" in snippet and ".transform(" not in snippet
    compile(snippet, "<usage snippet>", "exec")
