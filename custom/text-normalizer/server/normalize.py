"""Rule-based text normalisation: a placeholder with the contract a punctuation model would keep.

1. Unicode NFKC        2. collapse whitespace       3. capitalise sentence starts
4. add terminal punctuation when missing ("?" if the last sentence opens like a question, else ".")
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, ConfigDict

_WS = re.compile(r"\s+")
_SENTENCE_START = re.compile(r"(^|[.!?…]\s+)([a-z])")
_LAST_SENTENCE = re.compile(r"(?:^|[.!?…]\s+)([^.!?…]*)$")
_TERMINAL = ("." , "!", "?", "…", '"', "'", ")", "]", "”", "’")
_QUESTION_OPENERS = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "am", "was", "were", "do", "does", "did", "can", "could", "will", "would",
    "should", "shall", "may", "might", "must", "have", "has", "had",
}


class Options(BaseModel):
    """`options` accepted by POST /v1/text/transform."""

    model_config = ConfigDict(extra="forbid")
    terminal_punctuation: bool = True
    capitalize: bool = True


def normalize(text: str, opts: Options | None = None) -> str:
    opts = opts or Options()
    out = _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()
    if not out:
        return out
    if opts.capitalize:
        out = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), out)
    if opts.terminal_punctuation and not out.endswith(_TERMINAL) and out[-1].isalnum():
        last = _LAST_SENTENCE.search(out)
        first_word = (last.group(1) if last else out).split(" ", 1)[0].lower()
        out += "?" if first_word in _QUESTION_OPENERS else "."
    return out
