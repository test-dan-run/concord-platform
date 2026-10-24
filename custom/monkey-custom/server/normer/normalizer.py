# Vendored from github.com/test-dan-run/transcript-normer @e98d4ae (app/services/normalizer.py).
# Changed for the platform: package-relative imports only. Update by re-copying at a pinned commit.
"""Core ASR transcript normalization pipeline.

Stages run in a fixed order:

1. Disfluency cleanup: strip verbal fillers and Singlish discourse particles.
2. Acronym stitching: "U S A" / "U.S.A." -> "USA".
3. Output formatting: ``canonical`` hands the text to whisper-normalizer
   (lowercasing, number/currency ITN, spelling standardisation); ``verbatim``
   returns the cleaned text with its original casing and wording;
   ``punctuated`` restores punctuation and casing on the verbatim text with a
   neural model (see ``punctuator.py``), adding no words and removing none.

No stage ever resolves self-corrections: "five no wait ten" stays intact.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, get_args

from whisper_normalizer.english import EnglishTextNormalizer

if TYPE_CHECKING:
    from .punctuator import Punctuator

Mode = Literal["canonical", "verbatim", "punctuated"]
MODES = get_args(Mode)


class PunctuationUnavailableError(RuntimeError):
    """``punctuated`` mode was requested but no punctuation model is loaded."""

# Word edges that also treat hyphens and apostrophes as part of a word, so
# "la" never matches inside "la-la" and "lo" never matches inside "lo-fi".
_WORD_START = r"(?<![\w'-])"
_WORD_END = r"(?![\w'-])"

# The word directly before / after a match, with only whitespace in between.
# Punctuation in between ("what, you know, ...") means no adjacent word.
_PREV_WORD = re.compile(r"([\w']+)\s+$")
_NEXT_WORD = re.compile(r"\s+([\w']+)")

_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "what", "which",
    "whatever", "any", "some", "every", "each", "no", "one", "same", "another",
    "my", "your", "his", "her", "its", "our", "their", "different",
})


@dataclass(frozen=True)
class _Guard:
    """Neighbouring words that signal a filler phrase is being used literally."""

    before: frozenset = field(default_factory=frozenset)
    after: frozenset = field(default_factory=frozenset)


# Multi-word fillers that are also ordinary phrases. A phrase is kept when its
# neighbours show the literal reading: "do you know him", "what I mean is",
# "what kind of car".
_FILLER_GUARDS = {
    "you know": _Guard(
        before=frozenset({
            "do", "did", "does", "don't", "didn't", "doesn't", "would", "will",
            "could", "can", "should", "might", "if", "as", "whether", "how",
            "what", "who",
        }),
        after=frozenset({"him", "her", "them", "me", "us", "it"}),
    ),
    "i mean": _Guard(before=frozenset({"what", "whatever"}), after=frozenset({"it"})),
    "kind of": _Guard(before=_DETERMINERS),
    "sort of": _Guard(before=_DETERMINERS),
}

# Single-letter tokens that are real English words. A run made only of these
# is speech ("I I think", "am I a fool"), not a spelled-out acronym.
_LETTER_WORDS = frozenset({"I", "i", "a"})

# Dotted acronyms: "U.S.A.", "U.S", "H.D.B." (but not "x.com" or "Ph.D.").
_DOTTED_ACRONYM = re.compile(r"(?<![\w.])[A-Za-z](?:\.[A-Za-z])+\.?(?!\w)")
# Runs of two or more space-separated single letters: "U S A", "H D B".
_LETTER_RUN = re.compile(r"(?<![\w'.-])[A-Za-z](?:\s+[A-Za-z])+(?![\w-])")
# End of text or a following capitalised word: an acronym's final dot also
# ends the sentence ("the U.S.A. Then ..." vs "the U.S.A. is ...").
_SENTENCE_END = re.compile(r"\s*$|\s+[A-Z]")

_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
_REPEATED_SOFT_PUNCT = re.compile(r"([,;:])[,;:]+")
_SOFT_BEFORE_HARD_PUNCT = re.compile(r"[,;:]+([.?!])")
_LEADING_PUNCT = re.compile(r"^(?:[,;:]+\s*|[.?!]\s+)")
_TRAILING_SOFT_PUNCT = re.compile(r"\s*[,;:]+$")
_HAS_WORD = re.compile(r"\w")


class TranscriptNormalizer:
    def __init__(self, punctuator: Optional["Punctuator"] = None) -> None:
        self.whisper_norm = EnglishTextNormalizer()
        self.punctuator = punctuator

        # Standard verbal fillers
        self.fillers = {
            "um", "umm", "uh", "uhh", "er", "err", "ah", "ahh",
            "you know", "i mean", "sort of", "kind of",
        }

        # Singlish discourse particles & loanword interjections
        self.singlish_particles = {
            "lah", "la", "lar",
            "lor", "lorh", "lo",
            "leh", "ley",
            "meh",
            "hor", "horh",
            "sia", "siah",
            "mah",
            "liao",
            "kena",
            "alamak",
            "walao", "walau",
        }

        self._compile_regexes()

    def _compile_regexes(self) -> None:
        self.fillers_regex = self._phrase_regex(self.fillers)
        self.particles_regex = self._phrase_regex(self.singlish_particles)

    @staticmethod
    def _phrase_regex(phrases: set[str]) -> re.Pattern:
        # Longest first so "you know" wins over any shorter overlapping entry;
        # words inside a phrase may be separated by any run of whitespace.
        alternatives = [
            r"\s+".join(re.escape(word) for word in phrase.split())
            for phrase in sorted(phrases, key=len, reverse=True)
        ]
        return re.compile(rf"{_WORD_START}(?:{'|'.join(alternatives)}){_WORD_END}", re.IGNORECASE)

    # ------------------------------------------------------------------ #
    # Stage 1: disfluencies
    # ------------------------------------------------------------------ #
    def _remove_fillers(self, text: str) -> str:
        def replace(match: re.Match) -> str:
            guard = _FILLER_GUARDS.get(" ".join(match.group(0).lower().split()))
            if guard is not None:
                prev_word = _PREV_WORD.search(text, 0, match.start())
                next_word = _NEXT_WORD.match(text, match.end())
                if (prev_word and prev_word.group(1).lower() in guard.before) or (
                    next_word and next_word.group(1).lower() in guard.after
                ):
                    return match.group(0)
            return ""

        return self.fillers_regex.sub(replace, text)

    def _remove_particles(self, text: str) -> str:
        # In mixed-case text an all-caps token is an abbreviation, not a
        # particle: "flight to LA lah" keeps "LA".
        keep_all_caps = not text.isupper()

        def replace(match: re.Match) -> str:
            token = match.group(0)
            if keep_all_caps and len(token) > 1 and token.isupper():
                return token
            return ""

        return self.particles_regex.sub(replace, text)

    @staticmethod
    def _tidy(text: str) -> str:
        """Repair spacing and punctuation left behind by removed tokens."""
        text = _WHITESPACE.sub(" ", text).strip()
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _REPEATED_SOFT_PUNCT.sub(r"\1", text)
        text = _SOFT_BEFORE_HARD_PUNCT.sub(r"\1", text)
        text = _LEADING_PUNCT.sub("", text)
        text = _TRAILING_SOFT_PUNCT.sub("", text)
        return text if _HAS_WORD.search(text) else ""

    def clean_disfluencies(
        self, text: str, remove_fillers: bool = True, remove_particles: bool = True
    ) -> str:
        if remove_fillers:
            text = self._remove_fillers(text)
        if remove_particles:
            text = self._remove_particles(text)
        return self._tidy(text)

    # ------------------------------------------------------------------ #
    # Stage 2: acronyms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _join_dotted(match: re.Match) -> str:
        acronym = match.group(0)
        letters = acronym.replace(".", "")
        # Keep the final dot only when it doubles as a full stop.
        if acronym.endswith(".") and _SENTENCE_END.match(match.string, match.end()):
            return letters + "."
        return letters

    @staticmethod
    def _join_letter_run(match: re.Match) -> str:
        tokens = match.group(0).split()
        if all(token in _LETTER_WORDS for token in tokens):
            return match.group(0)

        # Next to capitals, a lowercase "a" is the article: "a U S visa".
        head, tail = [], []
        if any(token.isupper() for token in tokens):
            while tokens and tokens[0] == "a":
                head.append(tokens.pop(0))
            while tokens and tokens[-1] == "a":
                tail.insert(0, tokens.pop())

        core = ["".join(tokens)] if len(tokens) >= 2 else tokens
        return " ".join(head + core + tail)

    def stitch_acronyms(self, text: str) -> str:
        text = _DOTTED_ACRONYM.sub(self._join_dotted, text)
        text = _LETTER_RUN.sub(self._join_letter_run, text)
        return text

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def normalize(
        self,
        text: str,
        remove_fillers: bool = True,
        remove_particles: bool = True,
        mode: Mode = "canonical",
    ) -> str:
        return self.normalize_batch([text], remove_fillers, remove_particles, mode)[0]

    def normalize_batch(
        self,
        texts: Sequence[str],
        remove_fillers: bool = True,
        remove_particles: bool = True,
        mode: Mode = "canonical",
    ) -> List[str]:
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode!r}")
        if mode == "punctuated" and self.punctuator is None:
            raise PunctuationUnavailableError("Punctuation model is not loaded")

        # Step 1: Strip fillers and particles. Step 2: Stitch acronyms.
        stitched = [
            self.stitch_acronyms(self.clean_disfluencies(text, remove_fillers, remove_particles))
            if text and text.strip() else ""
            for text in texts
        ]

        # Step 3: Format output based on mode
        if mode == "canonical":
            # whisper-normalizer lowercases, maps numbers to digits, standardizes spellings
            return [self.whisper_norm(text) if text else "" for text in stitched]
        if mode == "punctuated":
            # One model call for the whole batch.
            return self.punctuator.punctuate_batch(stitched)

        # Verbatim mode: preserve original casing and layout, only clean disfluencies
        return stitched
