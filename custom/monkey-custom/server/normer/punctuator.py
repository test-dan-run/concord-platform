# Vendored from github.com/test-dan-run/transcript-normer @e98d4ae (app/services/punctuator.py).
# Changed for the platform: package-relative imports only. Update by re-copying at a pinned commit.
"""Punctuation and true-casing restoration for cleaned transcripts.

Runs the ``1-800-BAD-CODE/punct_cap_seg_en`` ONNX model directly through
onnxruntime + sentencepiece (no PyTorch). The model reads lowercase,
unpunctuated text and predicts, per subword token, a trailing punctuation mark
and per-character capitalisation.

The model only ever *adds* punctuation and changes letter case: output words
are rebuilt from the input words, never from decoded tokens, so characters the
tokenizer cannot represent (hyphens, ``%``, ``$``) survive untouched, and no
word is added, dropped or reordered.

Run ``python -m app.services.punctuator [DIR]`` to download the model ahead of
time (the Docker build does this).
"""

import hashlib
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from sentencepiece import SentencePieceProcessor

MODEL_REPO = "1-800-BAD-CODE/punct_cap_seg_en"
MODEL_REVISION = "b26fd1c40e88678859048898218ea4edcc24c84a"
# filename -> sha256 at MODEL_REVISION
MODEL_FILES = {
    "punct_cap_seg_en.onnx": "dd922d459da618cd324280889740608b76fb3e9e61d3f402291be1251f91421b",
    "spe_32k_lc_en.model": "9e86d0263de80b3b68327a21f5350c8cdf846e4c4400253c9baf05e3d44871c3",
}
_ONNX_FILE, _SPM_FILE = MODEL_FILES

# From the model's config.yaml at MODEL_REVISION.
_POST_LABELS = (None, "<ACRONYM>", ".", ",", "?")
_MAX_LENGTH = 256  # tokens per window, including BOS/EOS
_OVERLAP = 16  # tokens shared between consecutive windows of a long text
_BATCH_TOKENS = 4096  # padded tokens per inference call
_DOWNLOAD_ATTEMPTS = 3

_TERMINAL = frozenset(".?")
# Punctuation stripped from word edges before inference; the model re-predicts it.
_EDGE_PUNCT = re.compile(r"^[,.;:!?]+|[,.;:!?]+$")


def default_model_dir() -> Path:
    return Path.home() / ".cache" / "transcript-normer" / "punct_cap_seg_en"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(model_dir: Path) -> Path:
    """Fetch any missing model files into ``model_dir``, verifying checksums."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in MODEL_FILES.items():
        target = model_dir / filename
        if target.exists():
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}"
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                _fetch(url, partial, expected)
                break
            # URLError, timeouts and resets are OSErrors; a truncated body
            # shows up as a checksum mismatch (ValueError).
            except (OSError, ValueError) as exc:
                partial.unlink(missing_ok=True)
                print(f"Download attempt {attempt}/{_DOWNLOAD_ATTEMPTS} of {filename} failed: {exc}", file=sys.stderr)
                if attempt == _DOWNLOAD_ATTEMPTS:
                    raise RuntimeError(f"Failed to download {url}") from exc
                time.sleep(2 * attempt)
        partial.rename(target)
    return model_dir


def _fetch(url: str, dest: Path, sha256: str) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    actual = _sha256(dest)
    if actual != sha256:
        raise ValueError(f"checksum mismatch: expected {sha256}, got {actual}")


@dataclass
class _Window:
    """A slice of one text's tokens that fits in the model's context."""

    text_idx: int
    offset: int  # position of ids[0] within the text's tokens
    ids: List[int]  # without BOS/EOS
    keep_from: int  # predictions for ids[keep_from:keep_to] are used;
    keep_to: int  # the rest is context shared with a neighbouring window


class Punctuator:
    def __init__(self, model_dir: Optional[Path] = None, download: bool = True) -> None:
        model_dir = Path(model_dir) if model_dir else default_model_dir()
        if download:
            download_model(model_dir)
        self._sp = SentencePieceProcessor(model_file=str(model_dir / _SPM_FILE))
        self._session = ort.InferenceSession(
            str(model_dir / _ONNX_FILE), providers=["CPUExecutionProvider"]
        )

    def punctuate(self, text: str) -> str:
        return self.punctuate_batch([text])[0]

    def punctuate_batch(self, texts: Sequence[str]) -> List[str]:
        # Words with their edge punctuation removed; the model predicts it afresh.
        words = [[w for w in (_EDGE_PUNCT.sub("", w) for w in text.split()) if w] for text in texts]

        # Tokenise word by word so every token knows which word it belongs to.
        token_ids: List[List[int]] = []
        token_words: List[List[int]] = []
        for text_words in words:
            ids, owners = [], []
            for word_idx, word in enumerate(text_words):
                for token_id in self._sp.encode(word.lower()):
                    ids.append(token_id)
                    owners.append(word_idx)
            token_ids.append(ids)
            token_words.append(owners)

        post_preds = [[0] * len(ids) for ids in token_ids]
        cap_preds: List[List[Optional[np.ndarray]]] = [[None] * len(ids) for ids in token_ids]
        for batch in self._batches(self._windows(token_ids)):
            for window, post, caps in self._infer(batch):
                for i in range(window.keep_from, window.keep_to):
                    post_preds[window.text_idx][window.offset + i] = int(post[i])
                    cap_preds[window.text_idx][window.offset + i] = caps[i]

        return [
            self._assemble(texts[i], words[i], token_ids[i], token_words[i], post_preds[i], cap_preds[i])
            for i in range(len(texts))
        ]

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @staticmethod
    def _windows(token_ids: List[List[int]]) -> List[_Window]:
        """Split long texts into overlapping windows the model can handle.

        At each seam the left window drops its last OVERLAP/2 predictions and
        the right window starts OVERLAP/2 tokens early, so every kept
        prediction has context on both sides.
        """
        size, half = _MAX_LENGTH - 2, _OVERLAP // 2
        windows = []
        for text_idx, ids in enumerate(token_ids):
            start = 0  # first token the next window is responsible for
            while start < len(ids):
                lo = max(0, start - half)
                hi = min(len(ids), lo + size)
                keep_to = hi if hi == len(ids) else hi - half
                windows.append(_Window(text_idx, lo, ids[lo:hi], start - lo, keep_to - lo))
                start = keep_to
        return windows

    @staticmethod
    def _batches(windows: List[_Window]) -> Iterator[List[_Window]]:
        """Group windows of similar length, capping padded tokens per call."""
        batch: List[_Window] = []
        for window in sorted(windows, key=lambda w: len(w.ids), reverse=True):
            # Sorted longest first, so batch[0] sets the padded width.
            if batch and (len(batch[0].ids) + 2) * (len(batch) + 1) > _BATCH_TOKENS:
                yield batch
                batch = []
            batch.append(window)
        if batch:
            yield batch

    def _infer(self, batch: List[_Window]):
        """Yield (window, post_preds, cap_preds) with BOS/EOS positions removed."""
        width = len(batch[0].ids) + 2
        input_ids = np.full((len(batch), width), self._sp.pad_id(), dtype=np.int64)
        for row, window in enumerate(batch):
            input_ids[row, : len(window.ids) + 2] = [self._sp.bos_id(), *window.ids, self._sp.eos_id()]
        _, post, caps, _ = self._session.run(None, {"input_ids": input_ids})
        for row, window in enumerate(batch):
            n = len(window.ids)
            yield window, post[row, 1 : n + 1], caps[row, 1 : n + 1]

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def _assemble(self, original, words, ids, owners, post_preds, cap_preds) -> str:
        if not words:
            return ""
        # Group token predictions by word.
        word_tokens: List[List[int]] = [[] for _ in words]
        for position, word_idx in enumerate(owners):
            word_tokens[word_idx].append(position)

        keep_caps = not original.isupper()
        out = []
        for word, positions in zip(words, word_tokens):
            cased = self._case_word(word, [ids[p] for p in positions], [cap_preds[p] for p in positions])
            # Spelled-out / stitched acronyms from the input stay upper case.
            if keep_caps and len(word) > 1 and word.isupper():
                cased = word
            label = _POST_LABELS[post_preds[positions[-1]]] if positions else None
            # "<ACRONYM>" would add dots (U.S.A.); stitched acronyms stay dotless.
            out.append(cased + (label if label and label != "<ACRONYM>" else ""))

        # Always produce a sentence: capitalised start, terminal mark at the end.
        out[0] = out[0][:1].upper() + out[0][1:]
        last = out[-1].rstrip(",")
        out[-1] = last if last[-1:] in _TERMINAL else last + "."
        return " ".join(out)

    def _case_word(self, word: str, ids: List[int], caps: List[np.ndarray]) -> str:
        lower = word.lower()
        # Pieces the tokenizer could represent (not <unk>) with their case
        # flags. Flags are indexed by character *including* a leading "▁".
        known = []
        for token_id, token_caps in zip(ids, caps):
            raw = self._sp.id_to_piece(token_id)
            skip = 1 if raw.startswith("▁") else 0
            piece = raw[skip:]
            if piece and not (piece.startswith("<") and piece.endswith(">")):
                known.append((piece, token_caps[skip : skip + len(piece)]))
        flags = [bool(flag) for _, piece_caps in known for flag in piece_caps]
        if "".join(piece for piece, _ in known) == lower and len(flags) == len(lower):
            # Tokens spell the word exactly: apply per-character casing.
            return "".join(ch.upper() if flag else ch for ch, flag in zip(lower, flags))
        # Some characters were unknown to the tokenizer: fall back to
        # upper-casing the whole word or just its first letter.
        if flags and all(flags) and len(lower) > 1:
            return word.upper()
        if flags[:1] == [True]:
            return word[:1].upper() + lower[1:]
        return lower

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.getenv("PUNCTUATION_MODEL_DIR") or default_model_dir())
    print(f"Model ready in {download_model(target)}")
