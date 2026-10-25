"""A version's references for the UI: `read` caches the parsed references of a version, and `compare` is the
comparison of two versions.

Reading a references file (`parse`, `file_summary`, `file_refs`) is part of the version format, so it is in
aicore.dataset_format, which FakeDatasetCatalog shares.
"""

from __future__ import annotations

import difflib
import functools
from pathlib import Path
from typing import Any

# The reading itself, which callers also find here as refs.parse, refs.file_summary and so on.
from aicore.dataset_format import (UID_TOLERANCE_S, file_refs, file_summary, flatten, parse,  # noqa: F401
                                   refs_file, segments)

EDIT_OVERLAP = 0.5  # unmatched segments that overlap this much (intersection over union) count as one edited segment


@functools.lru_cache(maxsize=16)
def _read(root: str, digest: str) -> dict[str, Any]:
    return parse(Path(root))


def read(root: Path, digest: str) -> dict[str, Any]:
    """`parse`, once per version: a version never changes, so its folder and digest are the cache key."""
    return _read(str(root), digest)


# ----------------------------------------------------------------------------- compare

def text_diff(a: str | None, b: str | None) -> list[dict[str, str]]:
    """A word diff: [{"op": "equal" | "delete" | "insert" | "replace", "a": words, "b": words}, ...]."""
    wa, wb = (a or "").split(), (b or "").split()
    return [{"op": op, "a": " ".join(wa[i1:i2]), "b": " ".join(wb[j1:j2])}
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes()]


def _same_times(x: dict[str, Any], y: dict[str, Any]) -> bool:
    return abs(x["start"] - y["start"]) <= UID_TOLERANCE_S and abs(x["end"] - y["end"]) <= UID_TOLERANCE_S


def _overlap(x: dict[str, Any], y: dict[str, Any]) -> float:
    inter = min(x["end"], y["end"]) - max(x["start"], y["start"])
    union = max(x["end"], y["end"]) - min(x["start"], y["start"])
    return inter / union if inter > 0 and union > 0 else 0.0


def _changed(x: dict[str, Any], y: dict[str, Any]) -> bool:
    return not _same_times(x, y) or x["speaker"] != y["speaker"] or x["text"] != y["text"] or x["labels"] != y["labels"]


def compare_segments(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Segments added, removed and edited between two lists of one file's segments. A segment is the same one in
    both when its uid matches, else when its times match within 0.01 s; left over segments that overlap by half or
    more are one segment whose times were edited."""
    left, right = list(a), list(b)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for match in (lambda x, y: x["uid"] is not None and x["uid"] == y["uid"], _same_times):
        for x in list(left):
            y = next((y for y in right if match(x, y)), None)
            if y is not None:
                pairs.append((x, y))
                left.remove(x)
                right.remove(y)
    for x in list(left):
        best = max(right, key=lambda y: _overlap(x, y), default=None)
        if best is not None and _overlap(x, best) >= EDIT_OVERLAP:
            pairs.append((x, best))
            left.remove(x)
            right.remove(best)
    edited = [{"a": x, "b": y, "diff": text_diff(x["text"], y["text"])} for x, y in pairs if _changed(x, y)]
    edited.sort(key=lambda e: e["b"]["start"])
    return {"added": right, "removed": left, "edited": edited}


def compare(a_rows: dict[str, dict[str, Any]], b_rows: dict[str, dict[str, Any]],
            pa: dict[str, Any], pb: dict[str, Any]) -> dict[str, Any]:
    """Version a against version b, given each one's files.csv rows by path and its parsed references."""
    added = sorted(b_rows.keys() - a_rows.keys())
    removed = sorted(a_rows.keys() - b_rows.keys())
    changed: list[dict[str, Any]] = []
    same = 0
    known = pa["segments"] is not None and pb["segments"] is not None
    for path in sorted(a_rows.keys() & b_rows.keys()):
        entry: dict[str, Any] = {"path": path}
        if a_rows[path]["sha256"] != b_rows[path]["sha256"]:
            entry["audio_changed"] = True
        if known:
            segs = compare_segments(pa["segments"].get(path, []), pb["segments"].get(path, []))
            if any(segs.values()):
                entry["segments"] = segs
        la, lb = pa["file_labels"].get(path, {}), pb["file_labels"].get(path, {})
        sa, sb = pa["speaker_labels"].get(path, {}), pb["speaker_labels"].get(path, {})
        if la != lb or sa != sb:
            entry["labels"] = {"a": {"files": la, "speakers": sa}, "b": {"files": lb, "speakers": sb}}
        if len(entry) > 1:
            changed.append(entry)
        else:
            same += 1
    return {"files": {"added": added, "removed": removed, "same": same}, "changed": changed,
            "segments_compared": known,
            "labels": {"added": sorted(set(pb["labels"]) - set(pa["labels"])),
                       "removed": sorted(set(pa["labels"]) - set(pb["labels"]))}}
