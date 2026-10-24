"""Read a version's references for the UI: segments per file, uids and labels, and the comparison of two versions.

The catalog reads the three references files Concord Label writes (segments.seglst.json, segments.jsonl and
refs.rttm) and STM. A version may carry any other format the apps read (contracts/datasets/version.md); the catalog
stores it unchanged and reports its segments as unknown (None).
"""

from __future__ import annotations

import csv
import difflib
import functools
import json
from pathlib import Path, PurePosixPath
from typing import Any

UID_TOLERANCE_S = 0.01
EDIT_OVERLAP = 0.5  # unmatched segments that overlap this much (intersection over union) count as one edited segment


def flatten(labels: Any, prefix: str = "") -> dict[str, Any]:
    """Labels with nested objects read as dotted names: {"accent": {"region": "n"}} -> {"accent.region": "n"}."""
    out: dict[str, Any] = {}
    if not isinstance(labels, dict):
        return out
    for k, v in labels.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _seg(start: Any, end: Any, speaker: Any = None, text: Any = None, labels: Any = None) -> dict[str, Any]:
    return {"start": round(float(start), 3), "end": round(float(end), 3), "speaker": speaker,
            "text": text, "uid": None, "labels": flatten(labels)}


def _segments(path: Path) -> list[tuple[str, dict[str, Any]]] | None:
    """(file stem or path, segment) for every record of a references file; None for a format the catalog doesn't
    read."""
    name = path.name.lower()
    text = path.read_text(encoding="utf-8")
    out: list[tuple[str, dict[str, Any]]] = []
    if name.endswith(".jsonl"):
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                out.append((r["file"], _seg(r["start"], r["end"], r.get("speaker"), r.get("text"), r.get("attributes"))))
        return out
    if name.endswith(".json"):
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(r, dict) and "session_id" in r for r in data):
            return None
        for r in data:
            out.append((r["session_id"], _seg(r["start_time"], r["end_time"], r.get("speaker"), r.get("words"),
                                              r.get("attributes"))))
        return out
    if name.endswith(".rttm"):
        for line in text.splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == "SPEAKER":
                start = float(f[3])
                out.append((f[1], _seg(start, start + float(f[4]), f[7])))
        return out
    if name.endswith(".stm"):
        for line in text.splitlines():
            if not line.strip() or line.startswith(";;"):
                continue
            f = line.split(maxsplit=5)
            if len(f) < 5:
                continue
            words = f[5] if len(f) > 5 else ""
            if words.startswith("<") and ">" in words:  # the optional <label> field
                words = words.split(">", 1)[1].strip()
            out.append((f[0], _seg(f[3], f[4], f[2], words)))
        return out
    return None


def refs_file(root: Path) -> Path | None:
    """The one references file of a version folder, or None (an audio-only version)."""
    d = root / "refs"
    files = sorted(p for p in d.glob("*") if p.is_file() and p.name not in ("uids.jsonl", "attributes.json")) \
        if d.is_dir() else []
    return files[0] if files else None


def parse(root: Path) -> dict[str, Any]:
    """A version folder's references: segments per file path (None for a format the catalog doesn't read), file and
    speaker labels, and the sorted names of every label."""
    with open(root / "files.csv", newline="", encoding="utf-8") as fh:
        paths = [r["path"] for r in csv.DictReader(fh)]
    by_key = {p: p for p in paths} | {PurePosixPath(p).stem.lower(): p for p in paths}
    rf = refs_file(root)
    segs: dict[str, list[dict[str, Any]]] | None = None
    if rf is not None:
        records = _segments(rf)
        if records is not None:
            segs = {p: [] for p in paths}
            for key, s in records:
                p = by_key.get(key) or by_key.get(PurePosixPath(key).stem.lower())
                if p is not None:
                    segs[p].append(s)
            for v in segs.values():
                v.sort(key=lambda s: (s["start"], s["end"]))
    uids = root / "refs" / "uids.jsonl"
    if segs is not None and uids.is_file():
        for line in uids.read_text(encoding="utf-8").splitlines():
            if line.strip():
                u = json.loads(line)
                for s in segs.get(u["file"], []):
                    if abs(s["start"] - u["start"]) <= UID_TOLERANCE_S and abs(s["end"] - u["end"]) <= UID_TOLERANCE_S:
                        s["uid"] = u["uid"]
                        break
    attrs_path = root / "refs" / "attributes.json"
    attrs = json.loads(attrs_path.read_text(encoding="utf-8")) if attrs_path.is_file() else {}
    file_labels = {p: flatten(v) for p, v in (attrs.get("files") or {}).items()}
    speaker_labels = {p: {spk: flatten(v) for spk, v in spks.items()} for p, spks in (attrs.get("speakers") or {}).items()}
    names: set[str] = set()
    for v in file_labels.values():
        names |= v.keys()
    for spks in speaker_labels.values():
        for v in spks.values():
            names |= {f"speaker.{k}" for k in v}
    for v in (segs or {}).values():
        for s in v:
            names |= s["labels"].keys()
    return {"format": rf.name if rf else None, "segments": segs, "file_labels": file_labels,
            "speaker_labels": speaker_labels, "labels": sorted(names)}


@functools.lru_cache(maxsize=16)
def _read(root: str, digest: str) -> dict[str, Any]:
    return parse(Path(root))


def read(root: Path, digest: str) -> dict[str, Any]:
    """`parse`, once per version: a version never changes, so its folder and digest are the cache key."""
    return _read(str(root), digest)


def file_summary(parsed: dict[str, Any], path: str) -> dict[str, Any]:
    segs = None if parsed["segments"] is None else parsed["segments"].get(path, [])
    return {"segments": None if segs is None else len(segs),
            "speakers": None if segs is None else sorted({s["speaker"] for s in segs if s["speaker"] is not None}),
            "labels": parsed["file_labels"].get(path, {})}


def file_refs(parsed: dict[str, Any], path: str) -> dict[str, Any]:
    segs = None if parsed["segments"] is None else parsed["segments"].get(path, [])
    return {"path": path, "format": parsed["format"], "segments": segs,
            "labels": parsed["file_labels"].get(path, {}), "speakers": parsed["speaker_labels"].get(path, {})}


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
