"""The Dataset Catalog's version format (contracts/datasets/version.md): the rules a version folder or a reference
package follows (`check_folder`), its digest (`digest`), and the reading of its references (`parse`).

    versions/<dataset_id>/<n>/manifest.json, files.csv, refs/<one references file>, refs/uids.jsonl,
                              refs/attributes.json

The catalog service (catalog/datasets) and FakeDatasetCatalog (aicore.datasets) both use this module, so the fake
refuses what the service refuses, with the same messages. It needs only the standard library.

The reader knows the three references files Concord Label writes (segments.seglst.json, segments.jsonl and
refs.rttm) and STM. A version may carry any other format the apps read; it is stored unchanged and its segments
are reported as unknown (None).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

FORMAT = "concord-reference-package"
FORMAT_VERSIONS = (1, 2, 3)  # format versions a reader takes
FORMAT_VERSION = 3  # the version the catalog writes for audio-only versions
LABEL_TYPES = ("asr", "diar")
MANIFEST_REQUIRED = ("format", "format_version", "name", "created_at", "label_types", "counts", "audio_included")
SOURCE_FIELDS = ("app", "install_id", "dataset_uid", "dataset_name")
FILES_HEADER = ["path", "sha256", "duration", "sample_rate"]
# Fields the catalog fills when it stores a version; the digest leaves them out, so the same content sent twice has
# the same digest.
CATALOG_FIELDS = ("source", "export_version", "created_at", "audio_included")
INSTALL_ID = "dataset-catalog"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma", ".webm"}
HEX64 = re.compile(r"[0-9a-f]{64}")  # with fullmatch: a "$" would also match before a final newline
UID_TOLERANCE_S = 0.01
HASH_CHUNK = 1 << 20


def rel_ok(rel: str) -> bool:
    """A relative POSIX path without '..' (paths inside packages and zips)."""
    p = PurePosixPath(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts and "\\" not in rel


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def read_files_csv(root: Path) -> list[dict[str, str]]:
    with open(root / "files.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def side_files(version: Any) -> set[str]:
    """Files under refs/ besides the references file. In format version 1, attributes.json is a references file."""
    return {"uids.jsonl"} if version == 1 else {"uids.jsonl", "attributes.json"}


def _references(root: Path, side: set[str]) -> list[Path]:
    """The files under refs/ that aren't side files."""
    d = root / "refs"
    return [p for p in d.glob("*") if p.is_file() and p.name not in side] if d.is_dir() else []


# ----------------------------------------------------------------------------- the rules

def check_folder(root: Path, *, package: bool = False, hash_audio: bool = True) -> list[str]:
    """Problems with a version folder (contracts/datasets/version.md); empty when it follows the format.

    With `package`, the folder may also be a reference package: audio/<path> and outputs/ are allowed. With
    `hash_audio`, each file under audio/ is checked against its sha256 (the catalog checks them as it stores them
    instead)."""
    mpath = root / "manifest.json"
    if not mpath.is_file():
        return ["no manifest.json: a package without a manifest is an annotated upload, which only the apps read"]
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"manifest.json is not JSON: {exc}"]
    if not isinstance(m, dict):
        return ["manifest.json must be an object"]
    problems = _manifest_problems(m)
    if not package:
        problems += [f"{d}/ is not part of a version" for d in ("audio", "outputs") if (root / d).exists()]
        if m.get("audio_included") not in (None, "none"):
            problems.append(f"manifest.json: audio_included must be 'none' in a version, not {m.get('audio_included')!r}")
    files_problems, paths = _files_problems(root, m, package=package, hash_audio=hash_audio)
    problems += files_problems
    version = m.get("format_version")
    types = m["label_types"] if isinstance(m.get("label_types"), list) else []
    found = _references(root, side_files(version))
    problems += _refs_problems(root, version, types, found, paths)
    if types and len(found) == 1 and not problems:
        try:  # the same reading publishing does, so a folder that passes here can be stored
            parsed = parse(root)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            problems.append(f"refs/{found[0].name}: can't be read ({exc!r})")
        else:
            problems += _time_problems(found[0].name, parsed) + _speaker_problems(found[0].name, parsed)
    return problems


def _time_problems(name: str, parsed: dict[str, Any]) -> list[str]:
    """One problem per segment whose start or end is NaN or infinite (float() takes "nan" and "inf", and JSON readers
    take NaN and Infinity): stored, a NaN reads back as null and breaks sorting segments by time."""
    problems = []
    for path, segs in (parsed["segments"] or {}).items():
        for s in segs:
            if not (math.isfinite(s["start"]) and math.isfinite(s["end"])):
                problems.append(f"refs/{name}: a segment of {path} has start {s['start']!r} and end {s['end']!r}; "
                                "times must be finite numbers")
    return problems


def _speaker_problems(name: str, parsed: dict[str, Any]) -> list[str]:
    """A speaker is a string, or missing: the apps read each speaker as a string, and attributes.json names
    speakers by string. One problem per file, for its first other speaker (a list, a number, true)."""
    problems = []
    for path, segs in (parsed["segments"] or {}).items():
        bad = [s["speaker"] for s in segs if s["speaker"] is not None and not isinstance(s["speaker"], str)]
        if bad:
            problems.append(f"refs/{name}: speaker {bad[0]!r} of {path} is not a string")
    return problems


def _manifest_problems(m: dict[str, Any]) -> list[str]:
    problems = [f"manifest.json: missing {k}" for k in MANIFEST_REQUIRED if k not in m]
    if m.get("format") != FORMAT:
        problems.append(f"manifest.json: format {m.get('format')!r}")
    version = m.get("format_version")
    if version not in FORMAT_VERSIONS:
        problems.append(f"manifest.json: format_version {version!r} (known: {', '.join(map(str, FORMAT_VERSIONS))})")
    types = m.get("label_types")
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types) \
            or not set(types) <= set(LABEL_TYPES) or (not types and version != 3):
        problems.append(f"manifest.json: label_types {types!r}")
    if m.get("audio_included") not in ("all", "partial", "none"):
        problems.append(f"manifest.json: audio_included {m.get('audio_included')!r}")
    if m.get("scope") not in (None, "complete", "any"):
        problems.append(f"manifest.json: scope {m.get('scope')!r}")
    if not isinstance(m.get("counts"), dict):
        problems.append("manifest.json: counts must be an object")
    if "name" in m and not isinstance(m["name"], str):
        problems.append("manifest.json: name must be a string")
    problems += [f"manifest.json: {k} must be a string" for k in ("language", "description")
                 if m.get(k) is not None and not isinstance(m[k], str)]
    tags = m.get("tags")
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
        problems.append("manifest.json: tags must be a list of strings")
    src = m.get("source")
    if src is not None:
        if not isinstance(src, dict):
            problems.append("manifest.json: source must be an object")
        else:
            problems += [f"manifest.json: source.{k} missing" for k in SOURCE_FIELDS if not src.get(k)]
        if not isinstance(m.get("export_version"), int) or m["export_version"] < 1:
            problems.append("manifest.json: export_version must be 1, 2, ... when source is given")
    return problems


def _files_problems(root: Path, m: dict[str, Any], *, package: bool, hash_audio: bool) -> tuple[list[str], set[str]]:
    """Problems with files.csv (and audio/ in a package), and the files.csv paths."""
    problems: list[str] = []
    rows: list[dict[str, str]] = []
    if not (root / "files.csv").is_file():
        problems.append("files.csv missing")
    else:
        try:
            with open(root / "files.csv", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames != FILES_HEADER:
                    problems.append(f"files.csv: header {reader.fieldnames}")
                rows = list(reader)
        except UnicodeDecodeError as exc:
            problems.append(f"files.csv is not UTF-8: {exc}")
        except csv.Error as exc:
            problems.append(f"files.csv can't be read: {exc}")
    stems: dict[str, str] = {}
    included = 0
    for r in rows:
        rel = r.get("path") or ""
        if not rel_ok(rel):
            problems.append(f"files.csv: bad path {rel!r}")
            continue
        stem = PurePosixPath(rel).stem.lower()
        if stem in stems:
            problems.append(f"files.csv: stem of {rel} repeats {stems[stem]}")
        stems[stem] = rel
        if not HEX64.fullmatch(r.get("sha256") or ""):
            problems.append(f"files.csv: sha256 of {rel}")
        for k, ok in (("duration", _duration_ok), ("sample_rate", _sample_rate_ok)):
            if not ok(r.get(k) or ""):
                problems.append(f"files.csv: {k} of {rel} is {r.get(k)!r}")
        audio = root / "audio" / rel
        if package and audio.is_file():
            included += 1
            if hash_audio and sha256_file(audio) != r.get("sha256"):
                problems.append(f"audio/{rel}: bytes do not match its sha256")
    counts = m.get("counts") if isinstance(m.get("counts"), dict) else {}
    if counts.get("files") != len(rows):
        problems.append(f"manifest.json: counts.files {counts.get('files')} but files.csv has {len(rows)} rows")
    if package:
        want = {"all": len(rows), "none": 0}.get(m.get("audio_included"))
        if want is not None and included != want:
            problems.append(f"audio_included is {m.get('audio_included')!r} but {included} of {len(rows)} files "
                            "are in audio/")
        if m.get("audio_included") == "partial" and not 0 < included < len(rows):
            problems.append("audio_included is 'partial' but audio/ has none or all of the files")
    return problems, set(stems.values())


def _duration_ok(text: str) -> bool:
    """Seconds as a finite number of 0 or more (0 is audio whose length isn't known). float() also takes "nan", "inf"
    and "1_0", which would break the hours of the dataset or its JSON later."""
    try:
        d = float(text)
    except ValueError:
        return False
    return "_" not in text and math.isfinite(d) and d >= 0


def _sample_rate_ok(text: str) -> bool:
    """Hz as a whole number above 0, without the "_" int() takes."""
    try:
        return "_" not in text and int(text) > 0
    except ValueError:
        return False


def _refs_problems(root: Path, version: Any, types: list[Any], found: list[Path], paths: set[str]) -> list[str]:
    """Problems with refs/: its files, attributes.json and uids.jsonl."""
    problems: list[str] = []
    side, refs = side_files(version), root / "refs"
    if types and len(found) != 1:
        problems.append(f"refs/ must hold exactly one references file besides {' and '.join(sorted(side))} "
                        f"(found {len(found)})")
    if not types and refs.is_dir() and any(refs.iterdir()):
        problems.append("refs/ must be empty when label_types is empty (a version with audio only)")
    if refs.is_dir() and any(p.is_dir() for p in refs.iterdir()):
        problems.append("refs/ must not hold folders")
    attrs = refs / "attributes.json"
    if "attributes.json" in side and types and attrs.is_file():
        try:
            problems += check_attributes(json.loads(attrs.read_text(encoding="utf-8")), paths)
        except ValueError as exc:
            problems.append(f"refs/attributes.json is not JSON: {exc}")
    if types and (refs / "uids.jsonl").is_file():
        problems += _uids_problems(refs / "uids.jsonl", paths)
    return problems


def _uids_problems(uids: Path, paths: set[str]) -> list[str]:
    try:
        lines = uids.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        return [f"refs/uids.jsonl is not UTF-8: {exc}"]
    problems = []
    for n, line in enumerate(lines, 1):
        if line.strip():
            try:
                rec = json.loads(line)
            except ValueError:
                rec = {}
            if not isinstance(rec, dict) or not {"file", "start", "end", "uid"} <= rec.keys() \
                    or not isinstance(rec["file"], str) or rec["file"] not in paths \
                    or not _number(rec["start"]) or not _number(rec["end"]):
                problems.append(f"refs/uids.jsonl line {n}: needs file (a files.csv path), start, end, uid")
    return problems


def _number(x: object) -> bool:
    """A finite JSON number: not true / false, NaN, Infinity or an integer too large for a float."""
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:
        return False


def _labels_ok(obj: object) -> bool:
    """A labels object: strings, numbers, true / false, or nested labels objects."""
    return isinstance(obj, dict) and all(isinstance(v, (str, int, float, bool)) or _labels_ok(v) for v in obj.values())


def check_attributes(a: object, paths: set[str]) -> list[str]:
    """Problems with refs/attributes.json, given the files.csv paths."""
    if not isinstance(a, dict) or set(a) - {"files", "speakers"}:
        return ["refs/attributes.json: must be an object with files and / or speakers"]
    problems = [f"refs/attributes.json: {k} must be an object" for k in ("files", "speakers")
                if a.get(k) is not None and not isinstance(a[k], dict)]
    if problems:
        return problems
    for path, labels in (a.get("files") or {}).items():
        if path not in paths:
            problems.append(f"refs/attributes.json: files.{path} is not a files.csv path")
        elif not _labels_ok(labels):
            problems.append(f"refs/attributes.json: files.{path} must be an object of labels")
    for path, speakers in (a.get("speakers") or {}).items():
        if path not in paths:
            problems.append(f"refs/attributes.json: speakers.{path} is not a files.csv path")
        elif not isinstance(speakers, dict) or not all(_labels_ok(v) for v in speakers.values()):
            problems.append(f"refs/attributes.json: speakers.{path} must map each speaker to an object of labels")
    return problems


# ----------------------------------------------------------------------------- the digest

def version_files(root: Path) -> list[tuple[str, Path]]:
    """(name, path) of the files that make a version, in digest order: manifest.json, files.csv, refs/ by name."""
    parts = [("manifest.json", root / "manifest.json"), ("files.csv", root / "files.csv")]
    if (root / "refs").is_dir():
        parts += [(f"refs/{p.name}", p) for p in sorted((root / "refs").iterdir(), key=lambda p: p.name) if p.is_file()]
    return parts


def canonical_json(obj: Any) -> str:
    """JSON as the digest hashes a manifest: keys sorted, no spaces, and every character past ASCII escaped, so
    {"name": "Café"} is {"name":"Caf\\u00e9"}."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(root: Path) -> str:
    """sha256 over each part as `<name>\\n<length>\\n<bytes>`, in `version_files` order. The manifest is hashed as
    canonical JSON without CATALOG_FIELDS."""
    h = hashlib.sha256()
    for name, path in version_files(root):
        data = path.read_bytes()
        if name == "manifest.json":
            m = json.loads(data)
            data = canonical_json({k: v for k, v in m.items() if k not in CATALOG_FIELDS}).encode()
        h.update(f"{name}\n{len(data)}\n".encode())
        h.update(data)
    return h.hexdigest()


# ----------------------------------------------------------------------------- reading references

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


def segments(path: Path) -> list[tuple[str, dict[str, Any]]] | None:
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


def _format_version(root: Path) -> Any:
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return m.get("format_version") if isinstance(m, dict) else None


def refs_file(root: Path, side: set[str] | None = None) -> Path | None:
    """The one references file of a version folder, or None (an audio-only version). `side` are the other files
    under refs/, by default the side files of the manifest's format version."""
    files = sorted(_references(root, side_files(_format_version(root)) if side is None else side))
    return files[0] if files else None


def _take_uid(segs: list[dict[str, Any]], u: dict[str, Any]) -> None:
    """Give uid record `u` to the nearest of a file's segments within UID_TOLERANCE_S that has no uid yet, as Concord
    Bench does, so each uid names one segment even when two speakers' segments have the same times."""
    free = [s for s in segs if s["uid"] is None
            and abs(s["start"] - u["start"]) <= UID_TOLERANCE_S and abs(s["end"] - u["end"]) <= UID_TOLERANCE_S]
    if free:
        min(free, key=lambda s: abs(s["start"] - u["start"]) + abs(s["end"] - u["end"]))["uid"] = u["uid"]


def parse(root: Path) -> dict[str, Any]:
    """A version folder's references: segments per file path (None for a format the catalog doesn't read), file and
    speaker labels, and the sorted names of every label."""
    paths = [r["path"] for r in read_files_csv(root)]
    by_key = {p: p for p in paths} | {PurePosixPath(p).stem.lower(): p for p in paths}
    side = side_files(_format_version(root))
    rf = refs_file(root, side)
    segs: dict[str, list[dict[str, Any]]] | None = None
    if rf is not None:
        records = segments(rf)
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
                _take_uid(segs.get(u["file"], []), u)
    attrs_path = root / "refs" / "attributes.json"
    attrs = json.loads(attrs_path.read_text(encoding="utf-8")) if "attributes.json" in side and attrs_path.is_file() \
        else {}
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


def file_summary(parsed: dict[str, Any], path: str) -> dict[str, Any]:
    """One file's segment count, speakers and file labels, from `parse` (None when the format isn't read)."""
    segs = None if parsed["segments"] is None else parsed["segments"].get(path, [])
    return {"segments": None if segs is None else len(segs),
            "speakers": None if segs is None else sorted({s["speaker"] for s in segs if s["speaker"] is not None}),
            "labels": parsed["file_labels"].get(path, {})}


def file_refs(parsed: dict[str, Any], path: str) -> dict[str, Any]:
    """One file's references, from `parse`: its segments, file labels and speaker labels."""
    segs = None if parsed["segments"] is None else parsed["segments"].get(path, [])
    return {"path": path, "format": parsed["format"], "segments": segs,
            "labels": parsed["file_labels"].get(path, {}), "speakers": parsed["speaker_labels"].get(path, {})}
