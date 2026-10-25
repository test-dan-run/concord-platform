"""refs.compare_segments pairs segments as the first, simple version did (kept below), and fast: it looks segments
up by uid and by start time instead of scanning every segment of b for each segment of a."""

import difflib
import random
import time

from catalog.datasets import refs
from catalog.datasets.refs import EDIT_OVERLAP, UID_TOLERANCE_S

NAN = float("nan")  # json.loads gives this same object for every NaN


# ----------------------------------------------------------------------------- the first version, as it was

def _text_diff(a, b):
    wa, wb = (a or "").split(), (b or "").split()
    return [{"op": op, "a": " ".join(wa[i1:i2]), "b": " ".join(wb[j1:j2])}
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes()]


def _same_times(x, y):
    return abs(x["start"] - y["start"]) <= UID_TOLERANCE_S and abs(x["end"] - y["end"]) <= UID_TOLERANCE_S


def _overlap(x, y):
    inter = min(x["end"], y["end"]) - max(x["start"], y["start"])
    union = max(x["end"], y["end"]) - min(x["start"], y["start"])
    return inter / union if inter > 0 and union > 0 else 0.0


def _changed(x, y):
    return not _same_times(x, y) or x["speaker"] != y["speaker"] or x["text"] != y["text"] or x["labels"] != y["labels"]


def slow_compare_segments(a, b):
    left, right = list(a), list(b)
    pairs = []
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
    edited = [{"a": x, "b": y, "diff": _text_diff(x["text"], y["text"])} for x, y in pairs if _changed(x, y)]
    edited.sort(key=lambda e: e["b"]["start"])
    return {"added": right, "removed": left, "edited": edited}


# ----------------------------------------------------------------------------- random files

UIDS = [None, None, "u1", "u2", "u3", 1, 1.0, True, 2, ["u", 1], {"k": "u1"}, NAN]


def _seg(r, t, uid_rate):
    start = round(t + r.choice([0.0, 0.0, 0.005, 0.01, 0.2, -0.3]), 3)
    end = round(start + r.choice([r.uniform(0.2, 3), 0.0, -0.5, r.uniform(5, 20)]), 3)
    uid = None
    if r.random() < uid_rate:
        uid = r.choice(UIDS) if r.random() < 0.3 else f"s{r.randrange(10 ** 6)}"
    return {"start": start, "end": end, "speaker": r.choice(["A", "B", None]), "text": r.choice(["hi there", "ok", "",
            None, "hi here"]), "uid": uid, "labels": r.choice([{}, {}, {"noise": "street"}])}


def _files(seed):
    r = random.Random(seed)
    n = r.randrange(0, 60)
    uid_rate = r.choice([0.0, 0.5, 1.0])
    base = r.choice([0.0, 0.0, 3600.0, 1e6])
    a, t = [], base
    for _ in range(n):
        a.append(_seg(r, t, uid_rate))
        t += r.choice([0.0, 0.5, 1.0, 2.5])
    b = []
    for x in a:
        p = r.random()
        if p < 0.1:
            continue  # deleted
        y = dict(x)
        if p < 0.2:
            y["start"] = round(y["start"] + r.choice([0.004, 0.01, 0.011, 0.3, 1.5, -0.2]), 3)
        elif p < 0.3:
            y["end"] = round(y["end"] + r.choice([0.009, 0.02, 0.8]), 3)
        elif p < 0.38:
            y["text"] = (y["text"] or "") + " more"
        elif p < 0.44:
            y["uid"] = r.choice(UIDS)
        elif p < 0.5:
            y["speaker"] = "C"
        b.append(y)
        if r.random() < 0.08:
            b.append(_seg(r, x["start"], uid_rate))  # added
        if r.random() < 0.05:
            b.append(dict(y))  # the same segment twice
    for s in (a, b):
        if r.random() < 0.8:
            s.sort(key=lambda s: (s["start"], s["end"]))
        else:
            r.shuffle(s)
    return a, b


def _shape(out):
    """The answer by identity too, so a swap of two equal segments shows."""
    return ([id(s) for s in out["added"]], [id(s) for s in out["removed"]],
            [(id(e["a"]), id(e["b"]), e["diff"]) for e in out["edited"]])


def test_compare_segments_pairs_as_the_first_version_did():
    for seed in range(3000):
        a, b = _files(seed)
        new, old = refs.compare_segments(a, b), slow_compare_segments(a, b)
        assert _shape(new) == _shape(old), seed
        assert new == old, seed


def test_compare_segments_leaves_its_input_alone():
    a, b = _files(7)
    a0, b0 = [dict(s) for s in a], [dict(s) for s in b]
    refs.compare_segments(a, b)
    assert (a, b) == (a0, b0)


def test_three_thousand_segments_compare_fast():
    # Every time moved by 0.05 s (another aligner, say): no uid or time matches, so each segment goes to the
    # overlap pass.
    r = random.Random(1)
    a, t = [], 0.0
    for i in range(3000):
        a.append({"start": round(t, 3), "end": round(t + r.uniform(0.5, 3), 3), "speaker": "A", "text": f"w{i} a b",
                  "uid": None, "labels": {}})
        t = a[-1]["end"] + r.choice([0.1, -0.2, 0.0])
    b = [dict(s, start=round(s["start"] + 0.05, 3), end=round(s["end"] + 0.05, 3)) for i, s in enumerate(a) if i % 50]
    t0 = time.perf_counter()
    out = refs.compare_segments(a, b)
    took = time.perf_counter() - t0
    assert (len(out["added"]), len(out["removed"]), len(out["edited"])) == (0, 60, 2940)
    assert took < 0.3, f"3000 segments took {took:.2f} s"
