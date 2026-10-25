"""FakeCatalog's output index (Bench and Label replay mode run on it): hit order, call log and lookup speed."""

import time

from aicore.catalog import FakeCatalog, index_key


def key(i: int, ph: str = "p" * 16) -> dict:
    return index_key("b" * 16, f"{i:064x}", None, None, ph)


def test_lookup_prefers_the_asking_apps_entry_then_other_owners_by_name():
    cat = FakeCatalog()
    cat.register_app("bench", "http://bench")
    cat.index_record("zeta", [{**key(1), "output_id": 1}, {**key(2), "output_id": 2}])
    cat.index_record("bench", [{**key(1), "output_id": 3}])
    cat.index_record("alpha", [{**key(2), "output_id": 4}])
    hits = cat.index_lookup("bench", [key(1), key(2), key(3), key(1, "q" * 16)])
    assert hits == [{"owner": "bench", "output_id": 3, "owner_url": "http://bench"},
                    {"owner": "alpha", "output_id": 4, "owner_url": None}, None, None]
    assert cat.index_lookup("label", [key(1)]) == [{"owner": "bench", "output_id": 3, "owner_url": "http://bench"}]
    hits[0]["owner"] = "changed"  # hits are copies
    assert cat.index_lookup("bench", [key(1)])[0]["owner"] == "bench"
    assert [c for c in cat.calls if c[0] == "index_lookup"] == [
        ("index_lookup", ("bench", 4)), ("index_lookup", ("label", 1)), ("index_lookup", ("bench", 1))]
    assert all(isinstance(k, tuple) and k[0][0] == "b" * 16 for k in cat._index)  # ((bk, sha, s, e, ph), app)


def test_lookup_of_many_keys_is_not_quadratic():
    cat = FakeCatalog()
    keys = [key(i) for i in range(5000)]
    cat.index_record("bench", [{**k, "output_id": i} for i, k in enumerate(keys)])
    t = time.perf_counter()
    hits = cat.index_lookup("label", keys)
    assert time.perf_counter() - t < 1.0  # 2.7 s when every key scanned the whole index
    assert [h["output_id"] for h in hits] == list(range(5000))
