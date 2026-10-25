"""aisvc edge cases: empty integer and boolean tensors, out of memory backoff on items of length 0, and the error
code of a handler's ValueError."""

import sys

import numpy as np
import pytest
from aisvc import Limits, MicroBatcher, create_oip_app, create_openai_app, create_openai_inspired_app
from conformance import running

EMPTY_SIG = {"inputs": [{"name": "N", "datatype": "INT32", "shape": [-1]}, {"name": "M", "datatype": "BOOL", "shape": [-1]},
                        {"name": "U", "datatype": "UINT8", "shape": [-1, 2]}],
             "outputs": [{"name": "K", "datatype": "INT32", "shape": [1]}]}


def test_empty_int_and_bool_tensors_are_accepted():
    seen = {}

    def handler(x, p):
        seen.update(x)
        return {"K": np.array([sum(v.size for v in x.values())], np.int32)}

    body = {"inputs": [{"name": "N", "datatype": "INT32", "shape": [0], "data": []},
                       {"name": "M", "datatype": "BOOL", "shape": [0], "data": []},
                       {"name": "U", "datatype": "UINT8", "shape": [0, 2], "data": [[]]}]}
    with running(create_oip_app("m", EMPTY_SIG, handler)) as c:
        r = c.post("/v2/models/m/infer", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["outputs"][0]["data"] == [0]
    assert {k: (v.dtype, v.shape) for k, v in seen.items()} == {
        "N": (np.int32, (0,)), "M": (np.bool_, (0,)), "U": (np.uint8, (0, 2))}


@pytest.mark.parametrize("name, data, fragment", [("N", [0.5], "INT32 data must be integers"),
                                                  ("N", ["x"], "INT32 data must be numeric"),
                                                  ("M", [1], "BOOL data must be true/false")])
def test_non_empty_tensors_are_still_checked(name, data, fragment):
    inputs = {"N": [0], "M": [True], "U": [[1, 2]]}
    inputs[name] = data
    shapes = {"N": [len(inputs["N"])], "M": [len(inputs["M"])], "U": [1, 2]}
    body = {"inputs": [{"name": k, "datatype": t, "shape": shapes[k], "data": inputs[k]}
                       for k, t in (("N", "INT32"), ("M", "BOOL"), ("U", "UINT8"))]}
    with running(create_oip_app("m", EMPTY_SIG, lambda x, p: {"K": np.array([0], np.int32)})) as c:
        r = c.post("/v2/models/m/infer", json=body)
    assert r.status_code == 400 and fragment in r.json()["error"], r.text


def test_out_of_memory_backoff_splits_items_of_length_0(monkeypatch):
    """Empty clips are 0 ms long, so a chunk of them had padded length 0 and planning it again never split it."""
    monkeypatch.setitem(sys.modules, "torch", None)
    chunks = []

    def fn(batch):
        chunks.append(len(batch))
        if len(chunks) > 20:
            raise RuntimeError("the backoff never splits the chunk")
        if len(batch) > 2:
            raise MemoryError()
        return [x * 2 for x in batch]

    lim = Limits(max_items=8)
    assert MicroBatcher(fn, length=lambda x: 0, limits=lim)([1, 2, 3, 4, 5, 6]) == [2, 4, 6, 8, 10, 12]
    assert chunks == [6, 3, 1, 1, 1, 3, 1, 1, 1] and lim.max_len == 1  # each length 0 counts as 1: 6 -> 3 -> 1


def bad_input(*args):
    raise ValueError("input 2 is not text the service can read")


def test_a_handler_value_error_is_invalid_request_on_openai_inspired():
    with running(create_openai_inspired_app("n", "text.transform", bad_input)) as c:
        r = c.post("/v1/text/transform", json={"model": "n", "input": "x"})
    assert r.status_code == 400
    assert r.json()["error"] == {"message": "input 2 is not text the service can read", "type": "invalid_request_error",
                                 "code": "invalid_request"}


def test_a_handler_value_error_is_invalid_request_on_openai():
    with running(create_openai_app("e", "embeddings", bad_input)) as c:
        r = c.post("/v1/embeddings", json={"model": "e", "input": "x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
