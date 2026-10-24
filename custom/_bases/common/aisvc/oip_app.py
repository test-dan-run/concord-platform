"""`oip-v2` contract: KServe Open Inference Protocol V2 (REST, JSON tensor encoding).

Spec followed: https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol
(the Open Inference Protocol, also published at github.com/kserve/open-inference-protocol).

    handler(inputs: dict[str, np.ndarray], parameters: dict) -> dict[str, np.ndarray]

Inputs are validated against `signature` (names, datatypes, shapes with -1 wildcards) and decoded
to numpy arrays shaped as sent; outputs are cast to the declared datatypes and encoded back to
row-major JSON. Errors are {"error": "<message>"}. Raise ValueError in the handler for a 400.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from ._core import LoadFn, create_base_app

NUMPY_DTYPES: dict[str, Any] = {
    "BOOL": np.bool_,
    "UINT8": np.uint8, "UINT16": np.uint16, "UINT32": np.uint32, "UINT64": np.uint64,
    "INT8": np.int8, "INT16": np.int16, "INT32": np.int32, "INT64": np.int64,
    "FP16": np.float16, "FP32": np.float32, "FP64": np.float64,
    "BYTES": np.object_,
}


class OIPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass(frozen=True)
class Tensor:
    name: str
    datatype: str
    shape: tuple[int, ...]

    @classmethod
    def of(cls, spec: Any) -> Tensor:
        get = spec.get if isinstance(spec, Mapping) else lambda k: getattr(spec, k)
        datatype = get("datatype")
        if datatype not in NUMPY_DTYPES:
            raise ValueError(f"unknown OIP datatype {datatype!r}")
        return cls(get("name"), datatype, tuple(int(d) for d in get("shape")))

    def meta(self) -> dict:
        return {"name": self.name, "datatype": self.datatype, "shape": list(self.shape)}


def shape_matches(declared: Sequence[int], actual: Sequence[int]) -> bool:
    return len(declared) == len(actual) and all(d == -1 or d == a for d, a in zip(declared, actual))


def parse_signature(signature: Any) -> tuple[list[Tensor], list[Tensor]]:
    get = signature.get if isinstance(signature, Mapping) else lambda k: getattr(signature, k)
    return [Tensor.of(t) for t in get("inputs")], [Tensor.of(t) for t in get("outputs")]


def decode_tensor(spec: Tensor, req: RequestInput) -> np.ndarray:
    if req.datatype != spec.datatype:
        raise OIPError(400, f"input {req.name!r}: datatype {req.datatype} does not match signature {spec.datatype}")
    if not shape_matches(spec.shape, req.shape):
        raise OIPError(400, f"input {req.name!r}: shape {req.shape} does not match signature {list(spec.shape)}")
    try:
        if spec.datatype == "BYTES":
            arr = np.array(_flatten(req.data), dtype=object)
        else:
            arr = np.asarray(req.data).reshape(-1)
            if spec.datatype == "BOOL" and arr.dtype != np.bool_:
                raise ValueError("BOOL data must be true/false")
            if spec.datatype != "BOOL" and arr.dtype.kind not in "iuf":
                raise ValueError(f"{spec.datatype} data must be numeric")
            if arr.dtype.kind == "f" and NUMPY_DTYPES[spec.datatype]().dtype.kind in "iu":
                raise ValueError(f"{spec.datatype} data must be integers")
            arr = arr.astype(NUMPY_DTYPES[spec.datatype])
    except (ValueError, TypeError) as exc:
        raise OIPError(400, f"input {req.name!r}: {exc}") from None
    expected = math.prod(req.shape)
    if arr.size != expected:
        raise OIPError(400, f"input {req.name!r}: {arr.size} values do not fill shape {req.shape} ({expected} expected)")
    return arr.reshape(req.shape)


def encode_tensor(spec: Tensor, value: Any) -> dict:
    arr = np.asarray(value)
    if spec.datatype == "BYTES":
        data = [v.decode() if isinstance(v, bytes) else str(v) for v in arr.reshape(-1).tolist()]
    else:
        data = arr.astype(NUMPY_DTYPES[spec.datatype]).reshape(-1).tolist()
    if not shape_matches(spec.shape, arr.shape):
        raise OIPError(500, f"output {spec.name!r}: handler returned shape {list(arr.shape)}, signature declares {list(spec.shape)}")
    return {"name": spec.name, "datatype": spec.datatype, "shape": list(arr.shape), "data": data}


def _flatten(data: Any) -> list:
    if isinstance(data, list):
        return [x for item in data for x in _flatten(item)]
    return [data]


class RequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    shape: list[int]
    datatype: str
    parameters: dict[str, Any] | None = None
    data: Any


class RequestOutput(BaseModel):
    name: str
    parameters: dict[str, Any] | None = None


class InferenceRequest(BaseModel):
    id: str | None = None
    parameters: dict[str, Any] | None = None
    inputs: list[RequestInput] = Field(min_length=1)
    outputs: list[RequestOutput] | None = None


def create_oip_app(name: str, signature: Any, handler: Callable[[dict[str, np.ndarray], dict], Mapping[str, Any]],
                   version: str = "1", *, platform: str = "python", load: LoadFn | None = None,
                   background_load: bool = True) -> FastAPI:
    """App serving every OIP v2 REST route (plus /health) for one model."""
    inputs, outputs = parse_signature(signature)
    in_specs = {t.name: t for t in inputs}
    out_specs = {t.name: t for t in outputs}

    app = create_base_app(name, title=f"{name} (oip-v2)", load=load, background=background_load)
    state = app.state.svc

    @app.exception_handler(OIPError)
    async def _oip(_: Request, exc: OIPError):
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body') or 'body'}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"error": "; ".join(parts) or "invalid request"}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    def check_model(model: str, ver: str | None) -> None:
        if model != name:
            raise OIPError(404, f"model {model!r} not found; this server serves {name!r}")
        if ver is not None and ver != version:
            raise OIPError(404, f"version {ver!r} of model {name!r} not found (available: {version})")

    # -- health / metadata
    @app.get("/v2/health/live")
    def live() -> dict:
        return {"live": True}

    @app.get("/v2/health/ready")
    def ready() -> JSONResponse:
        return JSONResponse({"ready": state.ready}, status_code=200 if state.ready else 503)

    @app.get("/v2")
    def server_metadata() -> dict:
        return {"name": "aisvc", "version": __version__, "extensions": []}

    def model_metadata(model: str, ver: str | None = None) -> dict:
        check_model(model, ver)
        return {"name": name, "versions": [version], "platform": platform,
                "inputs": [t.meta() for t in inputs], "outputs": [t.meta() for t in outputs]}

    def model_ready(model: str, ver: str | None = None) -> JSONResponse:
        check_model(model, ver)
        return JSONResponse({"name": name, "ready": state.ready}, status_code=200 if state.ready else 503)

    # -- inference
    def infer(model: str, body: InferenceRequest, ver: str | None = None) -> dict:
        check_model(model, ver)
        if not state.ready:
            raise OIPError(503, f"model {name!r} is not ready ({state.error or 'loading'})")
        seen: dict[str, np.ndarray] = {}
        for inp in body.inputs:
            spec = in_specs.get(inp.name)
            if spec is None:
                raise OIPError(400, f"unexpected input {inp.name!r}; expected {sorted(in_specs)}")
            if inp.name in seen:
                raise OIPError(400, f"input {inp.name!r} given twice")
            seen[inp.name] = decode_tensor(spec, inp)
        missing = sorted(set(in_specs) - set(seen))
        if missing:
            raise OIPError(400, f"missing inputs {missing}")
        wanted = [o.name for o in body.outputs] if body.outputs else list(out_specs)
        unknown = sorted(set(wanted) - set(out_specs))
        if unknown:
            raise OIPError(400, f"unknown outputs requested {unknown}; available {sorted(out_specs)}")

        try:
            result = handler(seen, dict(body.parameters or {}))
        except OIPError:
            raise
        except ValueError as exc:
            raise OIPError(400, str(exc)) from None
        except Exception as exc:
            raise OIPError(500, f"{type(exc).__name__}: {exc}") from None
        for out in wanted:
            if out not in result:
                raise OIPError(500, f"handler did not produce output {out!r}")
        response = {"model_name": name, "model_version": version,
                    "outputs": [encode_tensor(out_specs[o], result[o]) for o in wanted]}
        if body.id is not None:
            response["id"] = body.id
        return response

    app.get("/v2/models/{model}")(model_metadata)
    app.get("/v2/models/{model}/versions/{ver}")(model_metadata)
    app.get("/v2/models/{model}/ready")(model_ready)
    app.get("/v2/models/{model}/versions/{ver}/ready")(model_ready)
    app.post("/v2/models/{model}/infer")(infer)
    app.post("/v2/models/{model}/versions/{ver}/infer")(infer)
    return app
