"""Client for `oip-v2` services (KServe Open Inference Protocol V2, JSON tensors). Needs numpy."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without the extra
    raise ImportError("OIPClient needs numpy. Install the extra: uv pip install 'aicore[oip]'") from None

import httpx

from .endpoints import AicoreError

NP_TO_OIP = {
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.uint8): "UINT8", np.dtype(np.uint16): "UINT16", np.dtype(np.uint32): "UINT32", np.dtype(np.uint64): "UINT64",
    np.dtype(np.int8): "INT8", np.dtype(np.int16): "INT16", np.dtype(np.int32): "INT32", np.dtype(np.int64): "INT64",
    np.dtype(np.float16): "FP16", np.dtype(np.float32): "FP32", np.dtype(np.float64): "FP64",
}
OIP_TO_NP = {v: k for k, v in NP_TO_OIP.items()} | {"BYTES": np.dtype(object)}


class OIPError(AicoreError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


def oip_datatype(arr: np.ndarray) -> str:
    if arr.dtype.kind in "OUS":
        return "BYTES"
    try:
        return NP_TO_OIP[arr.dtype]
    except KeyError:
        raise AicoreError(f"numpy dtype {arr.dtype} has no OIP datatype") from None


def _shape_ok(declared: list[int], actual: tuple[int, ...]) -> bool:
    return len(declared) == len(actual) and all(d == -1 or d == a for d, a in zip(declared, actual))


class OIPClient:
    def __init__(self, base_url: str, model: str, signature: dict[str, Any] | None = None, *,
                 version: str | None = None, http: httpx.Client | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")  # root URL, without /v2
        self.model = model
        self.signature = signature
        self.version = version
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    @property
    def _model_path(self) -> str:
        path = f"/v2/models/{self.model}"
        return f"{path}/versions/{self.version}" if self.version else path

    def _check(self, r: httpx.Response) -> dict[str, Any]:
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]
            except (ValueError, KeyError, TypeError):
                msg = r.text
            raise OIPError(r.status_code, msg)
        return r.json()

    # -- protocol endpoints
    def live(self) -> bool:
        return self._http.get("/v2/health/live").status_code == 200

    def ready(self) -> bool:
        return self._http.get(f"{self._model_path}/ready").status_code == 200

    def server_metadata(self) -> dict[str, Any]:
        return self._check(self._http.get("/v2"))

    def metadata(self) -> dict[str, Any]:
        return self._check(self._http.get(self._model_path))

    # -- inference
    def _signature(self) -> dict[str, Any]:
        if self.signature is None:
            meta = self.metadata()
            self.signature = {"inputs": meta["inputs"], "outputs": meta["outputs"]}
        return self.signature

    def encode(self, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate numpy inputs against the signature and encode them as OIP JSON tensors."""
        specs = {t["name"]: t for t in self._signature()["inputs"]}
        unknown = sorted(set(inputs) - set(specs))
        missing = sorted(set(specs) - set(inputs))
        if unknown or missing:
            raise AicoreError(f"inputs do not match signature: unknown {unknown}, missing {missing}; expected {sorted(specs)}")
        tensors = []
        for name in sorted(inputs):
            spec, arr = specs[name], np.asarray(inputs[name])
            dt = oip_datatype(arr)
            if dt != spec["datatype"]:
                raise AicoreError(f"input {name!r}: dtype {arr.dtype} ({dt}) but signature needs {spec['datatype']}; "
                                  f"use .astype({OIP_TO_NP[spec['datatype']]})")
            if not _shape_ok(spec["shape"], arr.shape):
                raise AicoreError(f"input {name!r}: shape {list(arr.shape)} does not match signature {spec['shape']}")
            flat = arr.reshape(-1)
            data = [x.decode() if isinstance(x, bytes) else str(x) for x in flat.tolist()] if dt == "BYTES" else flat.tolist()
            tensors.append({"name": name, "datatype": dt, "shape": list(arr.shape), "data": data})
        return tensors

    @staticmethod
    def decode(outputs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        result = {}
        for out in outputs:
            dtype = OIP_TO_NP[out["datatype"]]
            result[out["name"]] = np.asarray(out["data"], dtype=dtype).reshape(out["shape"])
        return result

    def infer(self, inputs: dict[str, Any], parameters: dict[str, Any] | None = None,
              outputs: list[str] | None = None, request_id: str | None = None) -> dict[str, np.ndarray]:
        """Run inference; returns {output name: numpy array} with the declared datatypes and returned shapes."""
        body: dict[str, Any] = {"inputs": self.encode(inputs)}
        if parameters:
            body["parameters"] = parameters
        if outputs:
            body["outputs"] = [{"name": n} for n in outputs]
        if request_id:
            body["id"] = request_id
        return self.decode(self._check(self._http.post(f"{self._model_path}/infer", json=body))["outputs"])

    def close(self) -> None:
        self._http.close()
