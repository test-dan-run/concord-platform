"""Client for `openai-inspired` services (the text.transform and text.itn API types)."""

from __future__ import annotations

from typing import Any, overload

import httpx

from .endpoints import AicoreError


class ServiceError(AicoreError):
    """An OpenAI-shaped error returned by an openai-inspired service."""

    def __init__(self, status: int, message: str, type: str | None = None, code: str | None = None):
        super().__init__(f"{status} {type or 'error'}: {message}")
        self.status, self.message, self.type, self.code = status, message, type, code


class TextServiceClient:
    def __init__(self, base_url: str, model: str, api: str = "text.transform", *,
                 http: httpx.Client | None = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")  # root URL, without /v1
        self.model = model
        self.api = api
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self._http.post(path, json=body)
        if r.status_code >= 400:
            try:
                err = r.json()["error"]
                raise ServiceError(r.status_code, err.get("message", r.text), err.get("type"), err.get("code"))
            except (ValueError, KeyError, TypeError, AttributeError):
                raise ServiceError(r.status_code, r.text) from None
        return r.json()

    def models(self) -> list[str]:
        return [m["id"] for m in self._http.get("/v1/models").json()["data"]]

    @overload
    def transform(self, input: str, **options: Any) -> str: ...
    @overload
    def transform(self, input: list[str], **options: Any) -> list[str]: ...

    def transform(self, input: str | list[str], **options: Any) -> str | list[str]:
        """POST /v1/text/transform. A string returns a string; a list returns a list in order."""
        if self.api != "text.transform":
            raise AicoreError(f"service {self.model!r} implements {self.api!r}, not text.transform")
        data = self._post("/v1/text/transform", {"model": self.model, "input": input, "options": options})
        outputs = [d["output"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return outputs[0] if isinstance(input, str) else outputs

    def itn(self, input: str | list[str], *, classes: str | list[str] | None = None, return_spans: bool = False,
            **options: Any) -> Any:
        """POST /v1/text/itn. Without return_spans: a string returns a string, a list a list in order.
        With return_spans: the `data` item (output + spans), or a list of them. `classes` may be a list."""
        if self.api != "text.itn":
            raise AicoreError(f"service {self.model!r} implements {self.api!r}, not text.itn")
        if classes is not None:
            options["classes"] = classes if isinstance(classes, str) else ",".join(classes)
        if return_spans:
            options["return_spans"] = True
        data = self._post("/v1/text/itn", {"model": self.model, "input": input, "options": options})
        items = sorted(data["data"], key=lambda d: d["index"])
        results = items if return_spans else [d["output"] for d in items]
        return results[0] if isinstance(input, str) else results

    def close(self) -> None:
        self._http.close()
