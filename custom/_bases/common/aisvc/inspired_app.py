"""`openai-inspired` contract: text-input services with no OpenAI route.

Conventions (contracts/README.md): POST /v1/{resource}, JSON body {model, input: str | list[str],
options: {...}}, response {object: "list", model, created, data: [{object, index, ...}], usage},
OpenAI-shaped errors, GET /v1/models.

API types:
    text.transform   POST /v1/text/transform    handler(texts: list[str], options) -> list[str]
    text.itn         POST /v1/text/itn          handler(texts: list[str], options: ItnOptions)
                                                -> list[(output, spans)]

`options_model` (a pydantic model) validates `options`; invalid options are a 400. For text.itn it
must subclass `ItnOptions` (add a validator to reject unsupported languages or classes). A text.itn
handler always returns spans, each {input_start, input_end, output_start, output_end, class}; they are
checked and dropped from the response unless the request asked for them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ._core import LoadFn, create_base_app
from .openai_app import OpenAIError, guard, install_openai_errors, models_body

INSPIRED_APIS = {"text.transform": "/v1/text/transform", "text.itn": "/v1/text/itn"}
ITN_MAX_INPUTS = 256
ITN_MAX_CHARS = 10_000


class TextRequest(BaseModel):
    """Every openai-inspired API type takes text: a string or a list of strings, plus scalar options."""

    model_config = ConfigDict(extra="forbid")
    model: str
    input: str | list[str]
    options: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _non_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list) and not v:
            raise ValueError("input list must not be empty")
        return v

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class ItnOptions(BaseModel):
    """The options every text.itn service accepts (contracts/README.md, API type text.itn)."""

    model_config = ConfigDict(extra="forbid")
    language: str | None = None
    return_spans: bool = False
    classes: str | None = None

    @property
    def class_list(self) -> list[str] | None:
        """`classes` split on commas, or None for every class the service supports."""
        if self.classes is None:
            return None
        return [c.strip() for c in self.classes.split(",") if c.strip()]


def _checked_spans(spans: Sequence[dict[str, Any]], text: str, output: str) -> list[dict[str, Any]]:
    out, last = [], 0
    for s in spans:
        span = {k: s[k] for k in ("input_start", "input_end", "output_start", "output_end", "class")}
        ok = (last <= span["input_start"] <= span["input_end"] <= len(text)
              and 0 <= span["output_start"] <= span["output_end"] <= len(output) and span["class"])
        if not ok:
            raise RuntimeError(f"handler returned an invalid span {span} for {len(text)}-char input")
        last = span["input_end"]
        out.append(span)
    return out


def create_openai_inspired_app(name: str, api: str, handler: Callable[[list[str], Any], list[Any]], *,
                               options_model: type[BaseModel] | None = None, load: LoadFn | None = None,
                               background_load: bool = True) -> FastAPI:
    if api not in INSPIRED_APIS:
        raise ValueError(f"unknown openai-inspired api {api!r}; define it in contracts/README.md first (known: {list(INSPIRED_APIS)})")
    if api == "text.itn":
        options_model = options_model or ItnOptions
        if not issubclass(options_model, ItnOptions):
            raise ValueError("a text.itn options_model must subclass aisvc.ItnOptions")
    app = create_base_app(name, title=f"{name} ({api}, openai-inspired)", load=load, background=background_load)
    install_openai_errors(app)
    created = int(time.time())

    @app.get("/v1/models")
    def list_models() -> dict:
        return models_body(name, created)

    def parse_options(raw: dict) -> Any:
        if options_model is None:
            return raw
        try:
            return options_model.model_validate(raw)
        except ValidationError as exc:
            msg = "; ".join(f"options.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise OpenAIError(400, msg, code="invalid_options") from None

    def body(texts: list[str], items: list[dict], outputs: list[str]) -> dict:
        return {
            "object": "list",
            "model": name,
            "created": int(time.time()),
            "data": items,
            "usage": {"input_chars": sum(map(len, texts)), "output_chars": sum(map(len, outputs))},
        }

    if api == "text.transform":

        @app.post(INSPIRED_APIS[api])
        def transform(req: TextRequest) -> dict:
            texts = req.texts

            def run() -> list[str]:
                outputs = list(handler(texts, parse_options(req.options)))
                if len(outputs) != len(texts):
                    raise RuntimeError(f"handler returned {len(outputs)} outputs for {len(texts)} inputs")
                return outputs

            outputs = guard(app, name, req.model, run)
            items = [{"object": "text.transform", "index": i, "output": o} for i, o in enumerate(outputs)]
            return body(texts, items, outputs)

    if api == "text.itn":

        @app.post(INSPIRED_APIS[api])
        def itn(req: TextRequest) -> dict:
            texts = req.texts
            if len(texts) > ITN_MAX_INPUTS:
                raise OpenAIError(400, f"input has {len(texts)} items; at most {ITN_MAX_INPUTS}", code="invalid_request")
            if any(len(t) > ITN_MAX_CHARS for t in texts):
                raise OpenAIError(400, f"an input item is longer than {ITN_MAX_CHARS} characters", code="invalid_request")

            def run() -> tuple[ItnOptions, list[tuple[str, list[dict[str, Any]]]]]:
                options = parse_options(req.options)
                results = list(handler(texts, options))
                if len(results) != len(texts):
                    raise RuntimeError(f"handler returned {len(results)} outputs for {len(texts)} inputs")
                return options, [(out, _checked_spans(spans, text, out)) for (out, spans), text in zip(results, texts, strict=True)]

            options, results = guard(app, name, req.model, run)
            items = []
            for i, (out, spans) in enumerate(results):
                item: dict[str, Any] = {"object": "text.itn", "index": i, "output": out}
                if options.return_spans:
                    item["spans"] = spans
                items.append(item)
            return body(texts, items, [out for out, _ in results])

    return app
