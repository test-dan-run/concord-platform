"""`openai` contract: OpenAI-compatible routes for custom servers.

Handlers receive the parsed request model and return plain Python values:

    chat.completions      handler(ChatCompletionRequest)  -> str | dict(content=..., finish_reason=..., usage=...)
    completions           handler(CompletionRequest)      -> str | list[str]   (one per prompt)
    embeddings            handler(EmbeddingRequest)       -> list[list[float]] (one per input)
    audio.transcriptions  handler(TranscriptionRequest)   -> str | dict(text=..., language=..., duration=..., segments=..., words=...)
    audio.speech          handler(SpeechRequest)          -> (samples: np.ndarray float in [-1, 1], sample_rate)

Raise ValueError for bad input (400). Streaming is not implemented by the factory.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ._core import LoadFn, create_base_app
from .audio import CONTENT_TYPES, encode_audio

OPENAI_APIS = ("chat.completions", "completions", "embeddings", "audio.transcriptions", "audio.speech")
ROUTES = {
    "chat.completions": "/v1/chat/completions",
    "completions": "/v1/completions",
    "embeddings": "/v1/embeddings",
    "audio.transcriptions": "/v1/audio/transcriptions",
    "audio.speech": "/v1/audio/speech",
}


class OpenAIError(Exception):
    """Rendered as {"error": {"message", "type", "code"}} (OpenAI error shape)."""

    def __init__(self, status: int, message: str, type: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code = status, message, type, code

    def response(self) -> JSONResponse:
        return JSONResponse({"error": {"message": self.message, "type": self.type, "code": self.code}}, status_code=self.status)


def install_openai_errors(app: FastAPI) -> None:
    """OpenAI-shaped errors for our exceptions, validation failures and unknown routes."""

    @app.exception_handler(OpenAIError)
    async def _openai(_: Request, exc: OpenAIError):
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body') or 'body'}: {e['msg']}" for e in exc.errors()]
        return OpenAIError(400, "; ".join(parts) or "invalid request", code="invalid_request").response()

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else None
        return OpenAIError(exc.status_code, str(exc.detail), code=code).response()


def guard(app: FastAPI, name: str, model: str | None, call: Callable[[], Any], *,
          shape: Callable[[Any], Any] | None = None) -> Any:
    """Common pre/post handling: readiness, model name, handler errors, and `shape` (the handler's result -> the
    response body), whose errors are the server's: a 500 in the OpenAI shape, also for a ValueError."""
    state = app.state.svc
    if not state.ready:
        raise OpenAIError(503, f"{name} is not ready ({state.error or 'loading'})", type="service_unavailable", code="not_ready")
    if model is not None and model != name:
        raise OpenAIError(404, f"The model {model!r} does not exist; this service serves {name!r}", code="model_not_found")
    try:
        out = call()
    except OpenAIError:
        raise
    except ValueError as exc:
        raise OpenAIError(400, str(exc), code="invalid_request") from None
    except Exception as exc:
        raise OpenAIError(500, f"{type(exc).__name__}: {exc}", type="server_error") from None
    if shape is None:
        return out
    try:
        return shape(out)
    except Exception as exc:
        raise OpenAIError(500, f"{type(exc).__name__}: {exc}", type="server_error") from None


def models_body(name: str, created: int) -> dict:
    return {"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "aiplatform"}]}


# ----------------------------------------------------------------------------- request models

class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerate OpenAI params we do not use


class ChatMessage(_Req):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(_Req):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1, le=1)
    stream: Literal[False] = False


class CompletionRequest(_Req):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: Literal[False] = False


class EmbeddingRequest(_Req):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = None

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class SpeechRequest(_Req):
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str | None = None


@dataclass
class TranscriptionRequest:
    model: str
    file: bytes
    filename: str
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float | None = None
    timestamp_granularities: list[str] = field(default_factory=list)  # "segment" and/or "word" (verbose_json)


# ----------------------------------------------------------------------------- factory

def create_openai_app(name: str, api: str, handler: Callable[..., Any], *, load: LoadFn | None = None,
                      background_load: bool = True) -> FastAPI:
    """App serving GET /v1/models, GET /health and the OpenAI route for `api`."""
    if api not in OPENAI_APIS:
        raise ValueError(f"unsupported openai api {api!r}; expected one of {OPENAI_APIS}")
    app = create_base_app(name, title=f"{name} ({api}, openai)", load=load, background=background_load)
    install_openai_errors(app)
    created = int(time.time())

    @app.get("/v1/models")
    def list_models() -> dict:
        return models_body(name, created)

    if api == "chat.completions":

        @app.post(ROUTES[api])
        def chat(req: ChatCompletionRequest) -> dict:
            def shape(out: Any) -> dict:
                out = {"content": out} if isinstance(out, str) else dict(out)
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                    "model": name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out["content"]},
                                 "finish_reason": out.get("finish_reason", "stop")}],
                    "usage": out.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                }

            return guard(app, name, req.model, lambda: handler(req), shape=shape)

    elif api == "completions":

        @app.post(ROUTES[api])
        def completions(req: CompletionRequest) -> dict:
            def shape(out: Any) -> dict:
                texts = [out] if isinstance(out, str) else list(out)
                return {
                    "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()),
                    "model": name,
                    "choices": [{"index": i, "text": t, "finish_reason": "stop", "logprobs": None} for i, t in enumerate(texts)],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

            return guard(app, name, req.model, lambda: handler(req), shape=shape)

    elif api == "embeddings":

        @app.post(ROUTES[api])
        def embeddings(req: EmbeddingRequest) -> dict:
            def shape(vectors: Any) -> dict:
                return {
                    "object": "list", "model": name,
                    "data": [{"object": "embedding", "index": i, "embedding": [float(x) for x in v]}
                             for i, v in enumerate(vectors)],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }

            return guard(app, name, req.model, lambda: handler(req), shape=shape)

    elif api == "audio.transcriptions":

        @app.post(ROUTES[api])
        def transcriptions(
            file: UploadFile = File(...),
            model: str = Form(...),
            language: str | None = Form(None),
            prompt: str | None = Form(None),
            response_format: Literal["json", "text", "verbose_json"] = Form("json"),
            temperature: float | None = Form(None),
            timestamp_granularities: list[Literal["segment", "word"]] | None = Form(None, alias="timestamp_granularities[]"),
        ) -> Response:
            req = TranscriptionRequest(model, file.file.read(), file.filename or "audio", language, prompt,
                                       response_format, temperature, list(timestamp_granularities or []))

            def shape(out: Any) -> Response:
                out = {"text": out} if isinstance(out, str) else dict(out)
                if response_format == "text":
                    return Response(out["text"], media_type="text/plain")
                if response_format == "json":
                    out = {"text": out["text"]}
                return JSONResponse(out)

            return guard(app, name, model, lambda: handler(req), shape=shape)

    elif api == "audio.speech":

        @app.post(ROUTES[api])
        def speech(req: SpeechRequest) -> Response:
            def run():
                samples, sample_rate = handler(req)
                return encode_audio(samples, sample_rate, req.response_format)

            body = guard(app, name, req.model, run)
            return Response(body, media_type=CONTENT_TYPES[req.response_format])

    return app
