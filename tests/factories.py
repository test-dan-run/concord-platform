"""Manifest factories and a temp services/ directory helper for tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.schema import load_platform


def base_engine_manifest(**overrides) -> dict:
    m = {
        "name": "test-llm", "kind": "model", "engine": "vllm", "contract": "openai", "api": "chat.completions",
        "model": "/models/test-llm", "port": 8290, "gpu": {"util": 0.2},
        "tasks": ["chat"],
    }
    m.update(overrides)
    return {k: v for k, v in m.items() if v is not None}


def base_utility_manifest(**overrides) -> dict:
    m = {
        "name": "text-normalizer", "kind": "utility", "engine": "custom", "contract": "openai-inspired",
        "api": "text.transform", "contract_reason": "text in/out, no OpenAI route", "port": 8590,
        "tasks": ["text-normalization"],
        "build": {"context": "custom/text-normalizer", "image": "{registry}/x:0.1.0"},
    }
    m.update(overrides)
    return {k: v for k, v in m.items() if v is not None}


class ServicesDir:
    """A temp copy of services/ that tests can add manifests to or edit."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, data: dict, filename: str | None = None) -> Path:
        f = self.path / (filename or f"{data['name']}.yaml")
        f.write_text(yaml.safe_dump(data, sort_keys=False))
        return f

    def edit(self, name: str, **changes) -> Path:
        f = self.path / f"{name}.yaml"
        data = yaml.safe_load(f.read_text())
        for k, v in changes.items():
            if v is None:
                data.pop(k, None)
            else:
                data[k] = v
        return self.write(data, f.name)

    def load(self):
        return load_platform(services_dir=self.path)


# ----------------------------------------------------------------------------- a toy text.itn service

ITN_WORDS = {"one": "1", "two": "2", "three": "3", "twelve": "12", "twenty": "20", "thirty": "30"}
ITN_TENS = {"twenty", "thirty"}
ITN_UNITS = {"one", "two", "three"}


def toy_itn(texts: list[str], options) -> list[tuple[str, list[dict]]]:
    """Number words to digits ("twenty one" -> "21"), with CARDINAL spans in code points."""
    convert = options.class_list is None or "CARDINAL" in options.class_list
    results = []
    for text in texts:
        words = text.split(" ")
        out, spans, pos, k = "", [], 0, 0
        while k < len(words):
            if k:
                out += " "
                pos += 1
            word = words[k]
            digits = ITN_WORDS.get(word) if convert else None
            width = len(word)
            if digits and word in ITN_TENS and k + 1 < len(words) and words[k + 1] in ITN_UNITS:
                digits = digits[0] + ITN_WORDS[words[k + 1]]
                width += 1 + len(words[k + 1])
                k += 1
            if digits:
                spans.append({"input_start": pos, "input_end": pos + width, "output_start": len(out),
                              "output_end": len(out) + len(digits), "class": "CARDINAL"})
            out += digits or text[pos:pos + width]
            pos += width
            k += 1
        results.append((out, spans))
    return results


def toy_itn_app(handler=toy_itn):
    from aisvc import ItnOptions, create_openai_inspired_app
    from pydantic import field_validator

    class ToyOptions(ItnOptions):
        @field_validator("language")
        @classmethod
        def _language(cls, v):
            if v not in (None, "en-SG", "en-US"):
                raise ValueError(f"unsupported language {v!r}; supported: en-SG, en-US")
            return v

        @field_validator("classes")
        @classmethod
        def _classes(cls, v):
            unknown = [c.strip() for c in (v or "").split(",") if c.strip() and c.strip() != "CARDINAL"]
            if unknown:
                raise ValueError(f"unsupported classes {unknown}; supported: CARDINAL")
            return v

    return create_openai_inspired_app("itn-toy", "text.itn", handler, options_model=ToyOptions)
