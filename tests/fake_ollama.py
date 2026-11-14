"""
In-process fake of the Ollama HTTP API used by the unit tests.

It replaces requests.get / requests.post, so code under test runs unchanged and
no test ever needs (or reaches) a running Ollama server. Responses follow the
shape of the real endpoints closely enough for the code paths we exercise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

import requests


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload: dict | None, status_code: int = 200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} error")
            err.response = self
            raise err


def default_model(name: str, size: int, quant: str = "Q4_K_M", params: str = "1.5B") -> dict:
    """One /api/tags entry."""
    return {
        "name": name,
        "model": name,
        "size": size,
        "digest": ("%064x" % abs(hash(name)))[:64],
        "modified_at": "2026-11-01T10:00:00+01:00",
        "details": {
            "format": "gguf",
            "family": name.split(":")[0],
            "parameter_size": params,
            "quantization_level": quant,
        },
    }


@dataclass
class FakeOllama:
    """
    Configurable fake server.

    responder(model, prompt, payload) -> str is the text returned by /api/generate;
    by default every message is answered {"label": "valid"}.
    """

    models: list[dict] = field(
        default_factory=lambda: [
            default_model("qwen2.5:1.5b", 986_000_000),
            default_model("qwen2.5:7b", 4_700_000_000, params="7.6B"),
            default_model("llama3.2:latest", 2_019_000_000, params="3.2B"),
        ]
    )
    responder: Callable[[str, str, dict], str] | None = None
    version: str = "0.40.13"
    online: bool = True
    capabilities: list[str] = field(default_factory=lambda: ["completion"])
    calls: list[tuple[str, str, dict | None]] = field(default_factory=list)
    eval_count: int = 6
    prompt_eval_count: int = 900

    # ── routing ──
    def get(self, url: str, *args, **kwargs) -> FakeResponse:
        return self._route("GET", url, None)

    def post(self, url: str, *args, json: dict | None = None, **kwargs) -> FakeResponse:  # noqa: A002
        return self._route("POST", url, json)

    def _route(self, method: str, url: str, body: dict | None) -> FakeResponse:
        self.calls.append((method, url, body))
        if not self.online:
            raise requests.ConnectionError("fake Ollama is offline")
        path = urlparse(url).path
        if path == "/api/tags":
            return FakeResponse({"models": self.models})
        if path == "/api/version":
            return FakeResponse({"version": self.version})
        if path == "/api/ps":
            return FakeResponse({"models": []})
        if path == "/api/show":
            return FakeResponse(self._show(body or {}))
        if path == "/api/generate":
            return FakeResponse(self._generate(body or {}))
        return FakeResponse({"error": f"unknown path {path}"}, status_code=404)

    def _show(self, body: dict) -> dict:
        name = body.get("model") or body.get("name")
        model = next((m for m in self.models if m["name"] == name), None)
        if model is None:
            return {"error": "model not found"}
        family = model["details"]["family"]
        return {
            "details": model["details"],
            "capabilities": self.capabilities,
            "model_info": {f"{family}.context_length": 32768, "general.parameter_count": 1_543_714_304},
        }

    def _generate(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        model = body.get("model", "")
        if self.responder is not None:
            text = self.responder(model, prompt, body)
        else:
            text = '{"label": "valid"}'
        return {
            "model": model,
            "response": text,
            "done": True,
            "done_reason": "stop",
            "total_duration": 1_200_000_000,
            "load_duration": 5_000_000,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": 900_000_000,
            "eval_count": self.eval_count,
            "eval_duration": 250_000_000,
        }

    # ── helpers for assertions ──
    def generate_calls(self) -> list[dict]:
        return [body for method, url, body in self.calls if url.endswith("/api/generate")]
