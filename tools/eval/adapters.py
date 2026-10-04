"""HTTP adapters for Android debug APIs and other RAG applications."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from common import call_evaluate, call_health, call_upload_file

REQUEST_FIELDS = {
    "case_id": "case_id", "question": "question", "metadata": "metadata",
    "top_k": "top_k", "model_id": "model_id", "retrieval_only": "retrieval_only",
}
RESPONSE_FIELDS = {
    key: key for key in (
        "generated_answer", "retrieved_chunks", "retrieval_timing_ms",
        "generation_timing_ms", "total_latency_ms", "runtime", "decoding_metrics", "error",
    )
}


def _get(data: Any, path: str, default: Any = None) -> Any:
    for key in path.split("."):
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def _set(data: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        child = data.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Request field {path!r} conflicts with another field.")
        data = child
    data[keys[-1]] = value


def _field_map(value: Any, allowed: dict, name: str) -> dict:
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{name} must map supported field names: {', '.join(allowed)}")
    for key, path in value.items():
        if path is not None and (not isinstance(path, str) or not path or any(not p for p in path.split("."))):
            raise ValueError(f"{name}.{key} must be a dotted field path or null.")
    return {**allowed, **value}


def _endpoint(value: Any, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//") or urlsplit(value).fragment:
        raise ValueError(f"{name} must be a relative API path starting with /.")
    return value


@dataclass(frozen=True)
class AdapterProfile:
    name: str = "generic-android"
    evaluate_path: str = "/evaluate"
    health_path: str | None = None
    health_expect: dict[str, Any] = field(default_factory=dict)
    request_fields: dict[str, str | None] = field(default_factory=lambda: dict(REQUEST_FIELDS))
    response_fields: dict[str, str | None] = field(default_factory=lambda: dict(RESPONSE_FIELDS))
    chunk_fields: dict[str, str | None] = field(default_factory=dict)
    extra_payload: dict[str, Any] = field(default_factory=dict)
    headers_env: dict[str, str] = field(default_factory=dict)
    judge_domain: str = "general"
    # Corpus upload formats differ between apps; only BRITE currently implements it.
    brite_compatibility: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "AdapterProfile":
        if not isinstance(data, dict):
            raise ValueError("Adapter profile must be a JSON object.")
        allowed = set(cls.__dataclass_fields__) - {"brite_compatibility"}
        if set(data) - allowed:
            raise ValueError(f"Unknown profile fields: {', '.join(sorted(set(data) - allowed))}")
        values = dict(data)
        for key in ("name", "judge_domain"):
            if key in values and (not isinstance(values[key], str) or not values[key].strip()):
                raise ValueError(f"{key} must be a non-empty string.")
        for key in ("health_expect", "extra_payload", "headers_env"):
            if key in values and not isinstance(values[key], dict):
                raise ValueError(f"{key} must be an object.")
        if "evaluate_path" in values:
            values["evaluate_path"] = _endpoint(values["evaluate_path"], "evaluate_path")
        if "health_path" in values:
            values["health_path"] = _endpoint(values["health_path"], "health_path", optional=True)
        for key, default in (("request_fields", REQUEST_FIELDS), ("response_fields", RESPONSE_FIELDS),
                             ("chunk_fields", {"text": None, "source": None, "score": None})):
            if key in values:
                values[key] = _field_map(values[key], default, key)
        if values.get("request_fields", REQUEST_FIELDS).get("question") is None:
            raise ValueError("request_fields.question is required.")
        if values.get("response_fields", RESPONSE_FIELDS).get("retrieved_chunks") is None:
            raise ValueError("response_fields.retrieved_chunks is required for RAG evaluation.")
        for header, env in values.get("headers_env", {}).items():
            if not isinstance(header, str) or not header or not isinstance(env, str) or not env:
                raise ValueError("headers_env must map header names to environment variable names.")
        result = cls(**values)
        if result.health_expect and not result.health_path:
            raise ValueError("health_expect requires health_path.")
        return result


BRITE_PROFILE = AdapterProfile(
    name="brite-android", health_path="/health",
    health_expect={"status": "ok", "runtime_ready": True},
    judge_domain="medical first-aid", brite_compatibility=True,
)


def load_profile(path: str | Path | None = None, adapter: str = "brite") -> AdapterProfile:
    if path:
        return AdapterProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    if adapter == "brite":
        return BRITE_PROFILE
    if adapter == "generic":
        return AdapterProfile()
    raise ValueError(f"Unknown adapter: {adapter}")


class HttpRagAdapter:
    def __init__(self, api_base: str, profile: AdapterProfile = BRITE_PROFILE,
                 health_call=None, evaluate_call=None, upload_call=None):
        if urlsplit(api_base).scheme not in {"http", "https"} or not urlsplit(api_base).hostname:
            raise ValueError("API base must be an http:// or https:// URL.")
        self.api_base = api_base.rstrip("/")
        self.profile = profile
        self._health_call = health_call or call_health
        self._evaluate_call = evaluate_call or call_evaluate
        self._upload_call = upload_call or call_upload_file

    def _headers(self) -> dict[str, str]:
        headers = {}
        for header, env in self.profile.headers_env.items():
            value = os.environ.get(env)
            if not value:
                raise ValueError(f"Missing environment variable {env} for API authentication.")
            headers[header] = value
        return headers

    def check_health(self) -> dict[str, Any]:
        if not self.profile.health_path:
            return {"status": "skipped"}
        if self.profile.brite_compatibility:
            body = self._health_call(self.api_base)
        else:
            response = requests.get(self.api_base + self.profile.health_path, headers=self._headers(), timeout=180)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Health endpoint must return a JSON object.")
        for path, expected in self.profile.health_expect.items():
            if _get(body, path, object()) != expected:
                raise RuntimeError(f"Health check failed: expected {path}={expected!r}.")
        return body

    def build_request(self, case: dict, top_k=None, model_id=None, retrieval_only=False) -> dict:
        values = {"case_id": case["case_id"], "question": case["question"],
                  "metadata": {"category": case.get("category") or "", "notes": case.get("notes") or ""}}
        if top_k is not None:
            if int(top_k) < 1:
                raise ValueError("top_k must be positive.")
            values["top_k"] = int(top_k)
        if model_id:
            values["model_id"] = model_id
        if retrieval_only:
            values["retrieval_only"] = True
        payload = deepcopy(self.profile.extra_payload)
        for key, value in values.items():
            path = self.profile.request_fields.get(key)
            if path:
                _set(payload, path, value)
            elif key in {"top_k", "model_id", "retrieval_only"}:
                raise ValueError(f"Adapter {self.profile.name!r} does not support {key}.")
        return payload

    def evaluate(self, payload: dict, timeout: int = 180, retrieval_only: bool = False) -> dict:
        if self.profile.brite_compatibility:
            body = self._evaluate_call(self.api_base, payload, timeout=timeout)
            status = 200
        else:
            response = requests.post(self.api_base + self.profile.evaluate_path, json=payload,
                                     headers=self._headers(), timeout=timeout)
            status = response.status_code
            try:
                body = response.json()
            except ValueError:
                return {"error": {"code": "invalid_response", "message": f"API returned non-JSON data (HTTP {status})."}}
        if not isinstance(body, dict):
            return {"error": {"code": "invalid_response", "message": "Evaluation endpoint must return a JSON object."}}
        result = {key: _get(body, path) for key, path in self.profile.response_fields.items() if path}
        error = result.get("error")
        if error or status >= 400:
            normalized_error = dict(error) if isinstance(error, dict) else {"message": str(error) if error else f"Evaluation endpoint returned HTTP {status}."}
            normalized_error.setdefault("code", f"http_{status}" if status >= 400 else "api_error")
            normalized_error.setdefault("message", "RAG API returned an error.")
            # Malformed optional diagnostics on an error response must not break
            # the dashboard's row builder or prevent subsequent cases from running.
            return {"error": normalized_error, "adapter_raw_response": body}
        chunks = result.get("retrieved_chunks")
        if not isinstance(chunks, list) or any(not isinstance(c, (dict, str)) for c in chunks):
            raise ValueError("Missing or invalid retrieved_chunks; check response_fields in the adapter profile.")
        if not retrieval_only and not isinstance(result.get("generated_answer"), str):
            raise ValueError("Missing generated_answer; check response_fields or enable retrieval-only mode.")
        result["retrieved_chunks"] = [self._normalize_chunk(chunk) for chunk in chunks]
        for key in ("runtime", "decoding_metrics"):
            if result.get(key) is not None and not isinstance(result[key], dict):
                raise ValueError(f"{key} must be a JSON object when supplied.")
        # Preserve additional BRITE fields and the original response for custom APIs.
        return {**body, **result} if self.profile.brite_compatibility else {**result, "adapter_raw_response": body}

    def _normalize_chunk(self, chunk: dict | str) -> dict:
        if isinstance(chunk, str):
            return {"text": chunk}
        result = dict(chunk)
        for target, path in self.profile.chunk_fields.items():
            if path:
                result[target] = _get(chunk, path)
        text = next((result[k] for k in ("text", "contentSnippet", "content_snippet", "snippet", "content", "body")
                     if isinstance(result.get(k), str) and result[k].strip()), None)
        if text is None:
            raise ValueError("Retrieved chunk has no text; configure chunk_fields.text.")
        result["text"] = text
        return result

    def upload_corpus(self, path: str | Path) -> dict:
        if not self.profile.brite_compatibility:
            raise ValueError("Corpus upload is supported by the BRITE adapter only. Index the other app's corpus first.")
        result = self._upload_call(self.api_base, path)
        if result.get("error"):
            raise RuntimeError(f"Corpus upload failed: {result['error']}")
        return result
