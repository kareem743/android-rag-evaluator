from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlsplit
from typing import Any

import requests


DEFAULT_JUDGE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = ""
DEFAULT_JUDGE_API_KEY_ENV = "RAG_JUDGE_API_KEY"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 180

RETRIEVAL_EVAL_K = 5
SEMANTIC_PASS_THRESHOLD = 0.75

_RETRIEVAL_VERDICTS = {"pass", "medium", "fail"}
_ANSWER_VERDICTS = {"pass", "incomplete", "unsupported", "unsafe", "wrong"}
LOGGER = logging.getLogger("brite.eval.judge")

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


@dataclass(frozen=True)
class JudgeConfig:
    """API configuration; the key is deliberately excluded from repr and reports."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/"))
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "api_key", self.api_key.strip())
        if not self.model:
            raise ValueError("Judge model is required. Set --judge-model or RAG_JUDGE_MODEL.")
        if not self.api_key:
            raise ValueError("Judge API key is required. Set RAG_JUDGE_API_KEY or enter it in the local dashboard.")
        if "\r" in self.api_key or "\n" in self.api_key:
            raise ValueError("Judge API key must be a single line.")
        parsed = urlsplit(self.base_url)
        try:
            loopback = parsed.hostname == "localhost" or ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (not parsed.hostname or parsed.scheme not in {"https", "http"}
                or (parsed.scheme == "http" and not loopback)):
            raise ValueError("Judge API URL must use HTTPS (HTTP is allowed only on loopback).")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("Judge API URL cannot contain credentials, query parameters, or a fragment.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Judge timeout must be a positive finite number.")

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def report_metadata(self) -> dict[str, Any]:
        return {"judge_api_base": self.base_url, "judge_model": self.model,
                "judge_protocol": "openai-chat-completions"}


def make_judge_config(
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_JUDGE_API_KEY_ENV,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
) -> JudgeConfig:
    return JudgeConfig(
        base_url=base_url or os.environ.get("RAG_JUDGE_BASE_URL") or DEFAULT_JUDGE_BASE_URL,
        model=model or os.environ.get("RAG_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
        api_key=api_key or os.environ.get(api_key_env) or "",
        timeout=timeout,
    )


def evaluate_with_api(
        question: str,
        ground_truth: str,
        generated_answer: str,
        retrieved_chunks: list[dict],
        config: JudgeConfig,
        domain: str = "general",
) -> dict[str, Any]:
    """Judge through any compatible Chat Completions endpoint with Bearer auth."""
    prompt = _build_judge_prompt(question, ground_truth, generated_answer, retrieved_chunks, domain=domain)
    LOGGER.info("Sending semantic judge API request: question_chars=%d answer_chars=%d chunks=%d",
                len(question or ""), len(generated_answer or ""), len(retrieved_chunks or []))
    try:
        response = requests.post(
            config.endpoint,
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json={"model": config.model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=config.timeout,
            allow_redirects=False,
        )
    except requests.exceptions.Timeout:
        return judge_error("Judge API request timed out.")
    except requests.exceptions.RequestException:
        # Provider bodies and exception strings may contain the Authorization header.
        return judge_error("Judge API network request failed. Check the API URL and connection.")

    if response.status_code != 200:
        return judge_error(f"Judge API request failed (HTTP {response.status_code}). Check the API key, model, URL, and provider quota.")
    try:
        body = response.json()
        raw_text = _extract_chat_text(body)
    except (ValueError, TypeError, KeyError, IndexError):
        return judge_error("Judge API response did not contain Chat Completions message text.")

    # Do not persist a key even if a provider unexpectedly echoes it in its output.
    raw_text = raw_text.replace(config.api_key, "[REDACTED]")
    parsed = extract_first_json_object(raw_text)
    if parsed.get("judge_status") == "error":
        _log_judge_result(parsed)
        return parsed
    result = normalize_judge_response(parsed)
    _log_judge_result(result)
    return result


def _extract_chat_text(body: Any) -> str:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("No message text")
    return content


def extract_first_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        return judge_error("Expected a string while parsing API judge JSON output.")

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        return judge_error("Parsed JSON was not an object.", raw_response=text)
    except json.JSONDecodeError as direct_error:
        first_error: Exception = direct_error

    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                    return judge_error("Extracted JSON was not an object.", raw_response=text)
                except json.JSONDecodeError as exc:
                    first_error = exc
                    start = None

    return judge_error(f"Could not parse a balanced JSON object from API judge output: {first_error}", raw_response=text)


def normalize_judge_response(judge_json: dict[str, Any]) -> dict[str, Any]:
    retrieval = judge_json.get("retrieval")
    answer = judge_json.get("answer")
    if not isinstance(retrieval, dict) or not isinstance(answer, dict):
        return judge_error("Judge JSON is missing retrieval or answer objects.", raw_response=judge_json)

    chunk_relevance_raw = retrieval.get("chunk_relevance")
    if not isinstance(chunk_relevance_raw, list):
        return judge_error("Judge JSON is missing retrieval.chunk_relevance labels.", raw_response=judge_json)

    answer_score = _optional_float(answer.get("score"))
    if answer_score is None:
        return judge_error("Judge JSON is missing numeric answer.score.", raw_response=judge_json)

    chunk_relevance = _normalize_chunk_relevance(chunk_relevance_raw)
    distractor_chunks = _normalize_distractors(retrieval.get("distractor_chunks"))
    metrics = calculate_retrieval_metrics(chunk_relevance, k=RETRIEVAL_EVAL_K, distractor_chunks=distractor_chunks)
    dangerous_claims = _string_list(answer.get("dangerous_claims"))
    semantic_score = round(0.5 * metrics["retrieval_ndcg_at_5"] + 0.5 * answer_score, 4)
    passed = (
            metrics["retrieval_ndcg_at_5"] >= SEMANTIC_PASS_THRESHOLD
            and answer_score >= SEMANTIC_PASS_THRESHOLD
            and not dangerous_claims
    )

    normalized_retrieval = {
        "verdict": _choice(retrieval.get("verdict"), _RETRIEVAL_VERDICTS),
        "chunk_relevance": chunk_relevance,
        "found_facts": _string_list(retrieval.get("found_facts")),
        "missing_facts": _string_list(retrieval.get("missing_facts")),
        "distractor_chunks": distractor_chunks,
        "metrics": metrics,
    }
    normalized_answer = {
        "score": round(answer_score, 4),
        "verdict": _choice(answer.get("verdict"), _ANSWER_VERDICTS),
        "missing_facts": _string_list(answer.get("missing_facts")),
        "unsupported_claims": _string_list(answer.get("unsupported_claims")),
        "dangerous_claims": dangerous_claims,
        "explanation": _string_value(answer.get("explanation")),
    }

    return {
        "judge_status": "ok",
        "error_message": None,
        "pass": passed,
        "semantic_score": semantic_score,
        "needed_facts": _string_list(judge_json.get("needed_facts")),
        "retrieval": normalized_retrieval,
        "answer": normalized_answer,
        "raw_judge": judge_json,
    }


def calculate_retrieval_metrics(
        chunk_relevance: list[dict[str, Any]],
        k: int = RETRIEVAL_EVAL_K,
        distractor_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    relevance_by_rank: dict[int, int] = {}
    for item in chunk_relevance:
        if not isinstance(item, dict):
            continue
        rank = _positive_int(item.get("rank"))
        if rank is None or rank < 1:
            continue
        relevance_by_rank[rank] = _relevance_value(item.get("relevance"))

    relevances = [relevance_by_rank.get(rank, 0) for rank in range(1, k + 1)]
    first_direct_rank = next((rank for rank, relevance in enumerate(relevances, start=1) if relevance >= 2), None)
    dcg = _dcg(relevances)
    idcg = _dcg(sorted(relevances, reverse=True))
    ndcg = 0.0 if idcg == 0 else dcg / idcg

    return {
        "retrieval_hit_at_5": 1 if first_direct_rank is not None else 0,
        "retrieval_precision_at_5": round(sum(1 for relevance in relevances if relevance >= 2) / k, 4),
        "retrieval_soft_precision_at_5": round(sum(relevance / 2 for relevance in relevances) / k, 4),
        "retrieval_mrr_at_5": round(0.0 if first_direct_rank is None else 1 / first_direct_rank, 4),
        "retrieval_ndcg_at_5": round(ndcg, 4),
        "retrieval_distractor_count": len(distractor_chunks or []),
    }


def disabled_judge_result() -> dict[str, Any]:
    result = judge_error("Semantic judge disabled.", status="disabled")
    result["error_message"] = None
    result["answer"]["explanation"] = "Semantic judge disabled."
    return result


def api_error_judge_result(message: str) -> dict[str, Any]:
    return judge_error(message or "RAG API returned an error.", status="api_error")


def _log_judge_result(result: dict[str, Any]) -> None:
    retrieval = result.get("retrieval") or {}
    answer = result.get("answer") or {}
    metrics = retrieval.get("metrics") or {}
    if result.get("judge_status") != "ok":
        LOGGER.error(
            "Semantic judge error detail: status=%s error_message=%s raw_response_preview=%s",
            result.get("judge_status"),
            result.get("error_message"),
            _truncate(_string_value(result.get("raw_response")), 1000),
        )
    LOGGER.info(
        "Semantic judge result: status=%s pass=%s semantic_score=%s retrieval_verdict=%s ndcg5=%s answer_verdict=%s answer_score=%s dangerous_claims=%d",
        result.get("judge_status"),
        result.get("pass"),
        result.get("semantic_score"),
        retrieval.get("verdict"),
        metrics.get("retrieval_ndcg_at_5"),
        answer.get("verdict"),
        answer.get("score"),
        len(answer.get("dangerous_claims") or []),
    )


def judge_error(message: str, raw_response: Any | None = None, status: str = "error") -> dict[str, Any]:
    result = {
        "judge_status": status,
        "error_message": message,
        "pass": None,
        "semantic_score": None,
        "needed_facts": [],
        "retrieval": {
            "verdict": None,
            "chunk_relevance": [],
            "found_facts": [],
            "missing_facts": [],
            "distractor_chunks": [],
            "metrics": {
                "retrieval_hit_at_5": None,
                "retrieval_precision_at_5": None,
                "retrieval_soft_precision_at_5": None,
                "retrieval_mrr_at_5": None,
                "retrieval_ndcg_at_5": None,
                "retrieval_distractor_count": None,
            },
        },
        "answer": {
            "score": None,
            "verdict": None,
            "missing_facts": [],
            "unsupported_claims": [],
            "dangerous_claims": [],
            "explanation": message,
        },
    }
    if raw_response is not None:
        result["raw_response"] = raw_response
    return result


def _build_judge_prompt(
        question: str,
        ground_truth: str,
        generated_answer: str,
        retrieved_chunks: list[dict],
        domain: str = "medical first-aid",
) -> str:
    chunks = [_format_chunk_for_prompt(chunk, rank) for rank, chunk in enumerate(retrieved_chunks or [], start=1)]
    safety_rule = (
        "For answer quality, check correctness, completeness, support, and medical safety.\n"
        if domain == "medical first-aid" else
        "For answer quality, check correctness, completeness, support, and safety appropriate to the domain.\n"
    )
    danger_rule = (
        "- In first-aid/medical cases, dangerous advice must be flagged strongly.\n"
        if domain == "medical first-aid" else
        "- Dangerous advice appropriate to the domain must be flagged strongly.\n"
    )
    expected_schema = {
        "needed_facts": ["..."],
        "retrieval": {
            "verdict": "pass",
            "chunk_relevance": [{"rank": 1, "relevance": 2, "reason": "..."}],
            "found_facts": ["..."],
            "missing_facts": ["..."],
            "distractor_chunks": [{"rank": 5, "reason": "..."}],
        },
        "answer": {
            "score": 0.0,
            "verdict": "pass",
            "missing_facts": ["..."],
            "unsupported_claims": ["..."],
            "dangerous_claims": ["..."],
            "explanation": "...",
        },
    }
    return (
        f"You are evaluating a {domain} RAG system.\n"
        "You must judge retrieval and answer quality separately.\n"
        "Use the ground truth as the expected answer, but allow correct paraphrases and extra details if they are supported by retrieved chunks.\n"
        "Do not require exact wording.\n"
        "For retrieval, label each chunk independently using relevance 0, 1, or 2.\n"
        f"{safety_rule}"
        "Return strict JSON only. No markdown. No prose outside JSON.\n\n"
        "Retrieval relevance scale:\n"
        "- 2 = directly supports the ground truth or contains needed evidence\n"
        "- 1 = partially related or useful background, but not enough alone\n"
        "- 0 = irrelevant or misleading\n\n"
        "Retrieval verdict options: pass, medium, fail.\n"
        "Answer verdict options: pass, incomplete, unsupported, unsafe, wrong.\n"
        "Answer rules:\n"
        "- Correct paraphrases are allowed.\n"
        "- Extra details are allowed if supported by retrieved chunks.\n"
        "- Do not mark an answer unsupported just because it uses words not found in the short ground truth.\n"
        f"{danger_rule}"
        "- If the answer assumes the wrong condition, mark wrong or unsafe.\n\n"
        f"Question:\n{question or ''}\n\n"
        f"Ground truth:\n{ground_truth or ''}\n\n"
        f"Generated answer:\n{generated_answer or ''}\n\n"
        f"Retrieved chunks:\n{json.dumps(chunks, ensure_ascii=False, indent=2)}\n\n"
        f"Return JSON with exactly this schema shape:\n{json.dumps(expected_schema, ensure_ascii=False, indent=2)}"
    )


def _format_chunk_for_prompt(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "source": _chunk_source(chunk),
        "score": chunk.get("score") if isinstance(chunk, dict) else None,
        "snippet": _truncate(_chunk_text(chunk), 2400),
    }


def _normalize_chunk_relevance(items: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        rank = _positive_int(item.get("rank")) or index
        normalized.append(
            {
                "rank": rank,
                "relevance": _relevance_value(item.get("relevance")),
                "reason": _string_value(item.get("reason")),
            }
        )
    return sorted(normalized, key=lambda item: item["rank"])


def _normalize_distractors(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rank = _positive_int(item.get("rank"))
        if rank is None:
            continue
        normalized.append({"rank": rank, "reason": _string_value(item.get("reason"))})
    return normalized


def _chunk_source(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return "unknown"
    for key in ("sourceName", "source_name", "source", "title", "documentTitle", "document_title"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _chunk_text(chunk: dict[str, Any]) -> str:
    if not isinstance(chunk, dict):
        return ""
    for key in ("contentSnippet", "content_snippet", "snippet", "text", "content", "body"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3].rstrip()}..."


def _dcg(relevances: list[int]) -> float:
    return sum((2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(relevances, start=1))


def _choice(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in allowed else None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return max(0.0, min(1.0, number))


def _positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _relevance_value(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(2, number))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return [_string_value(value)] if _string_value(value) else []
    result: list[str] = []
    for item in value:
        text = _string_value(item)
        if text:
            result.append(text)
    return result


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)
