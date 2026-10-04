"""Shared batch execution for the CLI and dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator

import requests

from adapters import HttpRagAdapter
from judge import api_error_judge_result, judge_error


def iter_evaluations(
    adapter: HttpRagAdapter,
    cases: list[dict[str, Any]],
    judge_case: Callable[[dict, dict], dict],
    *, top_k: int | None = None, model_id: str | None = None,
    retrieval_only: bool = False, timeout: int = 180,
    on_case_start: Callable[[int, dict], None] | None = None,
) -> Iterator[dict[str, Any]]:
    # Validate controls before executing any cases (unsupported options are not
    # silently ignored, because that would invalidate experiment comparisons).
    if cases:
        adapter.build_request(cases[0], top_k, model_id, retrieval_only)
    for index, case in enumerate(cases, start=1):
        payload = adapter.build_request(case, top_k, model_id, retrieval_only)
        if on_case_start:
            on_case_start(index, case)
        try:
            response = adapter.evaluate(payload, timeout=timeout, retrieval_only=retrieval_only)
        except requests.exceptions.ReadTimeout:
            response = {"error": {"code": "request_timeout", "message": f"Evaluation timed out after {timeout}s."}}
        except Exception as exc:  # One failed query should not discard a batch.
            response = {"error": {"code": "request_failed", "message": str(exc)}}
        if response.get("error"):
            judge = api_error_judge_result(response["error"].get("message", "RAG API returned an error."))
        else:
            try:
                judge = judge_case(case, response)
            except Exception as exc:
                judge = judge_error(f"Judge failed: {exc}")
        yield {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "case": case, "request": payload, "response": response, "judge": judge,
            "pass": judge.get("pass"), "semantic_score": judge.get("semantic_score"),
            "answer_score": (judge.get("answer") or {}).get("score"),
        }
