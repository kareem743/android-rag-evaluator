from __future__ import annotations

import base64
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def load_cases_from_path(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    raw = target.read_text(encoding="utf-8")
    return parse_cases_text(raw, target.name)


def load_corpus_from_path(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    raw_bytes = target.read_bytes()
    return parse_corpus_bytes(raw_bytes, target.name)


def parse_cases_text(raw: str, filename: str = "cases.json") -> list[dict[str, Any]]:
    name = filename.lower()
    if name.endswith(".jsonl"):
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            items = payload.get("cases", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError("Unsupported cases payload format.")

    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        case_id = _pick_str(item, "id", "case_id", "caseId") or f"case_{idx + 1}"
        question = _pick_str(item, "question", "prompt")
        if not question:
            continue
        ground_truth = _pick_str(
            item,
            "groundTruthAnswer",
            "ground_truth_answer",
            "ground_truth",
            "reference_answer",
            "answer",
        ) or ""
        category = _pick_str(item, "category")
        notes = _pick_str(item, "notes")
        cases.append(
            {
                "case_id": case_id,
                "question": question,
                "ground_truth_answer": ground_truth,
                "category": category,
                "notes": notes,
            }
        )
    return cases


def parse_corpus_bytes(raw_bytes: bytes, filename: str = "corpus.txt") -> list[dict[str, str]]:
    name = filename.lower()
    if name.endswith((".json", ".jsonl")):
        raw = raw_bytes.decode("utf-8", errors="replace")
        if name.endswith(".jsonl"):
            payload: Any = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            payload = json.loads(raw)
        return _normalize_corpus_payload(payload)

    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    stem = Path(filename).stem or "uploaded_corpus"
    return [{"sourceId": stem, "title": filename, "content": text}]


def parse_upload_contents(contents: str, filename: str) -> bytes:
    if "," not in contents:
        raise ValueError("Upload content is malformed.")
    _, b64_data = contents.split(",", 1)
    return base64.b64decode(b64_data)


def call_health(api_base: str, timeout: int = 180) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/health"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_evaluate(
    api_base: str,
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/evaluate"
    response = requests.post(url, json=payload, timeout=timeout)
    body = response.json()
    if response.status_code >= 400:
        return body
    return body


def call_upload_file(
    api_base: str,
    path: str | Path,
    timeout: int = 300,
) -> dict[str, Any]:
    target = Path(path)

    mime_type = mimetypes.guess_type(target.name)[0]
    if mime_type is None:
        if target.suffix.lower() == ".pdf":
            mime_type = "application/pdf"
        else:
            mime_type = "application/octet-stream"

    data = {
        "name": target.stem,
        "description": "",
        "file_name": target.name,
        "mime_type": mime_type,
    }

    url = f"{api_base.rstrip('/')}/corpus/file"
    with target.open("rb") as handle:
        response = requests.post(
            url,
            data=data,
            files={"file": (target.name, handle, mime_type)},
            timeout=timeout,
        )
    body = response.json()

    if response.status_code >= 400:
        return body

    return body


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return output


def write_json_report(
    path: str | Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "brite_eval_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "summary": _build_report_summary(rows),
        "cases": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _normalize_corpus_payload(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        if "corpus" in payload and isinstance(payload["corpus"], list):
            items = payload["corpus"]
        elif "documents" in payload and isinstance(payload["documents"], list):
            items = payload["documents"]
        elif "content" in payload:
            items = [payload]
        else:
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    docs: list[dict[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        content = _pick_str(item, "content", "text", "body")
        if not content:
            continue
        source_id = _pick_str(item, "sourceId", "source_id", "id") or f"source_{idx + 1}"
        title = _pick_str(item, "title", "name") or source_id
        docs.append({"sourceId": source_id, "title": title, "content": content})
    return docs


def _pick_str(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = sum(1 for row in rows if row.get("pass") is True)
    fail_count = sum(1 for row in rows if row.get("pass") is False)
    scored_count = pass_count + fail_count
    return {
        "case_count": len(rows),
        "scored_count": scored_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": None if scored_count == 0 else pass_count / scored_count,
        "avg_semantic_score": _mean_score(row.get("semantic_score") for row in rows),
        "avg_answer_score": _mean_score(row.get("answer_score") for row in rows),
    }


def _mean_score(values: Any) -> float | None:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return None if not numbers else sum(numbers) / len(numbers)
