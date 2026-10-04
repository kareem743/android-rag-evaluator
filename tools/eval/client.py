from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from adapters import HttpRagAdapter, load_profile
from runner import iter_evaluations

from common import (
    call_evaluate,
    call_health,
    call_upload_file,
    load_cases_from_path,
    write_json_report,
    write_jsonl,
)
from judge import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    api_error_judge_result,
    disabled_judge_result,
    evaluate_with_ollama,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Android or HTTP RAG application through an adapter.")
    parser.add_argument("--adapter", choices=["brite", "generic"], default="brite",
                        help="Built-in API adapter (default: existing BRITE Android app).")
    parser.add_argument("--adapter-profile", help="JSON profile mapping another app's endpoints and fields.")
    parser.add_argument("--judge-domain", help="Judging domain override; defaults to the adapter's domain.")
    parser.add_argument(
        "--dataset",
        default="app/src/debug/assets/eval_cases.json",
        help="Path to eval dataset JSON/JSONL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: tools/eval/output/eval_run_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Structured JSON report path (default: same output stem with .json suffix)",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:9000",
        help="RAG application's HTTP API base URL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top_k override for retrieval",
    )
    parser.add_argument(
        "--llm-model-id", "--model-id",
        default=None,
        help="Optional model identifier understood by the selected application.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus file to index before evaluation (BRITE adapter only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of cases to run.",
    )
    parser.add_argument(
        "--no-ollama-judge",
        action="store_true",
        help="Disable semantic scoring and only store application outputs.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Request retrieval-only evaluation from an app that supports it.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL for semantic judging.",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model for semantic judging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases_from_path(args.dataset)
    if args.limit is not None and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases were loaded from the dataset.")

    try:
        profile = load_profile(getattr(args, "adapter_profile", None), getattr(args, "adapter", "brite"))
        adapter = HttpRagAdapter(args.api_base, profile, health_call=call_health,
                                 evaluate_call=call_evaluate, upload_call=call_upload_file)
        health = adapter.check_health()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Adapter setup failed: {exc}") from exc

    print(f"[health] status={health.get('status')} runtime_ready={health.get('runtime_ready')}")
    if args.llm_model_id:
        print(f"[model] selected={args.llm_model_id}")
    if args.corpus:
        upload_response = adapter.upload_corpus(args.corpus)
        upload_error = upload_response.get("error") if isinstance(upload_response, dict) else None
        if upload_error:
            raise SystemExit(f"Corpus upload failed: {upload_error}")
        print(
            f"[upload] rag_id={upload_response.get('rag_id')} "
            f"name={upload_response.get('name')} "
            f"node_count={upload_response.get('node_count')}"
        )
    print(f"[run] cases={len(cases)} top_k_override={args.top_k} retrieval_only={args.retrieval_only}")

    output_path = resolve_output_path(args.output)
    json_output_path = resolve_json_output_path(args.json_output, output_path)
    report_metadata = {
        "adapter": profile.name,
        "judge_domain": getattr(args, "judge_domain", None) or profile.judge_domain,
        "dataset": str(args.dataset),
        "api_base": args.api_base,
        "top_k_override": args.top_k,
        "corpus": args.corpus,
        "ollama_judge_enabled": not args.no_ollama_judge and not args.retrieval_only,
        "ollama_base_url": args.ollama_base_url,
        "ollama_model": args.ollama_model,
        "llm_model_id": args.llm_model_id,
        "retrieval_only": args.retrieval_only,
    }
    print(f"[output] json={json_output_path} jsonl={output_path}")

    rows: list[dict[str, Any]] = []

    def judge_case(case, api_response):
        return evaluate_case_with_judge(
            case=case, api_response=api_response,
            use_ollama_judge=(not args.no_ollama_judge) and (not args.retrieval_only),
            ollama_base_url=args.ollama_base_url,
            ollama_model=args.ollama_model,
            domain=getattr(args, "judge_domain", None) or profile.judge_domain,
        )

    runs = iter_evaluations(adapter, cases, judge_case, top_k=args.top_k,
                           model_id=args.llm_model_id, retrieval_only=args.retrieval_only)
    for index, row in enumerate(runs, start=1):
        case, api_response, judge = row["case"], row["response"], row["judge"]
        rows.append(row)
        write_jsonl(output_path, rows)
        write_json_report(json_output_path, rows, metadata=report_metadata)

        latency_ms = (
            api_response.get("total_latency_ms")
            if isinstance(api_response, dict)
            else None
        )
        error_code = (
            (api_response.get("error") or {}).get("code")
            if isinstance(api_response, dict)
            else None
        )
        print(
            f"[{index}/{len(cases)}] case={case['case_id']} pass={judge.get('pass')} "
            f"semantic_score={_format_score(judge.get('semantic_score'))} "
            f"judge={judge.get('judge_status')} latency_ms={latency_ms} error={error_code}"
        )

    write_jsonl(output_path, rows)
    write_json_report(json_output_path, rows, metadata=report_metadata)
    pass_count = sum(1 for row in rows if row.get("pass"))
    scored_count = sum(1 for row in rows if row.get("pass") is not None)
    avg_score = _mean_score(row.get("semantic_score") for row in rows)
    pass_rate = "n/a" if scored_count == 0 else f"{pass_count/scored_count:.2%}"
    print(
        f"[done] saved_json={json_output_path} saved_jsonl={output_path} "
        f"pass_rate={pass_rate} avg_semantic={_format_score(avg_score)}"
    )
    return 0


def evaluate_case_with_judge(
    case: dict[str, Any],
    api_response: dict[str, Any],
    use_ollama_judge: bool = True,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    domain: str = "medical first-aid",
) -> dict[str, Any]:
    error = (api_response.get("error") or {}) if isinstance(api_response, dict) else {}
    if error:
        return api_error_judge_result(error.get("message", "RAG API returned an error."))

    if not use_ollama_judge:
        return disabled_judge_result()

    return evaluate_with_ollama(
        question=case.get("question", ""),
        ground_truth=case.get("ground_truth_answer", ""),
        generated_answer=(api_response.get("generated_answer") or "").strip(),
        retrieved_chunks=api_response.get("retrieved_chunks") or [],
        ollama_base_url=ollama_base_url,
        model=ollama_model,
        domain=domain,
    )


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


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def resolve_output_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tools/eval/output") / f"eval_run_{timestamp}.jsonl"


def resolve_json_output_path(path_arg: str | None, jsonl_output_path: Path) -> Path:
    if path_arg:
        return Path(path_arg)
    return jsonl_output_path.with_suffix(".json")


if __name__ == "__main__":
    raise SystemExit(main())
