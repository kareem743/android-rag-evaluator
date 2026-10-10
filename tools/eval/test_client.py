from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import client


def test_resolve_output_path_defaults_to_eval_output_folder():
    path = client.resolve_output_path(None)

    assert path.parent == Path("tools/eval/output")
    assert path.name.startswith("eval_run_")
    assert path.suffix == ".jsonl"


def test_evaluate_case_with_judge_returns_api_error_on_response_error():
    result = client.evaluate_case_with_judge(
        case={"question": "q", "ground_truth_answer": "gt"},
        api_response={"error": {"message": "runtime failed"}},
    )

    assert result["pass"] is None
    assert result["judge_status"] == "api_error"
    assert result["error_message"] == "runtime failed"


def test_main_runs_end_to_end(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("RAG_JUDGE_API_KEY", "unit-test-secret")
    written: dict[str, object] = {}
    args = Namespace(
        dataset="dataset.json",
        output=None,
        json_output=None,
        api_base="http://127.0.0.1:9000",
        top_k=3,
        llm_model_id=None,
        corpus="corpus.txt",
        limit=1,
        no_judge=False,
        retrieval_only=False,
        judge_api_base="https://judge.example/v1",
        judge_model="test-model",
        judge_api_key_env="RAG_JUDGE_API_KEY",
        judge_timeout=180,
    )

    monkeypatch.setattr(client, "parse_args", lambda: args)
    monkeypatch.setattr(
        client,
        "load_cases_from_path",
        lambda path: [
            {
                "case_id": "case-1",
                "question": "What should be done?",
                "ground_truth_answer": "Provide oxygen and monitor saturation.",
                "category": "airway",
                "notes": "field",
            },
            {
                "case_id": "case-2",
                "question": "Unused because of limit",
                "ground_truth_answer": "unused",
            },
        ],
    )
    monkeypatch.setattr(client, "call_health", lambda api_base: {"status": "ok", "runtime_ready": True})
    monkeypatch.setattr(
        client,
        "call_upload_file",
        lambda api_base, path: {"rag_id": "rag-1", "name": "Guide", "node_count": 7},
    )
    monkeypatch.setattr(
        client,
        "call_evaluate",
        lambda api_base, payload, timeout=180: {
            "generated_answer": "Provide oxygen and monitor saturation.",
            "retrieved_chunks": [],
            "total_latency_ms": 42,
            "runtime": {"retrieval_confidence": "high"},
        },
    )
    monkeypatch.setattr(
        client,
        "evaluate_with_api",
        lambda **kwargs: {
            "judge_status": "ok",
            "error_message": None,
            "pass": True,
            "semantic_score": 0.91,
            "needed_facts": [],
            "retrieval": {
                "verdict": "pass",
                "chunk_relevance": [],
                "found_facts": [],
                "missing_facts": [],
                "distractor_chunks": [],
                "metrics": {
                    "retrieval_hit_at_5": 1,
                    "retrieval_precision_at_5": 0.2,
                    "retrieval_soft_precision_at_5": 0.2,
                    "retrieval_mrr_at_5": 1.0,
                    "retrieval_ndcg_at_5": 1.0,
                },
            },
            "answer": {
                "score": 0.82,
                "verdict": "pass",
                "missing_facts": [],
                "unsupported_claims": [],
                "dangerous_claims": [],
                "explanation": "Grounded.",
            },
        },
    )

    output_path = tmp_path / "eval.jsonl"
    json_output_path = tmp_path / "eval.json"
    monkeypatch.setattr(client, "resolve_output_path", lambda arg: output_path)
    monkeypatch.setattr(client, "resolve_json_output_path", lambda arg, jsonl_output_path: json_output_path)

    def fake_write_jsonl(path, rows):
        written["path"] = path
        written["rows"] = rows
        return Path(path)

    def fake_write_json_report(path, rows, metadata=None):
        written["json_path"] = path
        written["json_rows"] = rows
        written["json_metadata"] = metadata
        return Path(path)

    monkeypatch.setattr(client, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(client, "write_json_report", fake_write_json_report)

    exit_code = client.main()
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert written["path"] == output_path
    assert written["json_path"] == json_output_path
    assert len(written["rows"]) == 1
    assert written["json_rows"] == written["rows"]
    assert written["json_metadata"]["dataset"] == "dataset.json"
    assert written["rows"][0]["request"]["top_k"] == 3
    assert "corpus_documents" not in written["rows"][0]["request"]
    assert "[health] status=ok runtime_ready=True" in stdout
    assert "[upload] rag_id=rag-1 name=Guide node_count=7" in stdout
    assert "[done] saved_json=" in stdout


def test_missing_judge_key_fails_before_contacting_app(monkeypatch, tmp_path):
    import json
    import pytest
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"case_id": "one", "question": "Question"}]))
    monkeypatch.delenv("RAG_JUDGE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["client.py", "--dataset", str(dataset), "--judge-model", "model"])
    def unexpected_call(*args, **kwargs):
        pytest.fail("RAG app must not be contacted before judge configuration is valid")
    monkeypatch.setattr(client, "call_health", unexpected_call)
    with pytest.raises(SystemExit, match="API key is required"):
        client.main()
