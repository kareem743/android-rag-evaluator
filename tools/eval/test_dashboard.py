from __future__ import annotations

from pathlib import Path

import dashboard
from judge import JudgeConfig
import requests
import base64
import json

from adapters import AdapterProfile, HttpRagAdapter


def _sample_row(**overrides):
    row = {
        "case_id": "case-1",
        "category": "airway",
        "question": "What should be done?",
        "ground_truth_answer": "Provide oxygen and monitor saturation.",
        "generated_answer": "Provide oxygen and monitor saturation.",
        "retrieved_chunks": [
            {"sourceName": "Guide", "score": 0.91, "contentSnippet": "Provide oxygen and monitor saturation."}
        ],
        "retrieval_timing_ms": 10,
        "generation_timing_ms": 20,
        "total_latency_ms": 30,
        "retrieval_confidence": "high",
        "query_top_k": 5,
        "vector_candidate_count": 12,
        "bm25_candidate_count": 8,
        "fused_candidate_count": 14,
        "deduped_final_count": 5,
        "expanded_result_count": 5,
        "reranker_candidate_count": 8,
        "reranker_applied": True,
        "top_fused_score": 0.031,
        "top_final_score": 0.88,
        "tokens_per_second": 25.5,
        "time_to_first_token_ms": 110.0,
        "tokens_evaluated": 40,
        "tokens_predicted": 20,
        "total_tokens": 60,
        "pass": True,
        "semantic_judge_status": "ok",
        "semantic_score": 0.9,
        "retrieval_verdict": "pass",
        "retrieval_hit_at_5": 1,
        "retrieval_precision_at_5": 0.2,
        "retrieval_soft_precision_at_5": 0.2,
        "retrieval_mrr_at_5": 1.0,
        "retrieval_ndcg_at_5": 1.0,
        "retrieval_distractor_count": 0,
        "answer_score": 0.8,
        "answer_verdict": "pass",
        "needed_facts": ["oxygen", "monitor saturation"],
        "found_facts": ["oxygen", "monitor saturation"],
        "chunk_relevance": [{"rank": 1, "relevance": 2, "reason": "Direct support."}],
        "distractor_chunks": [],
        "answer_explanation": "Looks grounded.",
        "missing_facts": [],
        "unsupported_claims": [],
        "dangerous_claims": [],
        "error_code": None,
    }
    row.update(overrides)
    return row


def test_hex_to_rgba_returns_plotly_safe_color():
    assert dashboard._hex_to_rgba("#6366f1", 0.13) == "rgba(99, 102, 241, 0.130)"


def test_profile_upload_selects_custom_adapter_and_rejects_invalid_upload():
    def upload(data):
        return "data:application/json;base64," + base64.b64encode(json.dumps(data).encode()).decode()

    profile, label, kind = dashboard.on_adapter_profile_upload(upload({"name": "second-app"}), "profile.json")
    assert kind == "custom"
    assert "second-app" in label
    assert dashboard.make_application_adapter("http://localhost:9000", kind, profile).profile.name == "second-app"
    invalid, label, kind = dashboard.on_adapter_profile_upload(upload({"bad_field": True}), "bad.json")
    assert invalid.get("_error")
    import pytest
    with pytest.raises(ValueError, match="valid API profile"):
        dashboard.make_application_adapter("http://localhost:9000", kind, invalid)


def test_dashboard_serves_layout_with_new_adapter_controls():
    http = dashboard.app.server.test_client()
    assert http.get("/").status_code == 200
    response = http.get("/_dash-layout")
    assert response.status_code == 200
    assert b'"adapter-profile-upload"' in response.data
    assert b'"adapter-kind"' in response.data
    assert http.get("/_dash-dependencies").status_code == 200


def test_dashboard_runs_generic_app_without_android_readiness_or_diagnostics(monkeypatch, tmp_path):
    adapter = HttpRagAdapter("http://localhost:9000", AdapterProfile(name="second-app"))
    monkeypatch.setattr(adapter, "evaluate", lambda *args, **kwargs: {
        "generated_answer": "Answer", "retrieved_chunks": [{"text": "Evidence"}], "total_latency_ms": 5,
    })
    monkeypatch.setattr(dashboard, "make_application_adapter", lambda *args: adapter)
    monkeypatch.setattr(dashboard, "_default_output_path", lambda: tmp_path / "generic.jsonl")
    dashboard._initialize_run_state("generic", 1, "Queued", "Running")
    dashboard._run_evaluation_job(
        job_id="generic", api_base="http://localhost:9000", llm_model_id=None, top_k=3,
        use_judge=False, retrieval_only=False, judge_config=None, cases=[{"case_id": "one", "question": "Question"}], corpus_docs=None,
        total_loaded_cases=1, adapter_kind="generic",
    )
    snapshot = dashboard._snapshot_run_state()
    assert snapshot["status"] == "completed"
    assert snapshot["rows"][0]["generated_answer"] == "Answer"
    assert snapshot["rows"][0]["vector_candidate_count"] is None
    report = json.loads((tmp_path / "generic.json").read_text())
    assert report["metadata"]["adapter"] == "second-app"


def test_render_metrics_builds_figures_without_plotly_color_errors():
    metrics = dashboard.render_metrics(
        [_sample_row(), _sample_row(case_id="case-2", semantic_score=0.4, total_latency_ms=80, **{"pass": False})],
        [],
    )

    assert metrics[0] == "2"
    assert metrics[1] == "50.0%"
    score_fig = metrics[7]
    latency_fig = metrics[8]
    grid_rows = metrics[9]

    assert score_fig.data
    assert latency_fig.data
    assert all(getattr(trace, "fillcolor", None).startswith("rgba(") for trace in latency_fig.data)
    assert len(grid_rows) == 2


def test_show_case_detail_includes_runtime_and_retrieval_sections():
    content = dashboard.show_case_detail([_sample_row()])

    assert "#### Retrieval Metrics" in content
    assert "#### Dangerous Claims" in content
    assert "relevance=2" in content
    assert "#### 📈 Runtime Metrics" in content
    assert "#### 🧭 Retrieval Diagnostics" in content
    assert "Final Chunks" in content
    assert "Guide" in content


def test_run_evaluation_job_updates_state_and_logs(monkeypatch, tmp_path):
    output_path = tmp_path / "eval.jsonl"
    json_report_path = tmp_path / "eval.json"
    cases = [
        {
            "case_id": "case-1",
            "question": "What should be done?",
            "ground_truth_answer": "Provide oxygen and monitor saturation.",
            "category": "airway",
            "notes": "field",
        },
        {
            "case_id": "case-2",
            "question": "How should the burn be managed?",
            "ground_truth_answer": "Cool the burn and cover it with a sterile dressing.",
            "category": "burn",
            "notes": "",
        },
    ]

    def fake_call_health(api_base):
        return {
            "status": "ok",
            "runtime_ready": True,
            "litert_lm_model_loaded": True,
            "embedding_initialized": True,
        }

    def fake_call_evaluate(api_base, payload, timeout=180):
        return {
            "generated_answer": cases[0]["ground_truth_answer"] if payload["case_id"] == "case-1" else cases[1]["ground_truth_answer"],
            "retrieved_chunks": [],
            "retrieval_timing_ms": 12,
            "generation_timing_ms": 34,
            "total_latency_ms": 46,
            "decoding_metrics": {
                "tokensPerSecond": 21.5,
                "timeToFirstTokenMs": 95.0,
                "tokensEvaluated": 30,
                "tokensPredicted": 18,
            },
            "runtime": {
                "retrieval_confidence": "high",
                "retrieval_diagnostics": {
                    "query_top_k": payload.get("top_k", 5),
                    "vector_candidate_count": 10,
                    "bm25_candidate_count": 7,
                    "fused_candidate_count": 12,
                    "deduped_final_count": 5,
                    "expanded_result_count": 5,
                    "reranker_candidate_count": 8,
                    "reranker_applied": True,
                    "top_fused_score": 0.031,
                    "top_final_score": 0.88,
                },
            },
        }

    monkeypatch.setattr(dashboard, "call_health", fake_call_health)
    monkeypatch.setattr(dashboard, "call_evaluate", fake_call_evaluate)
    monkeypatch.setattr(
        dashboard,
        "evaluate_with_api",
        lambda **kwargs: {
            "judge_status": "ok",
            "error_message": None,
            "pass": True,
            "semantic_score": 0.9,
            "needed_facts": ["expected care"],
            "retrieval": {
                "verdict": "pass",
                "chunk_relevance": [{"rank": 1, "relevance": 2, "reason": "Direct support."}],
                "found_facts": ["expected care"],
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
                "score": 0.8,
                "verdict": "pass",
                "missing_facts": [],
                "unsupported_claims": [],
                "dangerous_claims": [],
                "explanation": "Grounded.",
            },
        },
    )
    monkeypatch.setattr(dashboard, "_default_output_path", lambda: output_path)
    monkeypatch.setattr(dashboard, "write_jsonl", lambda path, rows: Path(output_path))
    monkeypatch.setattr(dashboard, "write_json_report", lambda path, rows, metadata=None: Path(path))

    dashboard._publish_run_error("reset")
    dashboard._initialize_run_state(
        job_id="job-1",
        total_cases=len(cases),
        live_status="Queued",
        summary="Running",
    )

    dashboard._run_evaluation_job(
        job_id="job-1",
        api_base="http://127.0.0.1:9000",
        llm_model_id=None,
        top_k=5,
        use_judge=True,
        retrieval_only=False,
        judge_config=JudgeConfig("https://judge.example/v1", "test-model", "unit-test-secret"),
        cases=cases,
        corpus_docs=[],
        total_loaded_cases=len(cases),
    )

    snapshot = dashboard._snapshot_run_state()

    assert snapshot["status"] == "completed"
    assert snapshot["completed_cases"] == 2
    assert len(snapshot["rows"]) == 2
    assert snapshot["output_path"] == str(json_report_path)
    assert any("final=5" in log for log in snapshot["logs"])
    assert any("[done] saved_json=" in log for log in snapshot["logs"])


def test_run_evaluation_job_records_timeout_and_continues(monkeypatch, tmp_path):
    output_path = tmp_path / "eval_timeout.jsonl"
    json_report_path = tmp_path / "eval_timeout.json"
    cases = [
        {
            "case_id": "case-1",
            "question": "What should be done?",
            "ground_truth_answer": "Provide oxygen and monitor saturation.",
            "category": "airway",
            "notes": "field",
        },
        {
            "case_id": "case-2",
            "question": "How should the burn be managed?",
            "ground_truth_answer": "Cool the burn and cover it with a sterile dressing.",
            "category": "burn",
            "notes": "",
        },
    ]

    def fake_call_health(api_base):
        return {
            "status": "ok",
            "runtime_ready": True,
            "litert_lm_model_loaded": True,
            "embedding_initialized": True,
        }

    def fake_call_evaluate(api_base, payload, timeout=180):
        if payload["case_id"] == "case-2":
            raise requests.exceptions.ReadTimeout("timed out")
        return {
            "generated_answer": cases[0]["ground_truth_answer"],
            "retrieved_chunks": [],
            "retrieval_timing_ms": 12,
            "generation_timing_ms": 34,
            "total_latency_ms": 46,
            "runtime": {
                "retrieval_confidence": "high",
                "retrieval_diagnostics": {
                    "query_top_k": payload.get("top_k", 5),
                    "vector_candidate_count": 10,
                    "bm25_candidate_count": 7,
                    "fused_candidate_count": 12,
                    "deduped_final_count": 5,
                    "expanded_result_count": 5,
                    "reranker_candidate_count": 8,
                    "reranker_applied": True,
                    "top_fused_score": 0.031,
                    "top_final_score": 0.88,
                },
            },
        }

    monkeypatch.setattr(dashboard, "call_health", fake_call_health)
    monkeypatch.setattr(dashboard, "call_evaluate", fake_call_evaluate)
    monkeypatch.setattr(dashboard, "_default_output_path", lambda: output_path)
    monkeypatch.setattr(dashboard, "write_jsonl", lambda path, rows: Path(output_path))
    monkeypatch.setattr(dashboard, "write_json_report", lambda path, rows, metadata=None: Path(path))

    dashboard._publish_run_error("reset")
    dashboard._initialize_run_state(
        job_id="job-timeout",
        total_cases=len(cases),
        live_status="Queued",
        summary="Running",
    )

    dashboard._run_evaluation_job(
        job_id="job-timeout",
        api_base="http://127.0.0.1:9000",
        llm_model_id=None,
        top_k=5,
        use_judge=False,
        retrieval_only=True,
        judge_config=JudgeConfig("https://judge.example/v1", "test-model", "unit-test-secret"),
        cases=cases,
        corpus_docs=[],
        total_loaded_cases=len(cases),
    )

    snapshot = dashboard._snapshot_run_state()

    assert snapshot["status"] == "completed"
    assert snapshot["completed_cases"] == 2
    assert len(snapshot["rows"]) == 2
    assert snapshot["rows"][1]["error_code"] == "request_timeout"
    assert any("timeout case=case-2" in log for log in snapshot["logs"])
    assert snapshot["output_path"] == str(json_report_path)


def test_dashboard_key_is_masked_and_environment_key_is_not_in_layout(monkeypatch):
    monkeypatch.setenv("RAG_JUDGE_API_KEY", "dashboard-env-secret")
    http = dashboard.app.server.test_client()
    layout = http.get("/_dash-layout")
    assert layout.status_code == 200
    assert b'dashboard-env-secret' not in layout.data
    def find(node):
        if isinstance(node, dict):
            if node.get("props", {}).get("id") == "judge-api-key":
                return node["props"]
            for value in node.values():
                found = find(value)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = find(child)
                if found:
                    return found
    props = find(layout.json)
    assert props["type"] == "password"
    assert props["value"] == ""
    assert props["persistence"] is False
    dependencies = http.get("/_dash-dependencies").json
    assert any(s["id"] == "judge-api-key" for callback in dependencies for s in callback["state"])


def test_dashboard_missing_key_blocks_run_before_starting_worker(monkeypatch):
    monkeypatch.delenv("RAG_JUDGE_API_KEY", raising=False)
    dashboard._publish_run_error("reset")
    result = dashboard.start_evaluation(
        n_clicks=1, api_base="http://localhost:9000", llm_model_id=None, top_k=None,
        max_questions=None, use_judge_flags=["enabled"], retrieval_only_flags=[],
        judge_base_url="https://judge.example/v1", judge_model="model",
        cases=[{"case_id": "one", "question": "Question"}], corpus_docs=None,
        adapter_kind="generic", judge_api_key="",
    )
    assert result["job_id"] is None
    assert "API key is required" in dashboard._snapshot_run_state()["live_status"]


def test_dashboard_passes_key_only_to_worker_and_not_run_state(monkeypatch):
    captured = {}
    class Thread:
        def __init__(self, target, args, daemon):
            captured["args"] = args
        def start(self):
            pass
    monkeypatch.setattr(dashboard.threading, "Thread", Thread)
    dashboard._publish_run_error("reset")
    result = dashboard.start_evaluation(
        n_clicks=1, api_base="http://localhost:9000", llm_model_id=None, top_k=None,
        max_questions=None, use_judge_flags=["enabled"], retrieval_only_flags=[],
        judge_base_url="https://judge.example/v1", judge_model="model",
        cases=[{"case_id": "one", "question": "Question"}], corpus_docs=None,
        adapter_kind="generic", judge_api_key="dashboard-input-secret",
    )
    assert result["job_id"]
    config = next(arg for arg in captured["args"] if isinstance(arg, JudgeConfig))
    assert config.api_key == "dashboard-input-secret"
    assert "dashboard-input-secret" not in json.dumps(result) + json.dumps(dashboard._snapshot_run_state()) + repr(config)
    dashboard._publish_run_error("reset")
