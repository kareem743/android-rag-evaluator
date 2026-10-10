from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import requests

import client
import judge


JUDGMENT = {
    "needed_facts": ["evidence"],
    "retrieval": {"verdict": "pass", "chunk_relevance": [{"rank": 1, "relevance": 2}],
                  "found_facts": ["evidence"], "missing_facts": [], "distractor_chunks": []},
    "answer": {"score": 0.92, "verdict": "pass", "missing_facts": [],
               "unsupported_claims": [], "dangerous_claims": [], "explanation": "Grounded."},
}


def test_judge_prompt_can_target_a_nonmedical_application():
    prompt = judge._build_judge_prompt("q", "gt", "answer", [], domain="software documentation")
    assert "software documentation RAG system" in prompt
    assert "medical" not in prompt


def test_extract_first_json_object_handles_extra_text():
    result = judge.extract_first_json_object('prefix {"answer": {"score": 0.9}, "retrieval": {}} suffix')
    assert result["answer"]["score"] == 0.9


def test_retrieval_metrics_are_deterministic_from_chunk_labels():
    result = judge.calculate_retrieval_metrics([
        {"rank": 1, "relevance": 1}, {"rank": 2, "relevance": 2},
        {"rank": 3, "relevance": 0}, {"rank": 4, "relevance": 2}, {"rank": 5, "relevance": 1},
    ])
    assert result["retrieval_hit_at_5"] == 1
    assert result["retrieval_precision_at_5"] == 0.4
    assert result["retrieval_soft_precision_at_5"] == 0.6
    assert result["retrieval_mrr_at_5"] == 0.5
    assert 0 < result["retrieval_ndcg_at_5"] <= 1


@pytest.fixture
def judge_api(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    requests_received = []
    state = {"status": 200, "body": {"choices": [{"message": {"content": json.dumps(JUDGMENT)}}]}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_received.append((self.path, self.headers.get("Authorization"), payload))
            if self.path == "/evaluate":
                body = {"generated_answer": "The guide supplies evidence.",
                        "retrieved_chunks": [{"text": "The guide supplies evidence.", "source": "Guide"}],
                        "total_latency_ms": 10}
                status = 200
            else:
                body, status = state["body"], state["status"]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests_received, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def evaluate(config):
    return judge.evaluate_with_api("Question", "Expected answer", "Expanded answer",
                                   [{"text": "Evidence", "source": "Guide"}], config)


def test_api_judge_authenticates_and_preserves_scoring_over_real_http(judge_api):
    base, received, _ = judge_api
    config = judge.JudgeConfig(base + "/custom/v1/", "provider-model", "fake-api-secret")
    result = evaluate(config)
    assert result["judge_status"] == "ok"
    assert result["pass"] is True
    assert result["answer"]["score"] == 0.92
    assert result["retrieval"]["metrics"]["retrieval_hit_at_5"] == 1
    path, auth, payload = received[0]
    assert path == "/custom/v1/chat/completions"
    assert auth == "Bearer fake-api-secret"
    assert payload["model"] == "provider-model"
    assert payload["stream"] is False
    assert "Question" in payload["messages"][0]["content"]
    assert "fake-api-secret" not in json.dumps(payload)
    assert "fake-api-secret" not in repr(config) + json.dumps(config.report_metadata()) + json.dumps(result)


@pytest.mark.parametrize("body", [
    {"choices": [{"message": {"content": "not json"}}]},
    {"choices": [{"message": {"content": None, "refusal": "Cannot judge"}}]},
    {"choices": []}, {"unexpected": True},
    {"choices": [{"message": {"content": '{"answer": {}, "retrieval": {}}'}}]},
])
def test_malformed_api_judgments_keep_scores_unavailable(judge_api, body):
    base, _, state = judge_api
    state["body"] = body
    result = evaluate(judge.JudgeConfig(base + "/v1", "model", "secret"))
    assert result["judge_status"] == "error"
    assert result["pass"] is None
    assert result["answer"]["score"] is None
    assert result["retrieval"]["metrics"]["retrieval_ndcg_at_5"] is None


@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_api_errors_do_not_record_provider_body_or_key(judge_api, status, capfd):
    base, _, state = judge_api
    state.update(status=status, body={"error": "echo fake-api-secret"})
    result = evaluate(judge.JudgeConfig(base + "/v1", "model", "fake-api-secret"))
    assert result["pass"] is None
    assert f"HTTP {status}" in result["error_message"]
    captured = capfd.readouterr()
    assert "fake-api-secret" not in json.dumps(result) + captured.out + captured.err


def test_network_timeout_does_not_leak_exception_details(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("Authorization: Bearer fake-api-secret")
    monkeypatch.setattr(judge.requests, "post", timeout)
    result = evaluate(judge.JudgeConfig("https://judge.example/v1", "model", "fake-api-secret", timeout=2))
    assert result["judge_status"] == "error"
    assert "timed out" in result["error_message"]
    assert "fake-api-secret" not in json.dumps(result)


def test_provider_echoed_key_is_redacted(judge_api, capfd):
    base, _, state = judge_api
    judgment = {**JUDGMENT, "answer": {**JUDGMENT["answer"], "explanation": "echo fake-api-secret"}}
    state["body"] = {"choices": [{"message": {"content": json.dumps(judgment)}}]}
    result = evaluate(judge.JudgeConfig(base + "/v1", "model", "fake-api-secret"))
    assert "[REDACTED]" in result["answer"]["explanation"]
    captured = capfd.readouterr()
    assert "fake-api-secret" not in json.dumps(result) + captured.out + captured.err


def test_config_uses_general_environment_settings(monkeypatch):
    monkeypatch.setenv("CUSTOM_JUDGE_KEY", "secret")
    monkeypatch.setenv("RAG_JUDGE_BASE_URL", "https://provider.example/api/v2")
    monkeypatch.setenv("RAG_JUDGE_MODEL", "available-model")
    config = judge.make_judge_config(api_key_env="CUSTOM_JUDGE_KEY")
    assert config.endpoint == "https://provider.example/api/v2/chat/completions"
    assert config.model == "available-model"
    assert config.api_key == "secret"


@pytest.mark.parametrize("base", ["https://user:pass@provider.example/v1", "https://provider.example/v1?key=secret",
                                  "https://provider.example/v1#secret", "http://provider.example/v1"])
def test_config_rejects_credential_urls_and_nonlocal_plaintext(base):
    with pytest.raises(ValueError):
        judge.JudgeConfig(base, "model", "secret")


def test_config_requires_key_model_and_positive_timeout():
    for model, key, timeout in [("", "key", 180), ("model", "", 180), ("model", "key", 0),
                                ("model", "key", float("nan")), ("model", "key\nvalue", 180)]:
        with pytest.raises(ValueError):
            judge.JudgeConfig("https://provider.example/v1", model, key, timeout)


def test_redirects_are_disabled_and_full_endpoint_is_accepted(monkeypatch):
    received = {}
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": [{"type": "text", "text": json.dumps(JUDGMENT)}]}}]}
    def post(url, **kwargs):
        received.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(judge.requests, "post", post)
    assert evaluate(judge.JudgeConfig("https://judge.example/v1/chat/completions", "model", "key"))["pass"] is True
    assert received["url"] == "https://judge.example/v1/chat/completions"
    assert received["allow_redirects"] is False


def test_cli_with_actual_app_and_judge_http_requests_writes_compatible_reports(judge_api, tmp_path, monkeypatch, capsys):
    base, received, _ = judge_api
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{"case_id": "one", "question": "Question", "ground_truth_answer": "Evidence"}]))
    output = tmp_path / "run.jsonl"
    monkeypatch.setenv("RAG_JUDGE_API_KEY", "fake-api-secret")
    monkeypatch.setattr("sys.argv", ["client.py", "--adapter", "generic", "--api-base", base,
                                    "--dataset", str(dataset), "--judge-api-base", base + "/v1",
                                    "--judge-model", "test-model", "--output", str(output)])
    assert client.main() == 0
    report = json.loads(output.with_suffix(".json").read_text())
    assert [r[0] for r in received] == ["/evaluate", "/v1/chat/completions"]
    assert report["metadata"]["judge_enabled"] is True
    assert report["metadata"]["judge_model"] == "test-model"
    assert json.loads(output.read_text())["pass"] is True
    assert "fake-api-secret" not in output.read_text() + json.dumps(report) + capsys.readouterr().out


def test_batch_continues_after_a_judge_api_failure(judge_api):
    from adapters import AdapterProfile, HttpRagAdapter
    from runner import iter_evaluations
    base, _, state = judge_api
    state["status"] = 429
    config = judge.JudgeConfig(base + "/v1", "model", "secret")
    def judge_case(case, response):
        result = client.evaluate_case_with_judge(case, response, judge_config=config)
        state["status"] = 200
        return result
    rows = list(iter_evaluations(HttpRagAdapter(base, AdapterProfile()), [
        {"case_id": "one", "question": "Question", "ground_truth_answer": "Evidence"},
        {"case_id": "two", "question": "Question", "ground_truth_answer": "Evidence"},
    ], judge_case))
    assert rows[0]["judge"]["judge_status"] == "error"
    assert rows[0]["pass"] is None
    assert rows[1]["pass"] is True
