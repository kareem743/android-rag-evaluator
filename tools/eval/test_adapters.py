from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
import requests

import client
from adapters import AdapterProfile, BRITE_PROFILE, HttpRagAdapter, load_profile
from judge import disabled_judge_result
from runner import iter_evaluations

CASE = {"case_id": "one", "question": "Where is the evidence?", "ground_truth_answer": "In the guide."}


@pytest.fixture
def alternate_api(monkeypatch):
    # Exercise the real HTTP transport without depending on a phone or Ollama.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, body, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            received.append((self.path, None))
            self.send_json({"ready": True})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            if payload.get("query") == "fail":
                self.send_json({"error": "busy"}, 503)
            elif payload.get("query") == "invalid":
                self.send_json({"result": {"answer": "Incomplete API output"}})
            else:
                self.send_json({"result": {"answer": "In the guide.", "documents": [
                    {"body": "The evidence is in the guide.", "metadata": {"title": "Guide"}, "similarity": 0.9}
                ]}, "timings": {"total_ms": 17}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def example_profile():
    return load_profile(Path(__file__).parent / "profiles/other_android.example.json")


def test_alternate_android_contract_over_real_http(alternate_api):
    base, received = alternate_api
    adapter = HttpRagAdapter(base, example_profile())
    assert adapter.check_health() == {"ready": True}
    rows = list(iter_evaluations(adapter, [CASE], lambda case, response: disabled_judge_result(), top_k=3))
    assert received == [("/status", None), ("/ask", {"requestId": "one", "query": CASE["question"], "options": {"limit": 3}})]
    response = rows[0]["response"]
    assert response["generated_answer"] == "In the guide."
    assert response["retrieved_chunks"][0]["text"] == "The evidence is in the guide."
    assert response["retrieved_chunks"][0]["source"] == "Guide"
    assert response["retrieved_chunks"][0]["score"] == 0.9
    assert response["total_latency_ms"] == 17
    assert response["adapter_raw_response"]["result"]["answer"] == "In the guide."


def test_generic_adapter_does_not_require_android_health_fields():
    assert HttpRagAdapter("http://localhost:9000", AdapterProfile()).check_health() == {"status": "skipped"}


def test_brite_adapter_keeps_original_contract():
    payloads = []

    def evaluate(base, payload, timeout):
        payloads.append(payload)
        return {"generated_answer": "Answer", "retrieved_chunks": [{"contentSnippet": "Evidence"}],
                "runtime": {"retrieval_confidence": "high"}}

    adapter = HttpRagAdapter("http://localhost:9000", health_call=lambda base: {"status": "ok", "runtime_ready": True},
                             evaluate_call=evaluate)
    assert adapter.check_health()["runtime_ready"] is True
    result = adapter.evaluate(adapter.build_request(CASE, top_k=5, model_id="model"))
    assert payloads[0]["case_id"] == "one"
    assert payloads[0]["model_id"] == "model"
    assert result["runtime"]["retrieval_confidence"] == "high"


def test_brite_rejects_unready_runtime():
    adapter = HttpRagAdapter("http://localhost:9000", health_call=lambda base: {"status": "ok", "runtime_ready": False})
    with pytest.raises(RuntimeError, match="runtime_ready"):
        adapter.check_health()


def test_shared_runner_records_api_and_schema_errors_then_continues(alternate_api):
    base, _ = alternate_api
    adapter = HttpRagAdapter(base, example_profile())
    cases = [{**CASE, "question": q} for q in ["fail", "invalid", "works"]]
    rows = list(iter_evaluations(adapter, cases, lambda case, response: disabled_judge_result()))
    assert rows[0]["response"]["error"]["code"] == "http_503"
    assert rows[0]["judge"]["judge_status"] == "api_error"
    assert "retrieved_chunks" in rows[1]["response"]["error"]["message"]
    assert rows[2]["response"]["generated_answer"] == "In the guide."


@pytest.mark.parametrize("bad", [
    [], {"evaluate_path": "https://elsewhere.example/evaluate"},
    {"health_expect": {"ready": True}}, {"request_fields": {"question": None}},
    {"response_fields": {"retrieved_chunks": None}}, {"name": 123},
    {"request_fields": {"unknown": "x"}}, {"response_fields": {"generated_answer": 123}},
    {"headers_env": {"Authorization": 12}}, {"typo": "value"},
])
def test_invalid_profiles_are_rejected(bad):
    with pytest.raises(ValueError):
        AdapterProfile.from_dict(bad)


def test_unsupported_controls_fail_before_any_query(alternate_api):
    base, received = alternate_api
    adapter = HttpRagAdapter(base, example_profile())
    with pytest.raises(ValueError, match="model_id"):
        list(iter_evaluations(adapter, [CASE], lambda *_: {}, model_id="unsupported"))
    with pytest.raises(ValueError, match="retrieval_only"):
        adapter.build_request(CASE, retrieval_only=True)
    with pytest.raises(ValueError, match="BRITE adapter only"):
        adapter.upload_corpus("unused.txt")
    assert received == []


def test_headers_are_resolved_from_environment_only(monkeypatch):
    adapter = HttpRagAdapter("http://localhost:9000", AdapterProfile.from_dict({"headers_env": {"Authorization": "RAG_AUTH"}}))
    monkeypatch.delenv("RAG_AUTH", raising=False)
    with pytest.raises(ValueError, match="Missing environment variable"):
        adapter._headers()
    monkeypatch.setenv("RAG_AUTH", "Bearer test-secret")
    assert adapter._headers() == {"Authorization": "Bearer test-secret"}
    assert "test-secret" not in repr(adapter.profile)


def test_shared_runner_records_timeout_and_judge_failure():
    def evaluate(base, payload, timeout):
        if payload["question"] == "timeout":
            raise requests.exceptions.ReadTimeout()
        return {"generated_answer": "Answer", "retrieved_chunks": []}

    def failing_judge(*args):
        raise RuntimeError("judge offline")

    adapter = HttpRagAdapter("http://localhost:9000", evaluate_call=evaluate)
    rows = list(iter_evaluations(adapter, [{**CASE, "question": "timeout"}, CASE], failing_judge))
    assert rows[0]["response"]["error"]["code"] == "request_timeout"
    assert rows[1]["judge"]["judge_status"] == "error"
    assert rows[1]["pass"] is None


def test_cli_evaluates_alternate_api_and_writes_reports(alternate_api, tmp_path, monkeypatch):
    import sys

    base, received = alternate_api
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([CASE]))
    profile_path = Path(__file__).parent / "profiles/other_android.example.json"
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(sys, "argv", ["client.py", "--dataset", str(dataset), "--api-base", base,
                                     "--adapter-profile", str(profile_path), "--top-k", "3",
                                     "--no-ollama-judge", "--output", str(output)])
    assert client.main() == 0
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["metadata"]["adapter"] == "other-android-app"
    assert report["metadata"]["judge_domain"] == "general"
    assert report["cases"][0]["response"]["generated_answer"] == "In the guide."
    assert report["summary"]["scored_count"] == 0
    assert json.loads(output.read_text())["request"]["options"]["limit"] == 3
    assert received[1][0] == "/ask"
