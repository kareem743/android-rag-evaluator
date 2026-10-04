from __future__ import annotations

import base64
import json

import common


def test_parse_cases_text_from_json_object_normalizes_fields():
    raw = json.dumps(
        {
            "cases": [
                {
                    "caseId": "case-a",
                    "question": "What is the dose?",
                    "ground_truth": "Give 5 mg.",
                    "category": "dose",
                    "notes": "adult",
                }
            ]
        }
    )

    cases = common.parse_cases_text(raw, "cases.json")

    assert cases == [
        {
            "case_id": "case-a",
            "question": "What is the dose?",
            "ground_truth_answer": "Give 5 mg.",
            "category": "dose",
            "notes": "adult",
        }
    ]


def test_parse_cases_text_from_jsonl_skips_invalid_rows():
    raw = "\n".join(
        [
            json.dumps({"id": "case-1", "prompt": "Question 1", "answer": "Answer 1"}),
            json.dumps({"id": "case-2"}),
        ]
    )

    cases = common.parse_cases_text(raw, "cases.jsonl")

    assert len(cases) == 1
    assert cases[0]["case_id"] == "case-1"
    assert cases[0]["ground_truth_answer"] == "Answer 1"


def test_parse_corpus_bytes_supports_text_and_json_payloads():
    text_docs = common.parse_corpus_bytes(b"line one\nline two", "field-guide.txt")
    json_docs = common.parse_corpus_bytes(
        json.dumps(
            {
                "documents": [
                    {"id": "doc-1", "title": "Guide", "content": "Medical text"}
                ]
            }
        ).encode("utf-8"),
        "field-guide.json",
    )

    assert text_docs == [{"sourceId": "field-guide", "title": "field-guide.txt", "content": "line one\nline two"}]
    assert json_docs == [{"sourceId": "doc-1", "title": "Guide", "content": "Medical text"}]


def test_parse_upload_contents_and_write_outputs_round_trip(tmp_path):
    original = b"payload-data"
    contents = "data:text/plain;base64," + base64.b64encode(original).decode("ascii")

    decoded = common.parse_upload_contents(contents, "payload.txt")
    jsonl_path = common.write_jsonl(tmp_path / "run.jsonl", [{"a": 1}, {"b": 2}])
    json_path = common.write_json_report(
        tmp_path / "run.json",
        [{"pass": True, "semantic_score": 0.8, "answer_score": 0.6}],
        metadata={"dataset": "cases.json"},
    )

    assert decoded == original
    assert jsonl_path.read_text(encoding="utf-8").splitlines() == ['{"a": 1}', '{"b": 2}']
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "brite_eval_report_v1"
    assert report["metadata"]["dataset"] == "cases.json"
    assert report["summary"]["case_count"] == 1
    assert report["cases"][0]["semantic_score"] == 0.8


def test_call_health_and_call_evaluate_use_requests(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("bad response")

        def json(self):
            return self._payload

    def fake_get(url, timeout):
        captured["health"] = (url, timeout)
        return FakeResponse(200, {"status": "ok"})

    def fake_post(url, json, timeout):
        captured["evaluate"] = (url, json, timeout)
        return FakeResponse(400, {"error": {"code": "bad_request"}})

    monkeypatch.setattr(common.requests, "get", fake_get)
    monkeypatch.setattr(common.requests, "post", fake_post)

    health = common.call_health("http://127.0.0.1:9000")
    evaluate = common.call_evaluate("http://127.0.0.1:9000", {"case_id": "c1"})

    assert health == {"status": "ok"}
    assert evaluate == {"error": {"code": "bad_request"}}
    assert captured["health"] == ("http://127.0.0.1:9000/health", 180)
    assert captured["evaluate"] == ("http://127.0.0.1:9000/evaluate", {"case_id": "c1"}, 180)


def test_call_upload_file_streams_multipart_without_base64_json(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    corpus_path = tmp_path / "field-guide.txt"
    corpus_path.write_text("source text", encoding="utf-8")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"rag_id": "rag-1", "node_count": 3}

    def fake_post(url, data=None, files=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        captured["kwargs"] = kwargs
        filename, handle, mime_type = files["file"]
        captured["file"] = (filename, handle.read(), mime_type)
        return FakeResponse()

    monkeypatch.setattr(common.requests, "post", fake_post)

    response = common.call_upload_file("http://127.0.0.1:9000", corpus_path)

    assert response == {"rag_id": "rag-1", "node_count": 3}
    assert captured["url"] == "http://127.0.0.1:9000/corpus/file"
    assert captured["data"]["file_name"] == "field-guide.txt"
    assert "content_base64" not in captured["data"]
    assert captured["file"] == ("field-guide.txt", b"source text", "text/plain")
    assert captured["kwargs"] == {}
