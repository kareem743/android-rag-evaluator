from __future__ import annotations

import judge


def test_judge_prompt_can_target_a_nonmedical_application():
    prompt = judge._build_judge_prompt("q", "gt", "answer", [], domain="software documentation")
    assert "software documentation RAG system" in prompt
    assert "medical" not in prompt


def test_extract_first_json_object_handles_extra_text():
    result = judge.extract_first_json_object('prefix {"answer": {"score": 0.9}, "retrieval": {}} suffix')

    assert result["answer"]["score"] == 0.9


def test_retrieval_metrics_are_deterministic_from_chunk_labels():
    result = judge.calculate_retrieval_metrics(
        [
            {"rank": 1, "relevance": 1},
            {"rank": 2, "relevance": 2},
            {"rank": 3, "relevance": 0},
            {"rank": 4, "relevance": 2},
            {"rank": 5, "relevance": 1},
        ]
    )

    assert result["retrieval_hit_at_5"] == 1
    assert result["retrieval_precision_at_5"] == 0.4
    assert result["retrieval_soft_precision_at_5"] == 0.6
    assert result["retrieval_mrr_at_5"] == 0.5
    assert 0 < result["retrieval_ndcg_at_5"] <= 1


def test_evaluate_with_ollama_allows_supported_expanded_answer(monkeypatch):
    class FakeResponse:
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": """
                {
                  "needed_facts": ["history", "vital signs", "head-to-toe exam"],
                  "retrieval": {
                    "verdict": "pass",
                    "chunk_relevance": [
                      {"rank": 1, "relevance": 2, "reason": "Contains history."},
                      {"rank": 2, "relevance": 2, "reason": "Contains vital signs."},
                      {"rank": 3, "relevance": 2, "reason": "Contains head-to-toe exam."}
                    ],
                    "found_facts": ["history", "vital signs", "head-to-toe exam"],
                    "missing_facts": [],
                    "distractor_chunks": []
                  },
                  "answer": {
                    "score": 0.92,
                    "verdict": "pass",
                    "missing_facts": [],
                    "unsupported_claims": [],
                    "dangerous_claims": [],
                    "explanation": "Correct expanded answer grounded in retrieved chunks."
                  }
                }
                """
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return FakeResponse().json()

    monkeypatch.setattr(judge.ollama, "Client", FakeClient)

    result = judge.evaluate_with_ollama(
        question="What is a secondary survey?",
        ground_truth="A detailed assessment including history, vital signs, and a head-to-toe exam.",
        generated_answer="A secondary survey is a detailed assessment with history, vital signs, a head-to-toe exam, and first aid.",
        retrieved_chunks=[
            {"contentSnippet": "History is gathered during the secondary survey.", "score": 0.9},
            {"contentSnippet": "Record vital signs.", "score": 0.8},
            {"contentSnippet": "Perform a head-to-toe exam.", "score": 0.7},
        ],
    )

    assert result["judge_status"] == "ok"
    assert result["pass"] is True
    assert result["retrieval"]["verdict"] == "pass"
    assert result["retrieval"]["metrics"]["retrieval_hit_at_5"] == 1
    assert result["answer"]["verdict"] == "pass"
    assert result["answer"]["dangerous_claims"] == []


def test_evaluate_with_ollama_returns_structured_error_on_bad_json(monkeypatch):
    class FakeResponse:
        text = "not json"

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "not json"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return FakeResponse().json()

    monkeypatch.setattr(judge.ollama, "Client", FakeClient)

    result = judge.evaluate_with_ollama(
        question="q",
        ground_truth="gt",
        generated_answer="answer",
        retrieved_chunks=[],
    )

    assert result["judge_status"] == "error"
    assert result["answer"]["score"] is None
    assert result["retrieval"]["metrics"]["retrieval_ndcg_at_5"] is None
