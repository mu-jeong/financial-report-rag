import pytest

from src.utils.ranker import OpenRouterReranker


class FakeResponse:
    status_code = 200
    text = "OK"

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_openrouter_reranker_posts_documents_and_maps_results(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }
        )

    monkeypatch.setattr("src.utils.ranker.requests.post", fake_post)

    reranker = OpenRouterReranker(
        model="cohere/rerank-v3.5",
        api_key="test-key",
        app_url="https://example.test",
        app_title="finance_llm_test",
        data_collection="deny",
        timeout=12,
    )
    passages = [
        {"text": "first", "meta": {"id": 1}, "score": 1.0},
        {"text": "second", "meta": {"id": 2}, "score": 2.0},
    ]

    result = reranker.rerank("query", passages, top_n=2)

    assert [item["text"] for item in result] == ["second", "first"]
    assert [item["rerank_score"] for item in result] == [0.9, 0.5]
    assert captured["url"] == "https://openrouter.ai/api/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["HTTP-Referer"] == "https://example.test"
    assert captured["headers"]["X-Title"] == "finance_llm_test"
    assert captured["timeout"] == 12
    assert captured["json"] == {
        "model": "cohere/rerank-v3.5",
        "query": "query",
        "documents": ["first", "second"],
        "top_n": 2,
        "provider": {"data_collection": "deny"},
    }


@pytest.mark.parametrize("top_n", [1, 2, 3, 5])
def test_duplicate_indexes_preserve_first_score_and_fill_unique_results(monkeypatch, top_n):
    monkeypatch.setattr(
        "src.utils.ranker.requests.post",
        lambda *args, **kwargs: FakeResponse({"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.7},
        ]}),
    )
    passages = [{"text": text} for text in ("first", "second", "third")]
    result = OpenRouterReranker(model="test", api_key="test").rerank("q", passages, top_n)
    assert [item["text"] for item in result] == ["second", "third", "first"][:top_n]
    assert result[0]["rerank_score"] == 0.9
    assert all("rerank_score" not in passage for passage in passages)


def test_boolean_index_is_not_a_document_index(monkeypatch):
    monkeypatch.setattr(
        "src.utils.ranker.requests.post",
        lambda *args, **kwargs: FakeResponse({"results": [
            {"index": True, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 0.9},
        ]}),
    )
    result = OpenRouterReranker(model="test", api_key="test").rerank(
        "q", [{"text": "first"}, {"text": "second"}], 2,
    )
    assert [item["text"] for item in result] == ["first", "second"]
