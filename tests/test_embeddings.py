from types import SimpleNamespace

import pytest
from google.genai import errors

from hr_assistant.rag.embeddings import EmbeddingUnavailableError, GeminiEmbeddings


class FakeModels:
    def __init__(self, fail_with=None, fail_times=0):
        self.requests = []
        self.fail_with = fail_with
        self.fail_times = fail_times

    def embed_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.fail_times:
            self.fail_times -= 1
            raise self.fail_with
        texts = [c.parts[0].text if hasattr(c, "parts") else c for c in (contents if isinstance(contents, list) else [contents])]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(t)), 0.0, 1.0]) for t in texts])


def make(models, **kwargs):
    return GeminiEmbeddings(model="gemini-embedding-2", dimensions=3, client=SimpleNamespace(models=models), sleep=lambda s: None, **kwargs)


def sent_texts(request):
    return [c.parts[0].text for c in request["contents"]]


def test_documents_use_title_prefix_and_one_vector_each():
    models = FakeModels()
    vectors = make(models).embed_documents(["Kebijakan Cuti > Cuti Tahunan\n12 hari kerja per tahun."])
    assert len(vectors) == 1
    assert sent_texts(models.requests[0]) == ["title: Kebijakan Cuti > Cuti Tahunan | text: 12 hari kerja per tahun."]
    assert models.requests[0]["config"].output_dimensionality == 3


def test_query_uses_search_prefix():
    models = FakeModels()
    make(models).embed_query("berapa jatah cuti?")
    assert sent_texts(models.requests[0]) == ["task: search result | query: berapa jatah cuti?"]


def test_documents_are_batched():
    models = FakeModels()
    vectors = make(models, batch_size=2).embed_documents([f"Judul\nisi {i}" for i in range(5)])
    assert len(vectors) == 5
    assert [len(r["contents"]) for r in models.requests] == [2, 2, 1]


def test_rate_limit_retried_then_raises():
    error = errors.ClientError(429, {"error": {"code": 429, "message": "kuota habis", "status": "RESOURCE_EXHAUSTED"}})
    models = FakeModels(fail_with=error, fail_times=10)
    with pytest.raises(EmbeddingUnavailableError, match="kuota habis"):
        make(models, max_retries=2).embed_query("x")
    assert len(models.requests) == 3


def test_rate_limit_recovers():
    error = errors.ClientError(429, {"error": {"code": 429, "message": "kuota habis", "status": "RESOURCE_EXHAUSTED"}})
    models = FakeModels(fail_with=error, fail_times=1)
    vector = make(models).embed_query("x")
    assert len(models.requests) == 2 and len(vector) == 3


def test_bad_request_not_retried():
    error = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    models = FakeModels(fail_with=error, fail_times=10)
    with pytest.raises(EmbeddingUnavailableError):
        make(models).embed_query("x")
    assert len(models.requests) == 1
