import pytest
from fastapi.testclient import TestClient

from app import api
from app.generation import GroundedGenerator


@pytest.fixture()
def client(retriever_empty):
    api.app.dependency_overrides[api.get_retriever] = lambda: retriever_empty
    api.app.dependency_overrides[api.get_generator] = lambda: GroundedGenerator(None)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture()
def retriever_empty(tmp_path):
    from app.embeddings import HashEmbedder
    from app.retrieval import HybridRetriever
    return HybridRetriever(embedder=HashEmbedder(), persist_directory=str(tmp_path / "api"),
                           collection_name="api", min_cosine_score=0.10)


def upload(client, path, **data):
    with open(path, "rb") as f:
        return client.post("/upload", files=[("files", (path.name, f.read(), "application/pdf"))], data=data)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents_count"] == 0


def test_upload_ask_list_delete_flow(client, sample_pdf_path):
    r = upload(client, sample_pdf_path, chunking_strategy="fixed", chunk_size="400", chunk_overlap="0")
    assert r.status_code == 201 and r.json()["total_pages"] == 5

    docs = client.get("/documents").json()
    assert len(docs) == 1
    doc_id = docs[0]["doc_id"]

    ans = client.post("/ask", json={"question": "How long does an account lockout last?"}).json()
    assert not ans["refused"] and "30 minutes" in ans["answer"]
    assert ans["citations"] == ["enterprise_security_policy.pdf, p. 1"]

    scoped = client.post("/ask", json={"question": "lockout", "doc_ids": ["nope"]}).json()
    assert scoped["refused"]

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_unanswerable_question_is_refused(client, sample_pdf_path):
    upload(client, sample_pdf_path)
    ans = client.post("/ask", json={"question": "zxqv wumpus hoverboards"}).json()
    assert ans["refused"] and ans["citations"] == []


def test_bad_uploads(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    assert upload(client, bad).status_code == 400
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not really a pdf")
    assert upload(client, fake).status_code == 400
    assert upload(client, fake, chunking_strategy="weird").status_code == 400


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "retrieval_mode": "bad"}).status_code == 422
