import pytest

from app.chunking import chunk_documents
from app.embeddings import HashEmbedder
from app.retrieval import HybridRetriever, cosine, tokenize


class FakeReranker:
    """Scores chunks containing the word 'lockout' highest."""
    name = "fake"

    def score(self, query, texts):
        return [5.0 if "lockout" in t.lower() else -5.0 for t in texts]


def top_pages(results):
    return [r.page_number for r in results]


def test_tokenize_and_cosine():
    assert "the" not in tokenize("the password length")
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_all_modes_find_right_page(retriever, mode):
    res = retriever.search("How many failed login attempts lock an account?", top_k=3, mode=mode)
    assert 1 in top_pages(res)


def test_bm25_beats_confusable_terms(retriever):
    res = retriever.search("service account secret rotation", top_k=1, mode="bm25")
    assert "180" in res[0].text


def test_invalid_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")


def test_doc_filter_and_delete(tmp_path, pages):
    r = HybridRetriever(embedder=HashEmbedder(), persist_directory=str(tmp_path / "c"), collection_name="test_docs",
                        min_cosine_score=0.0)
    chunks = chunk_documents(pages, "fixed", 500, 100)
    r.add_chunks(chunks)
    doc_id = pages[0].doc_id
    assert r.search("password", doc_ids=["does-not-exist"]) == []
    assert r.search("password", doc_ids=[doc_id])
    assert r.list_documents()[0]["chunk_count"] == len(chunks)
    assert r.delete_document(doc_id) == len(chunks)
    assert r.search("password") == [] and r.delete_document(doc_id) == 0


def test_threshold_rejects_unrelated_query(retriever):
    retriever.min_cosine_score = 0.30
    assert retriever.search("zxqv wumpus hoverboard", top_k=3, mode="vector") == []
    assert retriever.search("zxqv wumpus hoverboard", top_k=3, mode="vector", apply_threshold=False)


def test_persistence_reloads_index(tmp_path, pages):
    path = str(tmp_path / "persist")
    r1 = HybridRetriever(embedder=HashEmbedder(), persist_directory=path, collection_name="persist_docs")
    r1.add_chunks(chunk_documents(pages, "fixed", 500, 100))
    r2 = HybridRetriever(embedder=HashEmbedder(), persist_directory=path, collection_name="persist_docs")
    assert len(r2.records) == len(r1.records) > 0
    assert r2.search("password length", top_k=2, mode="bm25")


def test_reranker_reorders_and_scores(retriever):
    retriever.reranker = FakeReranker()
    res = retriever.search("account rules", top_k=3, mode="hybrid", rerank=True)
    assert res and "lockout" in res[0].text.lower() and res[0].rerank_score > 0.9
    plain = retriever.search("account rules", top_k=3, mode="hybrid", rerank=False)
    assert all(r.rerank_score is None for r in plain)


@pytest.mark.parametrize("fusion", ["rrf", "minmax"])
def test_fusion_methods(retriever, fusion):
    retriever.fusion_method = fusion
    assert retriever.search("encryption AES-256 laptops", top_k=2, mode="hybrid", alpha=0.5)


def test_overview_chunks_are_first_pages(retriever):
    res = retriever.overview_chunks(max_pages=2, limit=5)
    assert res and all(r.page_number <= 2 for r in res)
