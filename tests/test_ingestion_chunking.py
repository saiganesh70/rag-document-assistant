import pytest

from app.chunking import chunk_documents, fixed_chunk, split_sentences
from app.embeddings import HashEmbedder
from app.ingestion import CorruptPDFError, EmptyPDFError, parse_pdf_bytes


def test_parse_keeps_page_metadata(pages):
    assert len(pages) == 5
    assert [p.page_number for p in pages] == [1, 2, 3, 4, 5]
    assert all(p.total_pages == 5 and p.filename == "enterprise_security_policy.pdf" for p in pages)
    assert len({p.doc_id for p in pages}) == 1


def test_corrupt_and_empty_pdf_raise():
    with pytest.raises(CorruptPDFError):
        parse_pdf_bytes(b"this is not a pdf", "bad.pdf")
    with pytest.raises(EmptyPDFError):
        parse_pdf_bytes(b"", "empty.pdf")


def test_fixed_chunks_keep_metadata_and_size(pages):
    chunks = fixed_chunk(pages, chunk_size=300, chunk_overlap=50)
    assert chunks
    assert all(len(c.text) <= 300 for c in chunks)
    assert all({"doc_id", "filename", "page_number", "total_pages"} <= set(c.metadata) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_zero_overlap_is_allowed_and_bad_values_rejected(pages):
    assert fixed_chunk(pages, 300, 0)
    with pytest.raises(ValueError):
        fixed_chunk(pages, 300, 300)


def test_semantic_chunking_and_unknown_strategy(pages):
    emb = HashEmbedder()
    chunks = chunk_documents(pages, "semantic", semantic_threshold=0.6, embed_fn=emb.embed)
    assert chunks and all(c.metadata["page_number"] in range(1, 6) for c in chunks)
    with pytest.raises(ValueError):
        chunk_documents(pages, "nonsense")
    with pytest.raises(ValueError):
        chunk_documents(pages, "semantic")  # no embed_fn


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
