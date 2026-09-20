import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_documents  # noqa: E402
from app.embeddings import HashEmbedder  # noqa: E402
from app.ingestion import parse_pdf_file  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402

SAMPLE_PDF = ROOT / "data" / "sample" / "enterprise_security_policy.pdf"


@pytest.fixture(scope="session")
def sample_pdf_path():
    if not SAMPLE_PDF.exists():
        from data.sample.generate_sample_pdf import main
        main()
    return SAMPLE_PDF


@pytest.fixture()
def pages(sample_pdf_path):
    return parse_pdf_file(sample_pdf_path)


@pytest.fixture()
def retriever(tmp_path, pages):
    r = HybridRetriever(embedder=HashEmbedder(), persist_directory=str(tmp_path / "chroma"),
                        collection_name="test", min_cosine_score=0.10)
    r.add_chunks(chunk_documents(pages, "fixed", 500, 100))
    return r
