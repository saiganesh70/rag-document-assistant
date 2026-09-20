"""FastAPI backend: upload PDFs, ask questions, manage documents.

Endpoints are plain `def` (not `async def`) on purpose: embedding and LLM calls are blocking, and
FastAPI runs sync endpoints in a thread pool so one slow request does not freeze the whole server.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chunking import chunk_documents
from app.config import get_settings
from app.embeddings import build_embedder
from app.generation import GroundedGenerator, build_provider
from app.ingestion import PDFIngestionError, parse_pdf_bytes
from app.rag import answer_question
from app.retrieval import CrossEncoderReranker, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@lru_cache
def get_retriever() -> HybridRetriever:
    s = get_settings()
    embedder = build_embedder(s.embedding_provider, s.embedding_model, s.openai_api_key, s.openai_embedding_model)
    reranker = None
    if s.rerank_enabled:
        try:
            reranker = CrossEncoderReranker(s.rerank_model)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); continuing without it.", exc)
    return HybridRetriever(
        embedder=embedder, persist_directory=s.chroma_persist_directory, collection_name=s.chroma_collection_name,
        default_alpha=s.hybrid_alpha, fusion_method=s.fusion_method, reranker=reranker,
        min_cosine_score=s.min_cosine_score, min_rerank_score=s.min_rerank_score,
    )


@lru_cache
def get_generator() -> GroundedGenerator:
    s = get_settings()
    provider = build_provider(s.llm_provider, s.gemini_api_key, s.gemini_model, s.openai_api_key,
                              s.openai_model, s.ollama_url, s.ollama_model)
    return GroundedGenerator(provider)


app = FastAPI(title="RAG Document Assistant API", version="2.0.0",
              description="PDF question answering with hybrid retrieval and verifiable citations.")
app.add_middleware(
    CORSMiddleware, allow_origins=get_settings().cors_origin_list, allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)


# ------------------------------------------------------------------ models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieval_mode: str = Field("hybrid", pattern="^(hybrid|vector|bm25)$")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: int = Field(4, ge=1, le=20)
    rerank: Optional[bool] = None
    doc_ids: Optional[List[str]] = Field(None, description="Restrict the search to these documents")


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    sources: List[Dict[str, Any]]
    is_grounded: bool
    refused: bool
    provider: str
    latency_seconds: float


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    chunk_count: int
    pages_indexed: int


class UploadResponse(BaseModel):
    message: str
    processed_files: List[str]
    total_pages: int
    total_chunks: int
    chunking_strategy: str


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health(r: HybridRetriever = Depends(get_retriever), g: GroundedGenerator = Depends(get_generator)):
    s = get_settings()
    return {
        "status": "ok",
        "documents_count": len(r.list_documents()),
        "chunks_count": len(r.records),
        "embedding_model": r.embedder.name,
        "llm_provider": g.provider_name,
        "reranker": r.reranker.name if r.reranker else None,
        "default_chunking_strategy": s.default_chunking_strategy,
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload(
    files: List[UploadFile] = File(...),
    chunking_strategy: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    semantic_threshold: Optional[float] = Form(None),
    r: HybridRetriever = Depends(get_retriever),
):
    s = get_settings()
    strategy = chunking_strategy or s.default_chunking_strategy
    size = s.chunk_size if chunk_size is None else chunk_size
    overlap = s.chunk_overlap if chunk_overlap is None else chunk_overlap  # 0 is a valid overlap
    threshold = s.semantic_distance_threshold if semantic_threshold is None else semantic_threshold
    max_bytes = s.max_upload_mb * 1024 * 1024

    all_chunks, names, total_pages = [], [], 0
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"'{f.filename}' is not a PDF. Only PDF files are supported.")
        content = f.file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(413, f"'{f.filename}' is larger than {s.max_upload_mb} MB.")
        try:
            pages = parse_pdf_bytes(content, f.filename)
            chunks = chunk_documents(pages, strategy=strategy, chunk_size=size, chunk_overlap=overlap,
                                     semantic_threshold=threshold, embed_fn=r.embed)
        except PDFIngestionError as exc:
            raise HTTPException(400, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            logger.exception("Failed processing %s", f.filename)
            raise HTTPException(500, f"Failed to process '{f.filename}': {exc}")
        all_chunks.extend(chunks)
        names.append(f.filename)
        total_pages += len(pages)

    n = r.add_chunks(all_chunks)
    return UploadResponse(message=f"Indexed {len(names)} document(s) as {n} chunk(s).", processed_files=names,
                          total_pages=total_pages, total_chunks=n, chunking_strategy=strategy)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, r: HybridRetriever = Depends(get_retriever), g: GroundedGenerator = Depends(get_generator)):
    res = answer_question(r, g, req.question, mode=req.retrieval_mode, top_k=req.top_k, alpha=req.alpha,
                          rerank=req.rerank, doc_ids=req.doc_ids)
    return AskResponse(question=res.question, answer=res.answer, citations=res.citations, sources=res.sources,
                       is_grounded=res.is_grounded, refused=res.refused, provider=res.provider,
                       latency_seconds=res.latency_seconds)


@app.get("/documents", response_model=List[DocumentInfo])
def documents(r: HybridRetriever = Depends(get_retriever)):
    return r.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, r: HybridRetriever = Depends(get_retriever)):
    n = r.delete_document(document_id)
    if n == 0:
        raise HTTPException(404, f"Document '{document_id}' not found.")
    return {"message": f"Deleted '{document_id}' ({n} chunks).", "deleted_chunks": n}
