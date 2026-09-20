"""One place that turns a question into an answer. Used by the API, the eval script and the tests."""

import re
from typing import List, Optional

from app.generation import GenerationResult, GroundedGenerator, REFUSAL
from app.retrieval import HybridRetriever

_GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b[\s!.?]*$", re.I)
_BROAD = re.compile(
    r"\b(what is (this|the) (pdf|document|file|paper|report)( about)?|what('s| is) (it|this) about|summari[sz]e|"
    r"summary|overview|give me an? (overview|summary)|tell me about (this|the) (pdf|document|file))\b", re.I)


def is_greeting(q: str) -> bool:
    return bool(_GREETING.match(q))


def is_broad_question(q: str) -> bool:
    return bool(_BROAD.search(q))


def answer_question(
    retriever: HybridRetriever,
    generator: GroundedGenerator,
    question: str,
    *,
    mode: str = "hybrid",
    top_k: int = 4,
    alpha: Optional[float] = None,
    rerank: Optional[bool] = None,
    doc_ids: Optional[List[str]] = None,
) -> GenerationResult:
    if is_greeting(question):
        return GenerationResult(
            question,
            "Hello! Upload a PDF and ask me a question about it. I'll answer from the document and cite the page.",
            provider="none",
        )
    broad = is_broad_question(question)
    if broad:
        # Broad questions rarely match any single chunk well, so use the opening pages plus top hits.
        results = retriever.overview_chunks(doc_ids=doc_ids, max_pages=2, limit=top_k + 2)
    else:
        results = retriever.search(question, top_k=top_k, mode=mode, alpha=alpha, rerank=rerank, doc_ids=doc_ids)
    return generator.generate(question, results, broad=broad)


__all__ = ["answer_question", "is_greeting", "is_broad_question", "REFUSAL"]
