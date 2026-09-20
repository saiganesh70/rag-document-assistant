"""Benchmark retrieval modes (and optionally full answers) on eval/qa_pairs.json.

Usage
  python eval/evaluate.py                     # retrieval metrics only (fast, no LLM calls)
  python eval/evaluate.py --with-llm          # also grade end-to-end answers with your configured LLM
  python eval/evaluate.py --pdf a.pdf --pdf b.pdf --qa my_pairs.json
  python eval/evaluate.py --rerank            # adds the cross-encoder variant (downloads the model once)

Every ablation row changes ONE thing at a time, all other settings are identical, models are warmed
up before timing, and every percentage is printed with its count so small samples are obvious.
"""

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_documents  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embeddings import build_embedder  # noqa: E402
from app.generation import GroundedGenerator, build_provider  # noqa: E402
from app.ingestion import parse_pdf_file  # noqa: E402
from app.rag import answer_question  # noqa: E402
from app.retrieval import CrossEncoderReranker, HybridRetriever  # noqa: E402


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}% ({n}/{d})" if d else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", help="PDF(s) to index; default: the sample policy")
    ap.add_argument("--qa", default=str(ROOT / "eval" / "qa_pairs.json"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--with-llm", action="store_true", help="grade full answers using the configured LLM")
    ap.add_argument("--rerank", action="store_true", help="include the cross-encoder reranker variant")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between LLM calls (rate limits)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = ap.parse_args()

    s = get_settings()
    pdfs = [Path(p) for p in (args.pdf or [ROOT / "data" / "sample" / "enterprise_security_policy.pdf"])]
    qa: List[Dict[str, Any]] = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    answerable = [q for q in qa if q["is_answerable"]]
    unanswerable = [q for q in qa if not q["is_answerable"]]

    embedder = build_embedder(s.embedding_provider, s.embedding_model, s.openai_api_key, s.openai_embedding_model)
    if embedder.name == "hash":
        print("!! EMBEDDING_PROVIDER=hash: SMOKE TEST ONLY. Do not publish these numbers.\n")

    tmp = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        reranker = CrossEncoderReranker(s.rerank_model) if args.rerank else None
        retriever = HybridRetriever(
            embedder=embedder, persist_directory=tmp, collection_name="eval_docs", default_alpha=s.hybrid_alpha,
            fusion_method=s.fusion_method, reranker=reranker, min_cosine_score=s.min_cosine_score,
            min_rerank_score=s.min_rerank_score,
        )
        for pdf in pdfs:
            pages = parse_pdf_file(pdf)
            retriever.add_chunks(chunk_documents(pages, s.default_chunking_strategy, s.chunk_size, s.chunk_overlap,
                                                 s.semantic_distance_threshold, embed_fn=embedder.embed))
        print(f"Indexed {len(pdfs)} PDF(s) -> {len(retriever.records)} chunks. "
              f"{len(answerable)} answerable + {len(unanswerable)} unanswerable questions.\n")

        variants = [
            ("vector", dict(mode="vector", rerank=False)),
            ("bm25", dict(mode="bm25", rerank=False)),
            (f"hybrid (alpha={s.hybrid_alpha}, {s.fusion_method})", dict(mode="hybrid", rerank=False)),
        ]
        if reranker:
            variants.append(("hybrid + cross-encoder", dict(mode="hybrid", rerank=True)))

        provider = build_provider(s.llm_provider, s.gemini_api_key, s.gemini_model, s.openai_api_key,
                                  s.openai_model, s.ollama_url, s.ollama_model)
        generator = GroundedGenerator(provider)

        # warm-up so model loading never lands inside a timed section
        for _, cfg in variants:
            retriever.search("warm up", top_k=args.k, apply_threshold=False, **cfg)

        results: Dict[str, Any] = {"n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
                                   "k": args.k, "embedding_model": embedder.name,
                                   "llm": generator.provider_name if args.with_llm else None, "variants": {}}

        for name, cfg in variants:
            hits, rr, lat = 0, [], []
            kept_ans, rejected_unans = 0, 0
            for q in answerable:
                t0 = time.perf_counter()
                res = retriever.search(q["question"], top_k=args.k, apply_threshold=False, **cfg)
                lat.append(time.perf_counter() - t0)
                pages_ret = [r.page_number for r in res]
                if q["expected_page"] in pages_ret:
                    hits += 1
                    rr.append(1.0 / (pages_ret.index(q["expected_page"]) + 1))
                else:
                    rr.append(0.0)
                kept_ans += any(retriever.passes_threshold(r) for r in res)
            for q in unanswerable:
                res = retriever.search(q["question"], top_k=args.k, apply_threshold=False, **cfg)
                rejected_unans += not any(retriever.passes_threshold(r) for r in res)

            row: Dict[str, Any] = {
                "recall_at_k": pct(hits, len(answerable)), "mrr": round(statistics.mean(rr), 3),
                "answerable_kept_by_threshold": pct(kept_ans, len(answerable)),
                "unanswerable_rejected_by_threshold": pct(rejected_unans, len(unanswerable)),
                "avg_retrieval_ms": round(1000 * statistics.mean(lat), 1),
            }

            if args.with_llm:
                correct = cited_ok = false_refusals = unsupported = 0
                for q in answerable:
                    r = answer_question(retriever, generator, q["question"], top_k=args.k, **cfg)
                    time.sleep(args.delay)
                    if r.refused:
                        false_refusals += 1
                    elif any(p.lower() in r.answer.lower() for p in q["key_phrases"]):
                        correct += 1
                        cited_ok += any(f"p. {q['expected_page']}" in c for c in r.citations)
                for q in unanswerable:
                    r = answer_question(retriever, generator, q["question"], top_k=args.k, **cfg)
                    time.sleep(args.delay)
                    unsupported += not r.refused
                row.update({
                    "answer_correct": pct(correct, len(answerable)),
                    "correct_answers_with_right_page_citation": pct(cited_ok, len(answerable)),
                    "false_refusals": pct(false_refusals, len(answerable)),
                    "unsupported_answers_on_unanswerable": pct(unsupported, len(unanswerable)),
                })
            results["variants"][name] = row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics = list(next(iter(results["variants"].values())).keys())
    names = list(results["variants"].keys())
    print("| Metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m in metrics:
        print(f"| {m} | " + " | ".join(str(results['variants'][n][m]) for n in names) + " |")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
