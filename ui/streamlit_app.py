"""Streamlit UI for the RAG Document Assistant (talks to the FastAPI backend)."""

import html
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    .cite {display:inline-block;background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;border-radius:4px;
           padding:2px 8px;font-size:.85rem;font-weight:600;margin:2px;}
    .src {background:rgba(127,127,127,.08);border-left:4px solid #3B82F6;padding:8px 12px;margin-bottom:8px;
          border-radius:0 6px 6px 0;font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, timeout: int = 5) -> Optional[Any]:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def render_sources(sources: List[Dict[str, Any]]) -> None:
    for i, s in enumerate(sources, start=1):
        # PDF text is untrusted: escape it before injecting into HTML.
        st.markdown(
            f'<div class="src"><b>Source {i}:</b> {html.escape(s.get("citation_tag", ""))} &nbsp;|&nbsp; '
            f'score {s.get("score", 0):.3f}<br/><i>{html.escape(s.get("text", ""))}</i></div>',
            unsafe_allow_html=True,
        )


def render_answer(msg: Dict[str, Any]) -> None:
    st.markdown(msg["content"])
    if msg.get("citations"):
        st.markdown("".join(f'<span class="cite">{html.escape(c)}</span>' for c in msg["citations"]),
                    unsafe_allow_html=True)
    if msg.get("warning"):
        st.caption(f"⚠️ {msg['warning']}")
    if msg.get("sources"):
        with st.expander(f"📚 Sources ({len(msg['sources'])})"):
            render_sources(msg["sources"])


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", "citations": [], "sources": [],
        "content": "Upload one or more PDFs, then ask me questions. I answer only from your documents and cite the page.",
    }]

health = api_get("/health")
docs: List[Dict[str, Any]] = api_get("/documents") or []

with st.sidebar:
    st.title("Settings")
    if health:
        st.success(f"Backend connected · {health['documents_count']} docs · {health['chunks_count']} chunks")
        st.caption(f"Embeddings: {health['embedding_model']} · LLM: {health['llm_provider']}"
                   + (f" · Reranker: {health['reranker']}" if health.get("reranker") else ""))
    else:
        st.error(f"Backend offline. Start it with: uvicorn app.api:app --port 8000  (expected at {API_URL})")

    st.subheader("Chunking (applies to new uploads)")
    strategy = st.radio("Strategy", ["fixed", "semantic"], horizontal=True)
    chunk_size = st.slider("Chunk size (chars)", 200, 1200, 500, 50)
    overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
    sem_threshold = st.slider("Semantic distance threshold", 0.10, 0.80, 0.35, 0.05,
                              disabled=(strategy != "semantic"))

    st.subheader("Retrieval")
    mode_label = st.selectbox("Mode", ["Hybrid (BM25 + vector)", "Vector only", "BM25 only"])
    mode = {"Hybrid (BM25 + vector)": "hybrid", "Vector only": "vector", "BM25 only": "bm25"}[mode_label]
    alpha = st.slider("Vector weight (alpha)", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "hybrid"),
                      help="1.0 = vector only, 0.0 = keyword only")
    top_k = st.slider("Top-K chunks", 1, 10, 4)
    use_rerank = st.checkbox("Cross-encoder rerank (if enabled on server)", value=True)

    st.subheader("Documents")
    selected_ids: List[str] = []
    if docs:
        options = {f"{d['filename']} ({d['total_pages']} pp)": d["doc_id"] for d in docs}
        chosen = st.multiselect("Ask only about (empty = all)", list(options))
        selected_ids = [options[c] for c in chosen]
        for d in docs:
            c1, c2 = st.columns([4, 1])
            c1.write(f"📄 {d['filename']} · {d['chunk_count']} chunks")
            if c2.button("🗑️", key=f"del_{d['doc_id']}"):
                try:
                    requests.delete(f"{API_URL}/documents/{d['doc_id']}", timeout=10)
                except requests.RequestException as exc:
                    st.error(str(exc))
                st.rerun()
    else:
        st.info("No documents indexed yet.")

st.title("📄 RAG Document Assistant")
st.caption("Hybrid BM25 + vector retrieval · answers grounded in your PDFs · citations as [file, p. X]")

with st.expander("📤 Upload PDFs", expanded=not docs):
    files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Process and index", type="primary"):
        with st.spinner("Parsing, chunking and indexing..."):
            try:
                r = requests.post(
                    f"{API_URL}/upload", timeout=300,
                    files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files],
                    data={"chunking_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": overlap,
                          "semantic_threshold": sem_threshold},
                )
                if r.status_code == 201:
                    st.success(r.json()["message"])
                    st.rerun()
                else:
                    st.error(r.json().get("detail", r.text))
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_answer(m)

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "citations": [], "sources": []})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            payload = {"question": question, "retrieval_mode": mode, "top_k": top_k,
                       "alpha": alpha if mode == "hybrid" else None,
                       "rerank": use_rerank, "doc_ids": selected_ids or None}
            try:
                r = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
                if r.status_code == 200:
                    d = r.json()
                    msg = {"role": "assistant", "content": d["answer"], "citations": d["citations"],
                           "sources": d["sources"],
                           "warning": None if d["is_grounded"] or d["refused"] else
                           "This answer has no verifiable citation. Check the sources below."}
                else:
                    msg = {"role": "assistant", "content": f"Error: {r.json().get('detail', r.text)}",
                           "citations": [], "sources": []}
            except requests.RequestException as exc:
                msg = {"role": "assistant", "content": f"Could not reach the backend: {exc}",
                       "citations": [], "sources": []}
        render_answer(msg)
    st.session_state.messages.append(msg)
