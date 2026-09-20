"""Grounded answer generation with verifiable citations.

How citations stay honest
- The prompt labels each retrieved chunk [S1], [S2], ...; the model cites those labels.
- Code maps labels back to "[filename, p. X]". Labels the model invented are dropped.
- Nothing is ever attached "just in case": if the model cites nothing valid, the answer is flagged
  as `is_grounded=False` instead of being given a fake citation.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.retrieval import RetrievalResult, tokenize

logger = logging.getLogger(__name__)

REFUSAL = "I couldn't find this in the documents."
_LABEL_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", re.IGNORECASE)

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer the question using ONLY the numbered sources provided.
Rules:
1. Write a clear, complete answer in one or more full sentences. Never answer with only a label or a single word.
2. After each fact, cite the supporting source label in square brackets, for example [S1] or [S2].
3. If the sources do not contain the answer, reply with exactly: {REFUSAL}
4. Do not use outside knowledge and do not guess."""


@dataclass
class GenerationResult:
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    is_grounded: bool = True
    refused: bool = False
    latency_seconds: float = 0.0
    provider: str = "extractive"


def build_context(results: List[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[S{i}] (file: {r.filename}, page {r.page_number})\n{r.text}" for i, r in enumerate(results, start=1)
    )


def resolve_citations(text: str, results: List[RetrievalResult]) -> Tuple[str, List[str]]:
    """Replace [S#] labels with [filename, p. X]; drop labels that do not exist."""
    seen: List[str] = []

    def repl(match: re.Match) -> str:
        tags = []
        for label in re.split(r"\s*,\s*", match.group(1)):
            idx = int(label[1:])
            if 1 <= idx <= len(results):
                r = results[idx - 1]
                tag = f"{r.filename}, p. {r.page_number}"
                tags.append(f"[{tag}]")
                if tag not in seen:
                    seen.append(tag)
        return " ".join(tags)

    cleaned = _LABEL_RE.sub(repl, text)
    return re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", cleaned)).strip(), seen


# ---------------------------------------------------------------- providers
class Provider:
    name = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self._model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.0, max_output_tokens=600),
        )
        return (resp.text or "").strip()


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model, temperature=0.0, max_tokens=600,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, url: str, model: str):
        self._url, self._model = url.rstrip("/"), model

    def complete(self, system: str, user: str) -> str:
        r = requests.post(
            f"{self._url}/api/chat", timeout=120,
            json={"model": self._model, "stream": False, "options": {"temperature": 0},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def build_provider(name: str, gemini_key: Optional[str], gemini_model: str, openai_key: Optional[str],
                   openai_model: str, ollama_url: str, ollama_model: str) -> Optional[Provider]:
    """Returns None for the offline extractive mode."""
    name = name.lower()
    if name == "auto":
        name = "gemini" if gemini_key else "openai" if openai_key else "extractive"
    try:
        if name == "gemini":
            if not gemini_key:
                raise ValueError("LLM_PROVIDER=gemini needs GEMINI_API_KEY")
            return GeminiProvider(gemini_key, gemini_model)
        if name == "openai":
            if not openai_key:
                raise ValueError("LLM_PROVIDER=openai needs OPENAI_API_KEY")
            return OpenAIProvider(openai_key, openai_model)
        if name == "ollama":
            return OllamaProvider(ollama_url, ollama_model)
    except Exception as exc:
        logger.error("Could not start LLM provider '%s' (%s); using extractive mode.", name, exc)
    return None


# ---------------------------------------------------------------- extractive (offline) mode
def extractive_answer(question: str, results: List[RetrievalResult], max_sentences: int = 3,
                      broad: bool = False) -> str:
    """Pick the best-matching sentences and cite the exact chunk each one came from."""
    q_tokens = set(tokenize(question))
    scored: List[Tuple[float, int, int, str]] = []
    for ci, r in enumerate(results, start=1):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r.text) if len(s.strip()) > 15]
        for si, s in enumerate(sentences):
            overlap = len(q_tokens & set(tokenize(s)))
            score = (1.0 if broad and ci <= 2 and si < 2 else 0.0) + overlap
            if score > 0:
                scored.append((score, ci, si, s))
    if not scored:
        return REFUSAL
    best = sorted(scored, key=lambda t: (-t[0], t[1], t[2]))[:max_sentences]
    best.sort(key=lambda t: (t[1], t[2]))
    return " ".join(f"{s.rstrip('.')}. [S{ci}]" for _, ci, _, s in best)


class GroundedGenerator:
    def __init__(self, provider: Optional[Provider] = None):
        self.provider = provider

    @property
    def provider_name(self) -> str:
        return self.provider.name if self.provider else "extractive"

    def generate(self, question: str, results: List[RetrievalResult], broad: bool = False) -> GenerationResult:
        t0 = time.time()
        sources = [r.to_dict() for r in results]
        if not results:
            return GenerationResult(question, REFUSAL, sources=[], refused=True,
                                    latency_seconds=round(time.time() - t0, 3), provider=self.provider_name)

        raw, used = "", self.provider_name
        if self.provider is not None:
            try:
                user_prompt = f"Sources:\n{build_context(results)}\n\nQuestion: {question}\n\nAnswer:"
                raw = self.provider.complete(SYSTEM_PROMPT, user_prompt)
            except Exception as exc:
                logger.error("LLM call failed (%s); falling back to extractive mode.", exc)
                raw, used = "", "extractive"
        if not raw:
            raw, used = extractive_answer(question, results, broad=broad), "extractive"

        if REFUSAL.lower().rstrip(".") in raw.lower():
            return GenerationResult(question, REFUSAL, sources=sources, refused=True,
                                    latency_seconds=round(time.time() - t0, 3), provider=used)

        answer, citations = resolve_citations(raw, results)
        return GenerationResult(
            question, answer, citations=citations, sources=sources,
            is_grounded=bool(citations), refused=False,
            latency_seconds=round(time.time() - t0, 3), provider=used,
        )
