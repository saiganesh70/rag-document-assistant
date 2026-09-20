from app.generation import REFUSAL, GroundedGenerator, extractive_answer, resolve_citations
from app.rag import answer_question, is_broad_question, is_greeting
from app.retrieval import RetrievalResult


class Canned:
    name = "canned"

    def __init__(self, text):
        self.text = text

    def complete(self, system, user):
        return self.text


class Boom:
    name = "boom"

    def complete(self, system, user):
        raise RuntimeError("api down")


def mk(text, page, fname="doc.pdf"):
    return RetrievalResult(f"c{page}", text, {"filename": fname, "page_number": page, "doc_id": "d"}, cosine_score=0.5)


R = [mk("Passwords need 14 characters.", 1), mk("Sessions time out after 15 minutes of inactivity.", 2)]


def test_labels_become_real_citations_and_bad_labels_dropped():
    text, cites = resolve_citations("14 characters [S1]. Also maybe [S9] and [S1, S2].", R)
    assert cites == ["doc.pdf, p. 1", "doc.pdf, p. 2"]
    assert "[S9]" not in text and "[doc.pdf, p. 1]" in text


def test_generator_uses_llm_answer_with_correct_page():
    g = GroundedGenerator(Canned("Sessions end after 15 minutes [S2]."))
    res = g.generate("idle timeout?", R)
    assert res.citations == ["doc.pdf, p. 2"] and res.is_grounded and not res.refused


def test_no_fake_citation_is_added():
    g = GroundedGenerator(Canned("Sessions end after 15 minutes."))
    res = g.generate("idle timeout?", R)
    assert res.citations == [] and res.is_grounded is False


def test_llm_refusal_and_empty_results_refuse():
    assert GroundedGenerator(Canned(REFUSAL)).generate("q", R).refused
    res = GroundedGenerator(None).generate("q", [])
    assert res.refused and res.answer == REFUSAL and res.sources == []


def test_llm_failure_falls_back_to_extractive():
    res = GroundedGenerator(Boom()).generate("How many characters must passwords have?", R)
    assert not res.refused and res.provider == "extractive" and "doc.pdf, p. 1" in res.citations


def test_extractive_cites_the_chunk_it_used():
    ans = extractive_answer("session idle timeout minutes", R)
    assert ans.endswith("[S2]")
    assert extractive_answer("zzz qqq", R) == REFUSAL


def test_question_routing():
    assert is_greeting("hi") and is_greeting("Hello!") and not is_greeting("hi, what is the password rule")
    assert is_broad_question("what is the pdf about") and is_broad_question("summarize this document")
    assert not is_broad_question("what is the lockout duration")


def test_greeting_and_broad_questions_end_to_end(retriever):
    g = GroundedGenerator(None)
    assert answer_question(retriever, g, "hi").provider == "none"
    res = answer_question(retriever, g, "what is this document about", doc_ids=None)
    assert not res.refused and res.citations
