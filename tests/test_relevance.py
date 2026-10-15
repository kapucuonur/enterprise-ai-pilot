"""Relevance gate: weak, off-topic matches must not reach the LLM."""
from pathlib import Path

from fastapi.testclient import TestClient

from app import retrieval
from app.main import NO_ANSWER, ROOT, create_app
from app.retrieval import Chunk, Index, load_corpus

SALARY_Q = "What are the salary bands for senior engineer?"


class CountingLLM:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, question, contexts):
        self.calls += 1
        return "Answer [hr_salary_bands]."


def ask(client, key, q):
    return client.post("/ask", json={"question": q}, headers={"X-API-Key": key}).json()


def test_bob_salary_question_refuses_without_calling_llm(tmp_path):
    llm = CountingLLM()
    c = TestClient(create_app(llm=llm, audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", SALARY_Q)
    assert r["answer"] == NO_ANSWER
    assert r["sources"] == []
    assert llm.calls == 0
    assert '"decision": "no_context"' in (tmp_path / "a.jsonl").read_text()


def test_alice_still_gets_salary_answer_with_source(tmp_path):
    c = TestClient(create_app(audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-alice", SALARY_Q)
    assert r["sources"][0]["doc_id"] == "hr_salary_bands"
    assert "4500" in r["answer"]


FILLER = [
    "Quarterly town halls are recorded for later viewing.",
    "Expense reports are approved by the line manager.",
    "Visitors sign in at the front desk on arrival.",
    "Printers on each floor support badge release.",
    "Recycling bins are emptied every Friday afternoon.",
]


def _index(*texts):
    """Chunks under test plus unrelated filler, so IDF behaves like a real corpus."""
    return Index([Chunk(f"d{i}", f"Doc {i}", frozenset({"employee"}), t)
                  for i, t in enumerate([*texts, *FILLER])])


def test_single_generic_term_is_not_enough():
    idx = _index("The on-call engineer pages the senior team.", "Lunch is served at noon in the canteen.")
    assert idx.search("salary bands for senior engineer", {"employee"}) == []


def test_two_distinct_terms_pass():
    idx = _index("Backups of the database run nightly.", "Lunch is served at noon in the canteen.")
    hits = idx.search("when do database backups run", {"employee"})
    assert [c.doc_id for c, _ in hits] == ["d0"]


def test_half_of_a_short_question_passes():
    idx = _index("Parking permits are issued by facilities.", "Lunch is served at noon in the canteen.")
    hits = idx.search("parking rules", {"employee"})  # 1 of 2 terms = 50%
    assert [c.doc_id for c, _ in hits] == ["d0"]


def test_min_score_is_enforced(monkeypatch):
    idx = _index("Backups of the database run nightly.", "Lunch is served at noon in the canteen.")
    monkeypatch.setattr(retrieval, "MIN_SCORE", 1000.0)
    assert idx.search("when do database backups run", {"employee"}) == []


def test_gate_runs_after_acl():
    """A highly relevant chunk the user may not read is still invisible: ACL decides first."""
    idx = Index(load_corpus(ROOT / "data" / "docs"))
    assert all(c.doc_id != "hr_salary_bands" for c, _ in idx.search(SALARY_Q, {"employee", "engineering"}))
    assert idx.search(SALARY_Q, {"employee", "hr"})[0][0].doc_id == "hr_salary_bands"
