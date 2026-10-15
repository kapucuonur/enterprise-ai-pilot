"""Citation checks: variants are normalised, unknown sources stripped, untraceable answers flagged."""
import pytest
from fastapi.testclient import TestClient

from app.main import _check_citations, create_app


class FixedLLM:
    name = "fixed"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, question, contexts):
        return self.reply


def ask(tmp_path, reply, question="How many vacation days?"):
    c = TestClient(create_app(llm=FixedLLM(reply), audit_path=tmp_path / "a.jsonl"))
    return c.post("/ask", json={"question": question}, headers={"X-API-Key": "key-bob"}).json()


@pytest.mark.parametrize("cite", ["[handbook]", "[doc_id: handbook]", "[doc: handbook]",
                                  "[Doc_ID:handbook]", "[source: handbook]"])
def test_variant_formats_are_normalised_and_pass(tmp_path, cite):
    r = ask(tmp_path, f"Employees get 25 vacation days {cite}.")
    assert r["answer"] == "Employees get 25 vacation days [handbook]."
    assert r["flags"] == []


def test_several_ids_in_one_bracket():
    out, removed, valid = _check_citations("See [doc_id: handbook, contacts; ghost_doc].",
                                           {"handbook", "contacts"})
    assert out == "See [handbook] [contacts]."
    assert removed and valid == 2


def test_out_of_context_id_is_stripped(tmp_path):
    r = ask(tmp_path, "Employees get 25 vacation days [handbook] [doc_id: hr_salary_bands].")
    assert "hr_salary_bands" not in r["answer"] and "[handbook]" in r["answer"]
    assert r["flags"] == ["ungrounded_citation_removed"]


def test_only_unknown_citations_leave_nothing_traceable(tmp_path):
    flags = ask(tmp_path, "Employees get 25 vacation days [ghost_doc].")["flags"]
    assert flags == ["ungrounded_citation_removed", "no_citation"]


def test_answer_without_citation_is_flagged_not_blocked(tmp_path):
    r = ask(tmp_path, "Employees get 25 vacation days.")
    assert r["answer"] == "Employees get 25 vacation days."
    assert r["flags"] == ["no_citation"]


def test_ordinary_bracketed_text_is_not_a_citation():
    out, removed, valid = _check_citations("Days are counted [see policy above] [handbook].", {"handbook"})
    assert out == "Days are counted [see policy above] [handbook]." and not removed and valid == 1


def test_model_refusal_is_not_flagged(tmp_path):
    assert ask(tmp_path, "I do not have enough information to answer that.")["flags"] == []


def test_mock_llm_answers_still_pass(tmp_path):
    c = TestClient(create_app(audit_path=tmp_path / "a.jsonl"))
    for key, q in [("key-bob", "How many vacation days do employees get?"),
                   ("key-alice", "What are the salary bands for senior engineer?"),
                   ("key-carol", "What was Q3 revenue and operating margin?")]:
        flags = c.post("/ask", json={"question": q}, headers={"X-API-Key": key}).json()["flags"]
        assert "no_citation" not in flags and "ungrounded_citation_removed" not in flags
