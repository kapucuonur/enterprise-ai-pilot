"""OllamaLLM provider with httpx mocked (no real network)."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.llm import SYSTEM_PROMPT, MockLLM, OllamaLLM, build_llm
from app.main import create_app
from app.retrieval import Chunk

CTX = [Chunk("handbook", "Handbook", frozenset({"employee"}), "Employees get 25 vacation days per year.")]


class FakePost:
    """Replaces httpx.post; records calls and returns a canned /api/chat reply or raises."""

    def __init__(self, content="Employees get 25 vacation days [handbook].", exc=None, status=200):
        self.content, self.exc, self.status, self.calls = content, exc, status, []

    def __call__(self, url, json=None, timeout=None, **kw):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exc:
            raise self.exc
        req = httpx.Request("POST", url)
        return httpx.Response(self.status, request=req,
                              json={"model": json["model"], "message": {"role": "assistant", "content": self.content},
                                    "done": True})


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "OLLAMA_URL", "OLLAMA_MODEL", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def patch_post(monkeypatch, fake):
    monkeypatch.setattr(llm_mod.httpx, "post", fake)
    return fake


def ask(client, key, q):
    return client.post("/ask", json={"question": q}, headers={"X-API-Key": key})


def test_mock_still_default(clean_env):
    assert isinstance(build_llm(), MockLLM)


def test_ollama_selected_with_defaults(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    llm = build_llm()
    assert isinstance(llm, OllamaLLM)
    assert llm.url == "http://localhost:11434" and llm.model == "llama3.2" and llm.name == "ollama:llama3.2"


def test_ollama_env_overrides(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("OLLAMA_URL", "http://gpu-box:11434/")
    clean_env.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    llm = build_llm()
    assert llm.url == "http://gpu-box:11434" and llm.model == "qwen2.5:7b"


def test_timeout_and_token_cap_from_env(clean_env, monkeypatch):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("OLLAMA_TIMEOUT", "90")
    clean_env.setenv("OLLAMA_NUM_PREDICT", "200")
    llm = build_llm()
    assert llm.timeout == 90.0 and llm.num_predict == 200
    fake = patch_post(monkeypatch, FakePost())
    llm.generate("vacation?", CTX)
    assert fake.calls[0]["timeout"] == 90.0
    assert fake.calls[0]["json"]["options"] == {"temperature": 0.1, "num_predict": 200}


def test_request_shape(monkeypatch):
    fake = patch_post(monkeypatch, FakePost())
    out = OllamaLLM().generate("How many vacation days?", CTX)
    assert out == "Employees get 25 vacation days [handbook]."
    call = fake.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == 30.0
    body = call["json"]
    assert body["model"] == "llama3.2" and body["stream"] is False
    assert body["options"] == {"temperature": 0.1}
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = body["messages"][1]
    assert user["role"] == "user"
    assert user["content"] == ("Context:\n[handbook] Employees get 25 vacation days per year."
                               "\n\nQuestion: How many vacation days?")


def test_empty_reply_raises(monkeypatch):
    patch_post(monkeypatch, FakePost(content="   "))
    with pytest.raises(RuntimeError):
        OllamaLLM().generate("vacation?", CTX)


def test_http_error_raises(monkeypatch):
    patch_post(monkeypatch, FakePost(status=404))  # e.g. model not pulled
    with pytest.raises(httpx.HTTPStatusError):
        OllamaLLM().generate("vacation?", CTX)


def test_ollama_answer_flows_through_pipeline(monkeypatch, tmp_path):
    patch_post(monkeypatch, FakePost())
    c = TestClient(create_app(llm=OllamaLLM(), audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", "How many vacation days do employees get?").json()
    assert "25 vacation days" in r["answer"] and not r["flags"]


def test_ollama_down_falls_back(monkeypatch, tmp_path):
    patch_post(monkeypatch, FakePost(exc=httpx.ConnectError("connection refused")))
    c = TestClient(create_app(llm=OllamaLLM(), audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", "How many vacation days?")
    assert r.status_code == 200
    assert "llm_fallback:ConnectError" in r.json()["flags"]
    assert "25 vacation days" in r.json()["answer"]


def test_acl_still_applies_before_ollama(monkeypatch, tmp_path):
    fake = patch_post(monkeypatch, FakePost())
    c = TestClient(create_app(llm=OllamaLLM(), audit_path=tmp_path / "a.jsonl"))
    ask(c, "key-bob", "What are the salary bands for senior engineer?")
    for call in fake.calls:
        assert "hr_salary_bands" not in call["json"]["messages"][1]["content"]
