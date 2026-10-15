"""Per-request model selection: registry from env, /providers, provider field on /ask."""
import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.llm import ClaudeLLM, MockLLM, OllamaLLM, build_providers
from app.main import create_app


class RecordingLLM:
    def __init__(self, name, reply=None):
        self.name, self.reply, self.contexts = name, reply, []

    def generate(self, question, contexts):
        self.contexts.append([c.doc_id for c in contexts])
        return self.reply or f"Employees get 25 vacation days [handbook]. ({self.name})"


class BrokenLLM:
    name = "broken"

    def generate(self, question, contexts):
        raise TimeoutError("down")


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "GEMINI_API_KEY",
                "OLLAMA_URL", "OLLAMA_MODEL", "OLLAMA_TIMEOUT", "OLLAMA_NUM_PREDICT",
                "OLLAMA_REMOTE_URL", "OLLAMA_REMOTE_MODEL", "OLLAMA_REMOTE_TIMEOUT", "OLLAMA_REMOTE_NUM_PREDICT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", lambda **kw: object())  # never a real client
    return monkeypatch


def ask(client, key, q, provider=None):
    body = {"question": q} if provider is None else {"question": q, "provider": provider}
    return client.post("/ask", json=body, headers={"X-API-Key": key})


# --- registry from env ---------------------------------------------------------

def test_only_mock_by_default(clean_env):
    providers, default = build_providers()
    assert list(providers) == ["mock"] and default == "mock"


def test_all_configured_providers_listed(clean_env):
    clean_env.setenv("LLM_PROVIDER", "claude")
    clean_env.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    clean_env.setenv("OLLAMA_URL", "http://localhost:11434")
    clean_env.setenv("OLLAMA_REMOTE_URL", "http://localhost:11435")
    clean_env.setenv("OLLAMA_REMOTE_MODEL", "llama3.2:3b")
    clean_env.setenv("OLLAMA_REMOTE_TIMEOUT", "90")
    providers, default = build_providers()
    assert set(providers) == {"mock", "claude", "ollama", "ollama-remote"}
    assert default == "claude"
    assert isinstance(providers["claude"], ClaudeLLM)
    remote = providers["ollama-remote"]
    assert isinstance(remote, OllamaLLM)
    assert (remote.url, remote.model, remote.timeout) == ("http://localhost:11435", "llama3.2:3b", 90.0)


def test_unconfigured_default_falls_back_to_mock(clean_env):
    clean_env.setenv("LLM_PROVIDER", "claude")  # no key -> not available
    providers, default = build_providers()
    assert "claude" not in providers and default == "mock"


# --- /providers and selection --------------------------------------------------

@pytest.fixture
def multi(tmp_path):
    a, b = RecordingLLM("model-a"), RecordingLLM("model-b")
    app = create_app(providers={"a": a, "b": b, "mock": MockLLM()}, audit_path=tmp_path / "a.jsonl")
    return TestClient(app), a, b, tmp_path


def test_providers_endpoint(multi):
    c, *_ = multi
    body = c.get("/providers").json()
    assert body["default"] == "a"
    assert body["providers"] == [{"id": "a", "model": "model-a"}, {"id": "b", "model": "model-b"},
                                 {"id": "mock", "model": "mock-extractive"}]


def test_default_used_without_provider(multi):
    c, a, b, _ = multi
    r = ask(c, "key-bob", "How many vacation days?").json()
    assert r["model"] == "model-a" and len(a.contexts) == 1 and not b.contexts


def test_selected_provider_is_used_and_audited(multi):
    c, a, b, tmp_path = multi
    r = ask(c, "key-bob", "How many vacation days?", provider="b").json()
    assert r["model"] == "model-b" and "(model-b)" in r["answer"]
    assert not a.contexts and len(b.contexts) == 1
    assert '"model": "model-b"' in (tmp_path / "a.jsonl").read_text()


def test_unknown_provider_rejected_without_llm_call(multi):
    c, a, b, tmp_path = multi
    r = ask(c, "key-bob", "How many vacation days?", provider="gpt-9")
    assert r.status_code == 400
    assert not a.contexts and not b.contexts
    assert "unknown_provider" in (tmp_path / "a.jsonl").read_text()


def test_acl_applies_whatever_provider_is_chosen(multi):
    c, a, b, _ = multi
    for pid in ("a", "b"):
        ask(c, "key-bob", "What are the salary bands for senior engineer?", provider=pid)
    for seen in a.contexts + b.contexts:
        assert "hr_salary_bands" not in seen


def test_selected_provider_failure_falls_back(tmp_path):
    c = TestClient(create_app(providers={"mock": MockLLM(), "broken": BrokenLLM()},
                              audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", "How many vacation days?", provider="broken").json()
    assert "llm_fallback:TimeoutError" in r["flags"]
    assert r["model"] == "mock-extractive" and "25 vacation days" in r["answer"]


def test_ui_has_model_selector(tmp_path):
    html = TestClient(create_app(audit_path=tmp_path / "a.jsonl")).get("/").text
    assert 'id="provider"' in html and "/providers" in html
