"""ClaudeLLM provider (SDK mocked, no network) and the "/" web UI route."""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.llm import SYSTEM_PROMPT, ClaudeLLM, MockLLM, build_llm
from app.main import create_app
from app.retrieval import Chunk

CTX = [Chunk("handbook", "Handbook", frozenset({"employee"}), "Employees get 25 vacation days per year.")]


def _resp(text="Employees get 25 vacation days [handbook].", stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, result):
        self.result, self.calls = result, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeAnthropic:
    """Stands in for anthropic.Anthropic; records constructor args."""
    instances: list["FakeAnthropic"] = []
    result = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = FakeMessages(FakeAnthropic.result or _resp())
        FakeAnthropic.instances.append(self)


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeAnthropic.instances, FakeAnthropic.result = [], None
    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", FakeAnthropic)
    return FakeAnthropic


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def ask(client, key, q):
    return client.post("/ask", json={"question": q}, headers={"X-API-Key": key})


# --- provider selection -------------------------------------------------------

def test_mock_is_default(clean_env):
    assert isinstance(build_llm(), MockLLM)


def test_claude_without_key_stays_mock(clean_env):
    clean_env.setenv("LLM_PROVIDER", "claude")
    assert isinstance(build_llm(), MockLLM)


def test_claude_selected_from_env(clean_env, fake_sdk):
    clean_env.setenv("LLM_PROVIDER", "claude")
    clean_env.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    clean_env.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    llm = build_llm()
    assert isinstance(llm, ClaudeLLM) and llm.name == "claude:claude-sonnet-5-5"
    kw = fake_sdk.instances[0].kwargs
    assert kw["api_key"] == "test-key-not-real" and kw["timeout"] == 15.0


def test_claude_model_defaults_when_unset(clean_env, fake_sdk):
    clean_env.setenv("LLM_PROVIDER", "claude")
    clean_env.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert build_llm().model == llm_mod.CLAUDE_DEFAULT_MODEL


# --- request shape ------------------------------------------------------------

def test_request_uses_system_prompt_context_and_limits(fake_sdk):
    llm = ClaudeLLM("k", model="claude-haiku-4-5")
    assert llm.generate("How many vacation days?", CTX) == "Employees get 25 vacation days [handbook]."
    call = fake_sdk.instances[0].messages.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["max_tokens"] == 400
    assert call["temperature"] == 0.1
    assert call["model"] == "claude-haiku-4-5"
    user_msg = call["messages"][0]["content"]
    assert "[handbook] Employees get 25 vacation days" in user_msg and "How many vacation days?" in user_msg


def test_newer_models_skip_temperature(fake_sdk):
    ClaudeLLM("k", model="claude-opus-5-5").generate("vacation?", CTX)
    call = fake_sdk.instances[0].messages.calls[0]
    assert "temperature" not in call  # would be a 400 on this model
    assert call["output_config"] == {"effort": "low"}


@pytest.mark.parametrize("resp", [_resp(text=None, stop_reason="max_tokens"),
                                  _resp(text="", stop_reason="end_turn"),
                                  _resp(text="no", stop_reason="refusal")])
def test_unusable_responses_raise(fake_sdk, resp):
    fake_sdk.result = resp
    with pytest.raises(RuntimeError):
        ClaudeLLM("k").generate("vacation?", CTX)


# --- pipeline integration (fallback contract unchanged) -------------------------

def test_claude_answer_flows_through_pipeline(fake_sdk, tmp_path):
    c = TestClient(create_app(llm=ClaudeLLM("k"), audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", "How many vacation days do employees get?").json()
    assert "25 vacation days" in r["answer"] and not r["flags"]


def test_claude_sdk_error_falls_back(fake_sdk, tmp_path):
    fake_sdk.result = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    c = TestClient(create_app(llm=ClaudeLLM("k"), audit_path=tmp_path / "a.jsonl"))
    r = ask(c, "key-bob", "How many vacation days?")
    assert r.status_code == 200
    assert "llm_fallback:APITimeoutError" in r.json()["flags"]
    assert "25 vacation days" in r.json()["answer"]


def test_acl_still_applies_before_claude(fake_sdk, tmp_path):
    llm = ClaudeLLM("k")
    c = TestClient(create_app(llm=llm, audit_path=tmp_path / "a.jsonl"))
    ask(c, "key-bob", "What are the salary bands for senior engineer?")
    for call in fake_sdk.instances[0].messages.calls:
        assert "hr_salary_bands" not in call["messages"][0]["content"]


# --- web UI -------------------------------------------------------------------

def test_ui_is_served_without_api_key(tmp_path):
    c = TestClient(create_app(audit_path=tmp_path / "a.jsonl"))
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    for key in ("key-alice", "key-bob", "key-carol", "key-dave"):
        assert key in html
    assert 'id="question"' in html and "Verify audit log" in html and "/audit/verify" in html


def test_ui_does_not_bypass_auth(tmp_path):
    c = TestClient(create_app(audit_path=tmp_path / "a.jsonl"))
    c.get("/")
    assert c.post("/ask", json={"question": "vacation days?"}).status_code == 401
    assert c.get("/audit/verify").status_code == 401
