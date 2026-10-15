"""Enterprise AI pilot: governed Q&A over internal documents.

Request pipeline (every step is auditable):
  authn -> rate limit -> input injection check -> ACL-filtered retrieval ->
  drop poisoned chunks -> (no context? refuse, skip LLM) -> LLM with fallback ->
  citation validation -> output PII redaction -> audit record
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .audit import AuditLog
from .llm import LLM, MockLLM, build_providers
from .retrieval import Index, load_corpus
from .security import RateLimiter, User, authenticate, looks_like_injection, redact

ROOT = Path(__file__).resolve().parent.parent
NO_ANSWER = "I do not have enough information to answer that from the documents available to you."
_CITATION = re.compile(r"\[([a-z0-9_\-]+)\]")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    provider: str | None = Field(default=None, max_length=32)  # id from GET /providers


def create_app(
    llm: LLM | None = None,
    providers: dict[str, LLM] | None = None,
    docs_dir: Path | None = None,
    audit_path: Path | None = None,
    rate_limit: int = 30,
) -> FastAPI:
    app = FastAPI(title="Enterprise AI Pilot", version="0.1.0")
    index = Index(load_corpus(docs_dir or ROOT / "data" / "docs"))
    audit = AuditLog(audit_path or Path(os.getenv("AUDIT_PATH", ROOT / "audit.jsonl")))
    limiter = RateLimiter(limit=rate_limit)
    if providers is not None:  # explicit registry (tests); first entry is the default
        registry, default_id = dict(providers), next(iter(providers))
    elif llm is not None:
        registry, default_id = {"default": llm, "mock": MockLLM()}, "default"
    else:
        registry, default_id = build_providers()
    primary = registry[default_id]
    fallback = MockLLM()
    app.state.audit = audit

    def current_user(x_api_key: str | None = Header(default=None)) -> User:
        user = authenticate(x_api_key)
        if user is None:
            audit.write(event="auth_failed")
            raise HTTPException(401, "Invalid or missing API key")
        return user

    @app.get("/", include_in_schema=False)
    def ui():  # static demo page; all data still goes through /ask with its API key
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "llm": primary.name}

    @app.get("/providers")
    def list_providers():  # only what the operator enabled via env; no secrets
        return {"default": default_id,
                "providers": [{"id": pid, "model": p.name} for pid, p in registry.items()]}

    @app.post("/ask")
    def ask(req: AskRequest, user: User = Depends(current_user)):
        rid, started = uuid.uuid4().hex[:12], time.perf_counter()
        base = {"request_id": rid, "user": user.name}
        flags: list[str] = []

        if not limiter.allow(user.name):
            audit.write(event="ask", decision="rate_limited", **base)
            raise HTTPException(429, "Rate limit exceeded")

        provider_id = req.provider or default_id
        if provider_id not in registry:
            audit.write(event="ask", decision="unknown_provider", provider=provider_id[:32], **base)
            raise HTTPException(400, "Unknown provider")
        chosen = registry[provider_id]

        safe_q, _ = redact(req.question)  # never persist raw PII from the question
        if looks_like_injection(req.question):
            audit.write(event="ask", decision="blocked_injection_query", question=safe_q, **base)
            raise HTTPException(400, "Request blocked by safety policy")

        hits = index.search(req.question, set(user.roles))
        clean = []
        for chunk, score in hits:
            if looks_like_injection(chunk.text):
                flags.append(f"injection_chunk_dropped:{chunk.doc_id}")
                continue
            clean.append((chunk, score))

        sources = [{"doc_id": c.doc_id, "title": c.title, "score": round(s, 2)} for c, s in clean]
        contexts = [c for c, _ in clean]

        if not contexts:
            audit.write(event="ask", decision="no_context", question=safe_q, flags=flags,
                        latency_ms=_ms(started), **base)
            return {"request_id": rid, "answer": NO_ANSWER, "sources": [], "flags": flags, "model": None}

        model_used = chosen.name
        try:
            answer = chosen.generate(req.question, contexts)
        except Exception as exc:  # provider down/timeout: degrade, don't fail
            flags.append(f"llm_fallback:{type(exc).__name__}")
            model_used = fallback.name
            answer = fallback.generate(req.question, contexts)

        allowed_ids = {c.doc_id for c in contexts}
        cited = set(_CITATION.findall(answer))
        invalid = cited - allowed_ids
        if invalid:  # citation that does not match any retrieved doc = ungrounded
            flags.append("ungrounded_citation_removed")
            answer = _CITATION.sub(lambda m: m.group(0) if m.group(1) in allowed_ids else "", answer)

        answer, pii = redact(answer)
        if pii:
            flags.append("pii_redacted")

        audit.write(event="ask", decision="answered", question=safe_q, model=model_used,
                    doc_ids=sorted(allowed_ids), flags=flags, pii=pii,
                    latency_ms=_ms(started), **base)
        return {"request_id": rid, "answer": answer, "sources": sources, "flags": flags, "model": model_used}

    @app.get("/audit/verify")
    def verify_audit(user: User = Depends(current_user)):
        if "admin" not in user.roles:
            raise HTTPException(403, "Admin only")
        ok, n = audit.verify()
        return {"intact": ok, "records": n}

    return app


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


app = create_app()
