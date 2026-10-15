"""Authn, PII redaction, prompt-injection heuristics, rate limiting."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Demo identities. In production: SSO/OIDC (Entra ID) + group-to-role mapping.
DEMO_USERS = {
    "key-alice": ("alice", {"employee", "hr"}),
    "key-bob": ("bob", {"employee", "engineering"}),
    "key-carol": ("carol", {"employee", "finance"}),
    "key-dave": ("dave", {"employee", "admin"}),
}


@dataclass(frozen=True)
class User:
    name: str
    roles: frozenset[str] = field(default_factory=frozenset)


def authenticate(api_key: str | None) -> User | None:
    if not api_key or api_key not in DEMO_USERS:
        return None
    name, roles = DEMO_USERS[api_key]
    return User(name, frozenset(roles))


_PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("HETU", re.compile(r"\b\d{6}[-+A]\d{3}[0-9A-Y]\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){3,6}(?:\s?\d{1,4})?\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d\s-]{7,14}\d")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in _PII_PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{label}]", text)
        if n:
            counts[label] = n
    return text, counts


_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions"
    r"|disregard\s+(the\s+)?(previous|system)"
    r"|reveal\s+.{0,40}system\s+prompt"
    r"|you\s+are\s+now\s+(?:dan|an?\s+unrestricted)"
    r"|developer\s+mode)",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


class RateLimiter:
    """Sliding-window limiter per user (in-memory; use Redis when scaled out)."""

    def __init__(self, limit: int = 30, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, who: str) -> bool:
        now = self.clock()
        q = self._hits[who]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
