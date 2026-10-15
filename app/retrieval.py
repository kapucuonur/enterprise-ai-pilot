"""Access-controlled retrieval (BM25 over paragraph chunks).

Key design rule: the ACL filter runs BEFORE ranking. Chunks a user may not see
never enter scoring, the prompt, the logs of "sources", or the LLM context.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for", "on", "at",
    "how", "what", "who", "when", "do", "does", "we", "our", "can", "i", "my", "be",
    "it", "this", "that", "with", "by", "as", "per",
}


# Relevance gate (applied after the ACL filter, on chunks the caller may see).
# A chunk is kept only if it actually overlaps the question:
#   - it matches at least MIN_MATCHED_TERMS distinct query terms, OR at least
#     MIN_MATCH_RATIO of them (so one- and two-word questions still work), and
#   - its BM25 score is at least MIN_SCORE, and
#   - its matches are not only GENERIC_TERMS (words like "engineer" appear everywhere,
#     so on their own they say nothing about relevance).
# If nothing passes, search() returns [] and the pipeline refuses without calling the LLM.
MIN_MATCHED_TERMS = 2
MIN_MATCH_RATIO = 0.5
MIN_SCORE = 1.5
GENERIC_TERMS = frozenset({"engineer", "senior", "employee", "staff", "team", "company"})


def tokenize(text: str) -> list[str]:
    out = []
    for w in _TOKEN.findall(text.lower()):
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s"):  # crude plural folding
            w = w[:-1]
        out.append(w)
    return out


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    access: frozenset[str]
    text: str


def load_corpus(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = raw
        if raw.startswith("---"):
            _, header, body = raw.split("---", 2)
            for line in header.strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        access = frozenset(a.strip() for a in meta.get("access", "").split(",") if a.strip())
        if not access:
            raise ValueError(f"{path.name}: missing 'access' (deny by default)")
        title = meta.get("title", path.stem)
        for para in re.split(r"\n\s*\n", body.strip()):
            para = para.strip()
            if len(para) >= 20 and not para.startswith("#"):
                chunks.append(Chunk(path.stem, title, access, para))
    return chunks


def _relevant(matched: set[str], n_terms: int, score: float) -> bool:
    if not matched or score < MIN_SCORE or matched <= GENERIC_TERMS:
        return False
    return len(matched) >= MIN_MATCHED_TERMS or len(matched) / n_terms >= MIN_MATCH_RATIO


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, query: str, user_roles: set[str], k: int = 3):
        # 1) ACL first: only chunks the caller may read are even considered.
        visible = [c for c in self.chunks if c.access & user_roles]
        if not visible:
            return []
        docs = [Counter(tokenize(c.text + " " + c.title)) for c in visible]
        n = len(visible)
        avg_len = sum(sum(d.values()) for d in docs) / n or 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(d.keys())
        q_terms = tokenize(query)
        distinct = set(q_terms)
        k1, b = 1.5, 0.75
        scored = []
        for chunk, d in zip(visible, docs):
            length = sum(d.values())
            score = 0.0
            for t in q_terms:
                if t not in d:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                f = d[t]
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_len))
            # 2) Relevance gate, see the rule at the top of this module.
            if _relevant(distinct & d.keys(), len(distinct), score):
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
