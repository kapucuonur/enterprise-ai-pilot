"""Tamper-evident audit log: JSONL where each record hashes the previous one."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = json.loads(line)["hash"]
        return last

    @staticmethod
    def _digest(prev: str, body: dict) -> str:
        payload = prev + json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def write(self, **event) -> dict:
        with self._lock:
            body = {"ts": round(time.time(), 3), **event}
            digest = self._digest(self._last, body)
            record = {**body, "prev": self._last, "hash": digest}
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            self._last = digest
            return record

    def verify(self) -> tuple[bool, int]:
        """Return (ok, records_checked)."""
        prev, count = GENESIS, 0
        if not self.path.exists():
            return True, 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
            if rec["prev"] != prev or self._digest(prev, body) != rec["hash"]:
                return False, count
            prev, count = rec["hash"], count + 1
        return True, count
