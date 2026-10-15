"""Record a short walkthrough of the web UI as demo/demo.webm (and demo/demo.mp4 if ffmpeg exists).

Run from the repo root:
    pip install -r demo/requirements.txt
    python -m playwright install chromium ffmpeg   # chromium optional if Google Chrome is installed
    python demo/record_demo.py

Safety: the app is started in mock mode with a scrubbed environment (no API keys,
no .env, no Ollama hosts), bound to 127.0.0.1 on a free port, with a throwaway audit log.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SIZE = {"width": 1280, "height": 720}
PAUSE_MS = 2500
SALARY_Q = "What are the salary bands for senior engineer?"
HELPDESK_Q = "Who is the IT helpdesk contact, email and phone?"

CAPTION_JS = """
(text) => {
  let el = document.getElementById("demo-caption");
  if (!el) {
    el = document.createElement("div");
    el.id = "demo-caption";
    el.setAttribute("aria-hidden", "true");
    Object.assign(el.style, {
      position: "fixed", left: "50%", bottom: "24px", transform: "translateX(-50%)",
      maxWidth: "min(1040px, calc(100% - 48px))", padding: "14px 22px", borderRadius: "12px",
      background: "rgba(15, 23, 42, .92)", color: "#fff", zIndex: "2147483647",
      font: "600 20px/1.35 system-ui, -apple-system, 'Segoe UI', sans-serif",
      boxShadow: "0 8px 30px rgba(0,0,0,.35)", textAlign: "center", transition: "opacity .25s",
    });
    document.body.appendChild(el);
    document.body.style.paddingBottom = "220px";  // room so content can scroll clear of the caption
  }
  el.style.opacity = text ? "1" : "0";
  if (text) el.textContent = text;
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def scrubbed_env(audit_path: Path) -> dict[str, str]:
    """Only what uvicorn needs; nothing that could enable a real provider or leak a key."""
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "SYSTEMROOT", "TMPDIR") if k in os.environ}
    return {**keep, "LLM_PROVIDER": "mock", "AUDIT_PATH": str(audit_path), "PYTHONUNBUFFERED": "1"}


def start_app(port: int, audit_path: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=scrubbed_env(audit_path), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("app exited early:\n" + proc.stderr.read().decode())
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if b"mock-extractive" not in r.read():
                    raise RuntimeError("app is not in mock mode; refusing to record")
                return proc
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("app did not start within 20 s")


def launch(p) -> Browser:
    try:
        return p.chromium.launch()  # bundled Chromium (python -m playwright install chromium)
    except Exception:
        return p.chromium.launch(channel="chrome")  # fall back to an installed Google Chrome


def caption(page: Page, text: str = "") -> None:
    page.evaluate(CAPTION_JS, text)


def show(page: Page, selector: str) -> None:
    """Scroll so the element sits in the middle of the viewport, above the caption."""
    page.eval_on_selector(selector, "el => el.scrollIntoView({block: 'center', behavior: 'smooth'})")
    page.wait_for_timeout(600)


def ask(page: Page, user_key: str, question: str) -> None:
    page.select_option("#user", user_key)
    page.wait_for_timeout(500)
    page.fill("#question", "")
    page.type("#question", question, delay=35)
    page.wait_for_timeout(300)
    page.click("#ask")
    page.wait_for_selector("#ask:not([disabled])")
    page.wait_for_selector("#result:not([hidden])")
    show(page, "#sources")
    page.wait_for_timeout(PAUSE_MS)


def walkthrough(page: Page, base: str) -> None:
    page.goto(base)
    page.wait_for_selector("#provider option[value='mock']", state="attached")
    page.evaluate("document.documentElement.style.zoom = '1.25'")  # readable at 1280x720
    caption(page, "Enterprise AI Pilot · governed Q&A over internal documents (offline mock model)")
    page.wait_for_timeout(PAUSE_MS)

    caption(page, "bob (engineering) asks about salary bands")
    ask(page, "key-bob", SALARY_Q)
    caption(page, "HR documents are filtered out before retrieval: no salary data reaches bob")
    page.wait_for_timeout(PAUSE_MS)

    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    caption(page, "Switch to alice (HR) and ask the same question")
    ask(page, "key-alice", SALARY_Q)
    caption(page, "Same question, different role, different answer, with its source")
    page.wait_for_timeout(PAUSE_MS)

    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    caption(page, "bob asks for the IT helpdesk email and phone")
    ask(page, "key-bob", HELPDESK_Q)
    caption(page, "Email and phone are redacted before the answer leaves the server")
    page.wait_for_timeout(PAUSE_MS)

    show(page, "#flags")
    caption(page, "Every answer shows its sources, scores and safety flags, and is written to a hash-chained audit log")
    page.wait_for_timeout(PAUSE_MS + 1000)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    video_dir = OUT / "_video"
    shutil.rmtree(video_dir, ignore_errors=True)
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        app = start_app(port, Path(tmp) / "audit.jsonl")
        try:
            with sync_playwright() as p:
                browser = launch(p)
                ctx = browser.new_context(viewport=SIZE, record_video_dir=str(video_dir),
                                          record_video_size=SIZE, color_scheme="light")
                page = ctx.new_page()
                walkthrough(page, f"http://127.0.0.1:{port}/")
                video = page.video
                ctx.close()  # finalises the video file
                browser.close()
                webm = OUT / "demo.webm"
                Path(video.path()).replace(webm)
        finally:
            app.terminate()
            app.wait(timeout=10)
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"wrote {webm.relative_to(ROOT)}")

    if shutil.which("ffmpeg"):
        mp4 = OUT / "demo.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23", str(mp4)], check=True)
        print(f"wrote {mp4.relative_to(ROOT)}")
    else:
        print("ffmpeg not found; skipped mp4 conversion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
