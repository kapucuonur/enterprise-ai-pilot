"""Capture the README screenshots into docs/images/ (mock mode, local port, no secrets).

Run from the repo root:
    pip install -r demo/requirements.txt
    python demo/take_screenshots.py

Writes:
    docs/images/bob-salary.png        bob (engineering) asks for salary bands
    docs/images/alice-salary.png      alice (HR) asks the same question
    docs/images/role-comparison.png   the two side by side
    docs/images/pii-redaction.png     bob asks for the IT helpdesk contact
"""
from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

from record_demo import ROOT, free_port, launch, start_app

OUT = ROOT / "docs" / "images"
MAX_WIDTH = 1280
SCALE = 1.75  # 720px page column -> 1260px image
SALARY_Q = "What are the salary bands for senior engineer?"
HELPDESK_Q = "Who is the IT helpdesk contact, email and phone?"


def save_png(png: bytes, path: Path) -> None:
    """Downscale to MAX_WIDTH if needed and store as an optimized palette PNG."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE) \
       .save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}  {img.width}x{img.height}  {path.stat().st_size // 1024} KB")


def ask(page: Page, user_key: str, question: str) -> None:
    page.select_option("#user", user_key)
    page.fill("#question", question)
    page.click("#ask")
    page.wait_for_selector("#ask:not([disabled])")
    page.wait_for_selector("#result:not([hidden])")
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.wait_for_timeout(250)


def comparison_html(left: bytes, right: bytes) -> str:
    def panel(png: bytes, who: str, role: str, verdict: str, tone: str) -> str:
        data = base64.b64encode(png).decode()
        return f"""
        <section>
          <div class="who"><b>{who}</b><span class="chip {tone}">{role}</span></div>
          <div class="verdict {tone}">{verdict}</div>
          <img src="data:image/png;base64,{data}" alt="">
        </section>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ margin: 0; background: #eef2f7; color: #172033;
              font: 15px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }}
      .wrap {{ padding: 28px 28px 24px; width: 1280px; box-sizing: border-box; }}
      h1 {{ font-size: 22px; margin: 0 0 4px; }}
      p.q {{ margin: 0 0 18px; color: #556275; }}
      p.q code {{ background: #fff; border: 1px solid #d9e0ea; border-radius: 6px; padding: 2px 6px; color: #172033; }}
      .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
      .who {{ display: flex; gap: 8px; align-items: center; font-size: 16px; margin-bottom: 6px; }}
      .chip {{ font-size: 12px; font-weight: 600; padding: 1px 9px; border-radius: 999px; border: 1px solid; }}
      .verdict {{ font-size: 13px; font-weight: 600; margin-bottom: 10px; }}
      .blue {{ background: #e6eefc; color: #1e3a8a; border-color: #bcd0f5; }}
      .pink {{ background: #fce7f1; color: #8a1a4b; border-color: #f2b8d2; }}
      .verdict.blue, .verdict.pink {{ background: none; border: 0; }}
      img {{ width: 100%; display: block; border-radius: 12px; box-shadow: 0 1px 2px rgba(23,32,51,.06), 0 4px 16px rgba(23,32,51,.08); }}
    </style></head><body><div class="wrap">
      <h1>Same question, different role, different answer</h1>
      <p class="q">Both users ask <code>{SALARY_Q}</code></p>
      <div class="grid">
        {panel(left, "bob", "engineering", "No answer. HR documents are filtered out before retrieval and the model is never called", "blue")}
        {panel(right, "alice", "hr", "Answer from the salary bands document, with its source", "pink")}
      </div>
    </div></body></html>"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        app = start_app(port, Path(tmp) / "audit.jsonl")
        try:
            with sync_playwright() as p:
                browser = launch(p)
                ctx = browser.new_context(viewport={"width": 760, "height": 900},
                                          device_scale_factor=SCALE, color_scheme="light")
                page = ctx.new_page()
                page.goto(f"http://127.0.0.1:{port}/")
                page.wait_for_selector("#provider option[value='mock']", state="attached")

                ask(page, "key-bob", SALARY_Q)
                bob_full, bob_card = page.locator("main").screenshot(), page.locator("#result").screenshot()
                save_png(bob_full, OUT / "bob-salary.png")

                ask(page, "key-alice", SALARY_Q)
                alice_full, alice_card = page.locator("main").screenshot(), page.locator("#result").screenshot()
                save_png(alice_full, OUT / "alice-salary.png")

                ask(page, "key-bob", HELPDESK_Q)
                save_png(page.locator("main").screenshot(), OUT / "pii-redaction.png")

                cmp_ctx = browser.new_context(viewport={"width": 1280, "height": 600},
                                              device_scale_factor=1, color_scheme="light")
                cmp = cmp_ctx.new_page()
                cmp.set_content(comparison_html(bob_card, alice_card))
                cmp.wait_for_timeout(200)
                save_png(cmp.locator(".wrap").screenshot(), OUT / "role-comparison.png")
                browser.close()
        finally:
            app.terminate()
            app.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
