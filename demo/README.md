# Demo video

`record_demo.py` records a ~30 s walkthrough of the web UI: role-based access (bob vs alice on
the same salary question), PII redaction, and the sources/flags panel.

```bash
pip install -r demo/requirements.txt
python -m playwright install chromium ffmpeg   # chromium is optional if Google Chrome is installed
python demo/record_demo.py              # -> demo/demo.webm, plus demo/demo.mp4 if ffmpeg is on PATH
python demo/take_screenshots.py         # -> docs/images/*.png used by the main README
```

The recording is not committed (see `.gitignore`); regenerate it with the command above.

What the script guarantees:

- The app runs in **mock mode only**, with a scrubbed environment: no `.env`, no API keys, no
  Ollama hosts. It refuses to record if `/health` reports anything other than the mock model.
- It binds to `127.0.0.1` on a free port and writes its audit log to a temporary directory.
- Only the page is recorded (no browser chrome, address bar, or desktop).
