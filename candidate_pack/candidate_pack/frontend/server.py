"""Local-only web interface for the two candidate-pack tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = Path(__file__).resolve().parent
PLACEMENT_DIR = ROOT / "placement"
MESSAGES_DIR = ROOT / "class_messages"
PREDICT_SCRIPT = PLACEMENT_DIR / "predict.py"
EXTRACT_SCRIPT = MESSAGES_DIR / "extract.py"
DEFAULT_MODEL = "qwen2.5:3b-instruct-q4_K_M"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_MESSAGE_FILES = 50
MAX_MESSAGE_BYTES = 512 * 1024
MAX_TOTAL_MESSAGE_BYTES = 8 * 1024 * 1024

sys.path.insert(0, str(PLACEMENT_DIR))
import predict as placement_predict  # noqa: E402


class RequestError(ValueError):
    """A well-formed request is missing valid task input."""


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise RequestError("Invalid Content-Length header.") from exc
    if length <= 0:
        raise RequestError("Request body is empty.")
    if length > MAX_BODY_BYTES:
        raise RequestError("Request is too large (12 MB maximum).")
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError("Request body must be valid UTF-8 JSON.") from exc
    if not isinstance(data, dict):
        raise RequestError("Request body must be a JSON object.")
    return data


def _placement(data: dict[str, Any]) -> dict[str, Any]:
    csv_text = data.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise RequestError("Choose a non-empty student CSV file first.")
    if len(csv_text.encode("utf-8")) > MAX_BODY_BYTES:
        raise RequestError("Student CSV exceeds the 12 MB upload limit.")
    with tempfile.TemporaryDirectory(prefix="goml-placement-") as tmp:
        input_path = Path(tmp) / "students.csv"
        output_path = Path(tmp) / "predictions.csv"
        input_path.write_text(csv_text, encoding="utf-8-sig", newline="")
        placement_predict.predict(input_path, output_path)
        import pandas as pd

        result = pd.read_csv(output_path, dtype={"student_id": "string"})
    records = [
        {"student_id": str(row.student_id), "placed": int(row.placed)}
        for row in result.itertuples(index=False)
    ]
    return {
        "predictions": records,
        "count": len(records),
        "placed_count": sum(row["placed"] == 1 for row in records),
        "not_placed_count": sum(row["placed"] == 0 for row in records),
    }


def _extract(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("messages")
    if not isinstance(items, list) or not items:
        raise RequestError("Choose one or more message .txt files first.")
    if len(items) > MAX_MESSAGE_FILES:
        raise RequestError(f"Choose no more than {MAX_MESSAGE_FILES} message files per run.")
    total_bytes = 0
    normalized: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise RequestError(f"Message {index} must contain text content.")
        content = item["content"]
        size = len(content.encode("utf-8"))
        if size > MAX_MESSAGE_BYTES:
            raise RequestError(f"Message {index} exceeds the 512 KB per-message limit.")
        total_bytes += size
        normalized.append(content)
    if total_bytes > MAX_TOTAL_MESSAGE_BYTES:
        raise RequestError("Combined message content exceeds the 8 MB limit.")

    with tempfile.TemporaryDirectory(prefix="goml-messages-") as tmp:
        tmp_path = Path(tmp)
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        for index, content in enumerate(normalized, start=1):
            # Client-provided filenames are intentionally ignored.
            (inbox / f"message-{index:03d}.txt").write_text(content, encoding="utf-8", newline="")
        output_path = tmp_path / "tasks.json"
        timeout = min(max(180 * len(normalized), 180), 1800)
        run = subprocess.run(
            [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--inbox",
                str(inbox),
                "--output",
                str(output_path),
                "--model",
                DEFAULT_MODEL,
            ],
            cwd=MESSAGES_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        message_usage: list[dict[str, Any]] = []
        usage_summary: dict[str, Any] = {}
        diagnostic_lines: list[str] = []
        for line in run.stderr.splitlines():
            if line.startswith("processed="):
                continue
            if line.startswith("message_usage="):
                try:
                    item = json.loads(line.removeprefix("message_usage="))
                    if isinstance(item, dict):
                        message_usage.append(item)
                except json.JSONDecodeError:
                    diagnostic_lines.append(line)
            elif line.startswith("usage_summary="):
                try:
                    item = json.loads(line.removeprefix("usage_summary="))
                    if isinstance(item, dict):
                        usage_summary = item
                except json.JSONDecodeError:
                    diagnostic_lines.append(line)
            else:
                diagnostic_lines.append(line)
        if not output_path.is_file():
            details = "\n".join(diagnostic_lines).strip() or "The extractor exited without writing a task file."
            raise RuntimeError(details[-3000:])
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Extractor wrote invalid JSON: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("tasks"), list):
        raise RuntimeError("Extractor output did not match the tasks array schema.")
    return {
        "tasks": result["tasks"],
        "partial": run.returncode != 0,
        "diagnostics": "\n".join(diagnostic_lines).strip()[-3000:],
        "message_usage": message_usage,
        "usage_summary": usage_summary,
        "exit_code": run.returncode,
    }


def _health() -> dict[str, Any]:
    request = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        available = any(isinstance(item, dict) and item.get("name") == DEFAULT_MODEL for item in models)
        return {"ollama_running": True, "model_available": available, "model": DEFAULT_MODEL}
    except (OSError, urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError):
        return {"ollama_running": False, "model_available": False, "model": DEFAULT_MODEL}


class Handler(BaseHTTPRequestHandler):
    server_version = "GoMLLocal/1.0"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", file=sys.stderr)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/api/health":
            self._json(200, _health())
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/placement": ("placement.html", "text/html; charset=utf-8"),
            "/placement.html": ("placement.html", "text/html; charset=utf-8"),
            "/messages": ("messages.html", "text/html; charset=utf-8"),
            "/messages.html": ("messages.html", "text/html; charset=utf-8"),
            "/architecture": ("architecture.html", "text/html; charset=utf-8"),
            "/architecture.html": ("architecture.html", "text/html; charset=utf-8"),
            "/ARCHITECTURE.md": ("ARCHITECTURE.md", "text/markdown; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = assets.get(route)
        if asset is None:
            self._json(404, {"error": "Not found."})
            return
        file_path = FRONTEND / asset[0]
        try:
            body = file_path.read_bytes()
        except OSError:
            self._json(500, {"error": "Frontend asset is unavailable."})
            return
        self._send(200, body, asset[1])

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        try:
            data = _read_json(self)
            if route == "/api/placement":
                self._json(200, _placement(data))
            elif route == "/api/messages":
                self._json(200, _extract(data))
            else:
                self._json(404, {"error": "Not found."})
        except RequestError as exc:
            self._json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._json(503, {"error": f"A required local model artifact is missing: {exc}"})
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "Local model processing timed out. Try a smaller batch."})
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            self._json(422, {"error": str(exc) or "The request could not be processed."})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (local-only by default)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("For privacy, the UI only binds to a loopback address.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in the range 1-65535.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"GoML local UI: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GoML local UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
