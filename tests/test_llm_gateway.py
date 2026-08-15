"""Tests for the local LLM gateway (rate-limit absorbing proxy).

All tests use in-process fake upstream servers — no real network.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.llm_gateway import Gateway, _join_url, _key_for, _rewrite_model


def _start_server(handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _json(status: int, payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return body  # handlers below attach status


class RateLimitOnce(BaseHTTPRequestHandler):
    """Fails with 429 on the first call, succeeds afterwards."""

    calls = 0

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))  # drain to avoid RST
        type(self).calls += 1
        if type(self).calls == 1:
            status, payload = 429, {
                "error": {"message": "Rate limit reached ... try again in 0s", "code": "rate_limit_exceeded"}
            }
        else:
            status, payload = 200, {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 10},
            }
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


class Always429(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))  # drain to avoid RST
        body = json.dumps({"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


class CaptureFallback(BaseHTTPRequestHandler):
    """Records the request body it receives and always answers 200."""

    seen: list = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).seen.append(json.loads(body))
        payload = json.dumps(
            {"choices": [{"message": {"content": "from fallback"}}], "usage": {"total_tokens": 5}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


class FallbackUpstream(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))  # drain to avoid RST
        body = json.dumps(
            {"choices": [{"message": {"content": "from fallback"}}], "usage": {"total_tokens": 5}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


def test_absorbs_rate_limit_and_retries():
    RateLimitOnce.calls = 0
    up = _start_server(RateLimitOnce)
    gw = Gateway(f"http://127.0.0.1:{up.server_port}", api_key=None, tpm=1_000_000)

    status, resp, served = gw.forward(
        "POST", "/v1/chat/completions",
        json.dumps({"model": "m", "messages": []}).encode(),
    )

    assert status == 200
    assert served == "upstream"
    assert json.loads(resp)["choices"][0]["message"]["content"] == "ok"
    assert RateLimitOnce.calls == 2  # the 429 was retried, not surfaced


def test_plan_wait_respects_tpm_budget():
    gw = Gateway("http://x", api_key=None, tpm=100)
    gw.record("http://x", 80)  # most of the budget already used
    assert gw.plan_wait("http://x", 50) > 0  # would exceed -> must wait
    assert gw.plan_wait("http://x", 10) == 0  # fits -> send now


def test_plan_wait_per_provider_isolated():
    # Throttling one provider must not throttle the other.
    gw = Gateway("http://a", api_key=None, tpm=100, fallback_upstream="http://b")
    gw.record("http://a", 100)
    assert gw.plan_wait("http://a", 10) > 0
    assert gw.plan_wait("http://b", 10) == 0


def test_falls_back_when_primary_stays_rate_limited():
    up = _start_server(Always429)
    fb = _start_server(FallbackUpstream)
    gw = Gateway(
        f"http://127.0.0.1:{up.server_port}",
        api_key=None,
        tpm=1_000_000,
        fallback_upstream=f"http://127.0.0.1:{fb.server_port}",
    )

    status, resp, served = gw.forward("POST", "/v1/chat/completions", b"{}")

    assert status == 200
    assert served == "fallback"
    assert json.loads(resp)["choices"][0]["message"]["content"] == "from fallback"


def test_fallback_rewrites_model_name():
    up = _start_server(Always429)
    fb = _start_server(CaptureFallback)
    CaptureFallback.seen = []
    gw = Gateway(
        f"http://127.0.0.1:{up.server_port}",
        api_key=None,
        tpm=1_000_000,
        fallback_upstream=f"http://127.0.0.1:{fb.server_port}",
        fallback_model="llama-3.1-8b-instant",
    )

    body = json.dumps({"model": "gemini-2.5-flash", "messages": []}).encode()
    status, _resp, served = gw.forward("POST", "/v1/chat/completions", body)

    assert status == 200
    assert served == "fallback"
    assert CaptureFallback.seen and CaptureFallback.seen[0]["model"] == "llama-3.1-8b-instant"


def test_rotates_through_extra_fallbacks():
    up = _start_server(Always429)          # rung 1: always rate-limited
    fb1 = _start_server(Always429)         # rung 2: also always rate-limited
    fb2 = _start_server(CaptureFallback)   # rung 3: serves
    CaptureFallback.seen = []
    gw = Gateway(
        f"http://127.0.0.1:{up.server_port}",
        api_key=None,
        tpm=1_000_000,
        fallback_upstream=f"http://127.0.0.1:{fb1.server_port}",
        extra_fallbacks=[
            (f"http://127.0.0.1:{fb2.server_port}", None, "llama-3.1-8b-instant", 1_000_000),
        ],
    )

    body = json.dumps({"model": "gemini-2.5-flash", "messages": []}).encode()
    status, resp, served = gw.forward("POST", "/v1/chat/completions", body)

    assert status == 200
    assert served == "fallback"
    assert json.loads(resp)["choices"][0]["message"]["content"] == "from fallback"
    # The third rung received the request with its own model rewrite applied.
    assert CaptureFallback.seen and CaptureFallback.seen[0]["model"] == "llama-3.1-8b-instant"


def test_primary_model_rewrite():
    primary = _start_server(CaptureFallback)
    CaptureFallback.seen = []
    gw = Gateway(
        f"http://127.0.0.1:{primary.server_port}",
        api_key=None,
        primary_model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )

    body = json.dumps({"model": "gemini-2.5-flash", "messages": []}).encode()
    status, _resp, served = gw.forward("POST", "/v1/chat/completions", body)

    assert status == 200
    assert served == "upstream"
    assert CaptureFallback.seen and CaptureFallback.seen[0]["model"] == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_rewrite_model_helpers():
    body = json.dumps({"model": "gemini-2.5-flash", "messages": []}).encode()
    assert json.loads(_rewrite_model(body, "llama-3.1-8b-instant"))["model"] == "llama-3.1-8b-instant"
    assert _rewrite_model(b"not json", "x") == b"not json"


def test_key_for_picks_env_by_provider(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-token")
    # Explicit value wins.
    assert _key_for("https://api.cloudflare.com/...", "explicit") == "explicit"
    # URL-based env lookup.
    assert _key_for("https://api.cloudflare.com/client/v4/accounts/x/ai/v1", None) == "cf-token"
    assert _key_for("https://api.groq.com/openai/v1", None) == "gsk-token"
    # Unknown host with no env -> None (tokenless endpoints like local Ollama).
    assert _key_for("http://localhost:11434/v1", None) is None


def test_join_url_v1_conventions():
    # Groq/Ollama bases already end in /v1 -> strip the incoming /v1 prefix.
    assert _join_url("https://api.groq.com/openai/v1", "/v1/chat/completions") == (
        "https://api.groq.com/openai/v1/chat/completions"
    )
    assert _join_url("http://localhost:11434/v1", "/v1/models") == "http://localhost:11434/v1/models"
    # Gemini's base ends in /openai -> keep the /v1 path as-is.
    assert _join_url(
        "https://generativelanguage.googleapis.com/v1beta/openai", "/v1/chat/completions"
    ) == "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
