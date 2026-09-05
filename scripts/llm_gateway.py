"""Local LLM gateway: an OpenAI-compatible proxy with rate-limit absorption
and a multi-provider failover chain.

Point Py-Nerve's agent (or any OpenAI-compatible client) at this server instead
of a provider. The gateway:

1. Forwards to the primary upstream (e.g. Gemini).
2. Paces requests against a per-provider tokens-per-minute budget.
3. Waits out 429 rate limits (so the agent never sees them) up to a cap.
4. Rotates through a chain of fallbacks (e.g. Groq llama-3.1-8b-instant, local
   Ollama) when the primary stays exhausted — rewriting the model name per
   fallback so each provider understands the request.

Everything runs on your machine — no third-party hop. The last rung should be
local (Ollama), which has no rate limits at all.

Usage:
    # terminal 1 — Gemini primary, Groq fallback, local Ollama last resort
    set GOOGLE_API_KEY=...
    set GROQ_API_KEY=...
    .venv/Scripts/python.exe scripts/llm_gateway.py --port 8787 ^
        --fallback-upstream https://api.groq.com/openai/v1 --fallback-model llama-3.1-8b-instant ^
        --fallback http://localhost:11434/v1 --fallback-model qwen2.5-coder:1.5b --fallback-tpm 0

    # terminal 2 — point the agent at the gateway (no --api-key needed)
    agent.bat --model gemini-2.5-flash --endpoint http://localhost:8787/v1 ...
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("llm_gateway")

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
WINDOW_SECONDS = 60.0
MAX_429_RETRIES = 5
# 404 included because OpenRouter's :free models go offline and answer 404
# ("no providers found"); that should fail over, not crash.
FALLBACK_ON_STATUS = (404, 429, 500, 502, 503, 504)


class Gateway:
    """Rate-limit-aware forwarding proxy with a failover chain.

    The chain is a list of (url, key, rewrite_model, tpm); the first entry is
    the primary (rewrite_model=None, the agent's model passes through).
    """

    def __init__(
        self,
        upstream: str,
        api_key: str | None = None,
        tpm: int = 0,
        fallback_upstream: str | None = None,
        fallback_key: str | None = None,
        fallback_model: str | None = None,
        fallback_tpm: int = 8000,
        max_wait: float = 30.0,
        extra_fallbacks: list[tuple[str, str | None, str | None, int]] | None = None,
        primary_model: str | None = None,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.primary_model = primary_model
        self.max_wait = max_wait
        self._chain: list[tuple[str, str | None, str | None, int]] = [
            (self.upstream, api_key, None, max(0, tpm)),
        ]
        if fallback_upstream:
            self._chain.append(
                (fallback_upstream.rstrip("/"), fallback_key, fallback_model, max(0, fallback_tpm))
            )
        for url, key, model, tpm2 in extra_fallbacks or []:
            self._chain.append((url.rstrip("/"), key, model, max(0, tpm2)))
        self._endpoints: dict[str, dict[str, Any]] = {
            url: {"tpm": tpm3, "usage": deque()} for url, _k, _m, tpm3 in self._chain
        }
        self._lock = threading.Lock()

    # -- per-provider TPM pacing -------------------------------------------
    def _window_used(self, endpoint: str, now: float) -> int:
        q = self._endpoints[endpoint]["usage"]
        while q and now - q[0][0] > WINDOW_SECONDS:
            q.popleft()
        return sum(t for _, t in q)

    def plan_wait(self, endpoint: str, req_tokens: int) -> float:
        """Seconds to wait before sending a request of ~req_tokens tokens to
        ``endpoint`` so its sliding 60s window stays under its TPM budget."""
        tpm = self._endpoints[endpoint]["tpm"]
        if tpm <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            used = self._window_used(endpoint, now)
            if used + req_tokens <= tpm:
                return 0.0
            remaining = used
            for ts, toks in sorted(self._endpoints[endpoint]["usage"]):
                remaining -= toks
                if remaining + req_tokens <= tpm:
                    return max(0.0, ts + WINDOW_SECONDS - now)
            return WINDOW_SECONDS

    def record(self, endpoint: str, total_tokens: int) -> None:
        """Account actual tokens used on ``endpoint`` (from the response)."""
        if total_tokens <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._window_used(endpoint, now)
            self._endpoints[endpoint]["usage"].append((now, total_tokens))

    # -- forwarding ---------------------------------------------------------
    def _call_primary(self, method: str, path: str, body: bytes, req_tokens: int) -> tuple[int, bytes]:
        """Primary call with 429 absorption (waits out short rate-limit windows)."""
        wait = self.plan_wait(self.upstream, req_tokens)
        if wait > 0.5:
            logger.info("pacing primary: waiting %.1fs (TPM budget)", wait)
            time.sleep(wait)

        body_p = _rewrite_model(body, self.primary_model) if self.primary_model else body
        total_wait = 0.0
        attempts = 0
        while True:
            status, resp = _call(self.upstream, self.api_key, method, path, body_p)
            if status != 429:
                return status, resp
            attempts += 1
            retry_after = _retry_seconds(resp)
            total_wait += retry_after
            if attempts >= MAX_429_RETRIES or total_wait > self.max_wait:
                return status, resp
            logger.warning("primary 429, waiting %.1fs (attempt %d/%d)", retry_after, attempts, MAX_429_RETRIES)
            time.sleep(retry_after)
        return status, resp  # pragma: no cover

    def forward(self, method: str, path: str, body: bytes) -> tuple[int, bytes, str]:
        """Forward to the chain. Returns (status, body, served_by)."""
        req_tokens = len(body) // 4
        model = _model_from_body(body) or "-"

        status, resp = self._call_primary(method, path, body, req_tokens)
        served_url = self.upstream

        if status in FALLBACK_ON_STATUS:
            for url, key, fb_model, _tpm in self._chain[1:]:
                logger.warning("primary -> %s; trying fallback %s (model=%s)", status, url, fb_model or model)
                wait = self.plan_wait(url, req_tokens)
                if wait > 0.5:
                    logger.info("pacing fallback: waiting %.1fs (TPM budget)", wait)
                    time.sleep(wait)
                fb_body = _rewrite_model(body, fb_model) if fb_model else body
                status, resp = _call(url, key, method, path, fb_body)
                if status not in FALLBACK_ON_STATUS:
                    served_url = url
                    break

        usage = _total_tokens(resp)
        self.record(served_url, usage)
        served = "upstream" if served_url == self.upstream else "fallback"
        logger.info(
            "%s %s model=%s -> %d (req=%d tok, used=%d tok, served=%s)",
            method, path, model, status, req_tokens, usage, served,
        )
        return status, resp, served


def _key_for(url: str, explicit: str | None) -> str | None:
    """Resolve the API key for an upstream URL: explicit value wins, otherwise
    the matching environment variable (so secrets never live in config files)."""
    if explicit:
        return explicit
    host = (url or "").lower()
    if "groq.com" in host:
        return os.environ.get("GROQ_API_KEY")
    if "generativelanguage" in host:
        return os.environ.get("GOOGLE_API_KEY")
    if "cloudflare" in host:
        return os.environ.get("CLOUDFLARE_API_TOKEN")
    if "openrouter" in host:
        return os.environ.get("OPENROUTER_API_KEY")
    if "mistral" in host:
        return os.environ.get("MISTRAL_API_KEY")
    if "cerebras" in host:
        return os.environ.get("CEREBRAS_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def _join_url(base: str, path: str) -> str:
    """Join an OpenAI-compatible base URL with a ``/v1/...`` path.

    Bases that already end in ``/v1`` (e.g. ``https://api.groq.com/openai/v1``)
    get the leading ``/v1`` stripped from the path; bases that don't (e.g.
    Gemini's ``.../v1beta/openai``) keep it.
    """
    if re.search(r"/v1$", base):
        path = re.sub(r"^/v1(?=/|$)", "", path)
    return base + path


def _call(base: str, key: str | None, method: str, path: str, body: bytes) -> tuple[int, bytes]:
    url = _join_url(base, path)
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    # GET must not carry a body: some upstreams reject it. Only POST sends data.
    data = body if method.upper() == "POST" else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        # Dropped connections, resets, and timeouts are transient: surface as
        # 502 so the caller treats them like an overloaded/rate-limited upstream.
        return 502, json.dumps({"error": {"message": f"gateway upstream error: {e}"}}).encode()


def _model_from_body(body: bytes) -> str | None:
    try:
        return json.loads(body).get("model")
    except Exception:
        return None


def _rewrite_model(body: bytes, new_model: str) -> bytes:
    """Replace the ``model`` field in an OpenAI request body (for fallbacks
    where the model name differs across providers)."""
    try:
        data = json.loads(body)
    except Exception:
        return body
    if isinstance(data, dict):
        data["model"] = new_model
        return json.dumps(data).encode()
    return body


def _retry_seconds(resp: bytes) -> float:
    text = resp.decode("utf-8", errors="replace")
    m = re.search(r"try again in ([\d.]+)s", text)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r'"retry_after"\s*:\s*([\d.]+)', text)
    if m:
        return float(m.group(1)) + 1.0
    return 0.5


def _total_tokens(resp: bytes) -> int:
    """Extract token usage, tolerating providers that only report prompt/completion splits."""
    try:
        usage = json.loads(resp).get("usage", {}) or {}
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)) and total > 0:
            return int(total)
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        return int(prompt) + int(completion)
    except Exception:
        return 0


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMGateway/0.1"
    gateway: "Gateway | None" = None

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return self._send(404, json.dumps({"error": {"message": f"not found: {self.path}"}}).encode())
        gateway = Handler.gateway
        assert gateway is not None
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        status, resp, _served = gateway.forward("POST", self.path, body)
        self._send(status, resp)

    def do_GET(self) -> None:
        gateway = Handler.gateway
        assert gateway is not None
        if self.path.startswith("/v1/models"):
            status, resp, _served = gateway.forward("GET", "/v1/models", b"")
            self._send(status, resp)
        else:
            self._send(404, json.dumps({"error": {"message": "not found"}}).encode())

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # silence noisy default
        logger.debug(fmt, *args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local LLM gateway (rate-limit absorption + failover chain)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--upstream", default="https://generativelanguage.googleapis.com/v1beta/openai")
    ap.add_argument("--model", default=None, help="Rewrite the model name for the primary rung (e.g. @cf/meta/llama-3.3-70b-instruct-fp8-fast)")
    ap.add_argument("--api-key", default=None, help="Primary key; defaults to GOOGLE_API_KEY / GROQ_API_KEY / OPENAI_API_KEY")
    ap.add_argument("--tpm", type=int, default=0, help="Primary TPM budget (0 = unlimited)")
    ap.add_argument("--fallback-upstream", default=None, help="Second rung (e.g. https://api.groq.com/openai/v1)")
    ap.add_argument("--fallback-key", default=None, help="Second-rung key; defaults to GROQ_API_KEY")
    ap.add_argument("--fallback-model", default=None, help="Model rewrite for the second rung (e.g. llama-3.1-8b-instant)")
    ap.add_argument("--fallback-tpm", type=int, default=8000, help="Second-rung TPM budget (0 = unlimited)")
    ap.add_argument("--fallback", action="append", default=None, help="Extra fallback upstream URL (repeatable; tried in order)")
    ap.add_argument("--fallback-key2", action="append", default=None, help="Keys for the extra fallbacks (repeatable, in order)")
    ap.add_argument("--fallback-model2", action="append", default=None, help="Model rewrites for extra fallbacks (repeatable, in order)")
    ap.add_argument("--fallback-tpm2", action="append", type=int, default=None, help="TPM budgets for extra fallbacks (repeatable, in order)")
    ap.add_argument("--max-wait", type=float, default=30.0, help="Max seconds to wait out primary 429s before failing over")
    ap.add_argument("--log-file", default="llm_gateway.log")
    args = ap.parse_args(argv)
    args.api_key = _key_for(args.upstream, args.api_key)
    args.fallback_key = _key_for(args.fallback_upstream or "", args.fallback_key)

    extra_fallbacks: list[tuple[str, str | None, str | None, int]] = []
    for i, url in enumerate(args.fallback or []):
        keys = args.fallback_key2 or []
        models = args.fallback_model2 or []
        tpms = args.fallback_tpm2 or []
        extra_fallbacks.append(
            (
                url,
                _key_for(url, keys[i] if i < len(keys) else None),
                models[i] if i < len(models) else None,
                tpms[i] if i < len(tpms) else 0,
            )
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logger.addHandler(fh)

    Handler.gateway = Gateway(
        args.upstream, args.api_key, args.tpm,
        args.fallback_upstream, args.fallback_key, args.fallback_model, args.fallback_tpm,
        args.max_wait, extra_fallbacks, args.model,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    chain = [args.upstream]
    if args.fallback_upstream:
        chain.append(args.fallback_upstream)
    chain.extend(args.fallback or [])
    print(f"LLM gateway on http://localhost:{args.port}/v1")
    if args.model:
        print(f"  primary model rewrite: {args.model}")
    for i, url in enumerate(chain, start=1):
        print(f"  rung {i}: {url}")
    print(f"Point the agent here: --endpoint http://localhost:{args.port}/v1")
    print("Logs: llm_gateway.log   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
