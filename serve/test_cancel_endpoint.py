"""POST /v1/cancel: a clean cancel that does not require the client to close the socket.

The endpoint sets the active request's cancel event (the same threading.Event that
note_client_disconnect sets on a socket close); the engine's _control thread sends STOP,
the engine replies DONE cancel=True, and the conversation is preserved (parked to RAM or
spilled to disk). The HTTP response confirms the cancel was acknowledged.

Cases covered:
  1. idle: no request in flight → 200 {"status": "idle"} (idempotent: two POSTs in a row)
  2. cancelled: a request is in flight → 200 {"status": "cancelled", "cancels": 1, "elapsed_ms": ...}
  3. timed_out: the engine did not acknowledge within timeout_s → 200 {"status": "timed_out", ...}
  4. authorization: missing API key → 401
  5. foreign Origin → 403
  6. invalid timeout_s query param → 400
"""
import io
import json
import os
import queue
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_lifecycle import ResidentEngine  # noqa: E402


class CancelEngine(ResidentEngine):
    """A ResidentEngine whose generate() blocks on a gate event so a request stays in flight
    until the test releases it (or the cancel arrives)."""
    def __init__(self, tok):
        super().__init__(tok)
        self.gate = threading.Event()
        self.cancel_seen = threading.Event()
        self.generated = 0

    def generate(self, ids, max_new, sampling, cancel, **kw):
        # Block until the test releases the gate OR the cancel arrives (whichever first).
        while not self.gate.is_set() and not cancel.is_set():
            if cancel.wait(0.05):
                break
        if cancel.is_set():
            self.cancel_seen.set()
            return
        # gate released normally: emit one token
        self.generated += 1
        yield 123


class CancelEndpointTest(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = CancelEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        # API key + CORS for the authorization tests
        self.svc.api_key = "test-secret"
        self.svc.cors_origins = ["https://allowed.example"]
        # Default the cancel timeout short so a stuck engine fails the test fast, not the suite.
        self.svc.stop_wait_s = 1.0
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        # Release any pending gate so the request thread finishes.
        self.engine.gate.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None, method="POST"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode() or "{}")

    def start_request_in_background(self):
        """Send a /v1/chat/completions request that blocks (engine.gate) so it stays in flight."""
        result = {}
        def go():
            try:
                code, body = self.request(
                    "/v1/chat/completions",
                    {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                    headers={"Authorization": "Bearer test-secret"},
                )
                result["code"] = code
                result["body"] = body
            except Exception as e:
                result["error"] = str(e)
        t = threading.Thread(target=go, daemon=True)
        t.start()
        # Wait for the request to register its cancel event with svc.active_cancels.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.svc.status_lock:
                if self.svc.active_cancels:
                    return t, result
            time.sleep(0.02)
        return t, result

    # ── 1. Idle: no request in flight ────────────────────────────────────────────────────
    def test_idle_when_no_request_in_flight(self):
        code, body = self.request("/v1/cancel", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "idle")

    def test_idle_is_idempotent(self):
        # Two POSTs in a row when nothing is running: both 200, both idle.
        for _ in range(2):
            code, body = self.request("/v1/cancel", headers={"Authorization": "Bearer test-secret"})
            self.assertEqual(code, 200)
            self.assertEqual(body["status"], "idle")

    # ── 2. Cancelled: a request is in flight ─────────────────────────────────────────────
    def test_cancel_running_request(self):
        self.engine.loaded = True
        t, result = self.start_request_in_background()
        # The request is now blocked in engine.generate(); its cancel event is in active_cancels.
        code, body = self.request("/v1/cancel", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(body["cancels"], 1)
        self.assertIn("elapsed_ms", body)
        # The engine saw the cancel.
        self.assertTrue(self.engine.cancel_seen.wait(timeout=2.0))
        # The blocked request eventually returns.
        t.join(timeout=5.0)
        self.assertNotIn("error", result)

    # ── 3. Timed out: engine does not acknowledge within timeout_s ───────────────────────
    def test_timed_out_when_engine_does_not_acknowledge(self):
        # Replace cancel_running's view of active_cancels with a fake that never gets pruned:
        # a request whose cancel event stays in active_cancels forever.
        fake_cancel = threading.Event()
        with self.svc.status_lock:
            self.svc.active_cancels.add(fake_cancel)
        try:
            # timeout_s=0.2 → must return "timed_out" quickly.
            code, body = self.request("/v1/cancel?timeout_s=0.2",
                                      headers={"Authorization": "Bearer test-secret"})
            self.assertEqual(code, 200, body)
            self.assertEqual(body["status"], "timed_out")
            self.assertEqual(body["cancels"], 1)
            # The cancel WAS set, even though the engine did not acknowledge.
            self.assertTrue(fake_cancel.is_set())
        finally:
            with self.svc.status_lock:
                self.svc.active_cancels.discard(fake_cancel)

    # ── 4. Authorization: missing API key → 401 ─────────────────────────────────────────
    def test_unauthorized_without_api_key(self):
        code, body = self.request("/v1/cancel")   # no Authorization header
        self.assertEqual(code, 401)

    # ── 5. Foreign Origin → 403 ──────────────────────────────────────────────────────────
    def test_foreign_origin_rejected(self):
        code, body = self.request("/v1/cancel",
                                  headers={"Authorization": "Bearer test-secret",
                                           "Origin": "https://attacker.example"})
        self.assertEqual(code, 403)

    # ── 6. Invalid timeout_s query param → 400 ──────────────────────────────────────────
    def test_invalid_timeout_s(self):
        for bad in ("?timeout_s=abc", "?timeout_s=-1", "?timeout_s=0"):
            code, body = self.request(f"/v1/cancel{bad}",
                                      headers={"Authorization": "Bearer test-secret"})
            self.assertEqual(code, 400, bad)
            self.assertIn("timeout_s", body["error"]["message"])

    # ── 7. The conversation survives the cancel (the cancel path preserves state) ───────
    def test_cancel_does_not_crash_the_engine(self):
        self.engine.loaded = True
        t, _ = self.start_request_in_background()
        code, body = self.request("/v1/cancel", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "cancelled")
        # The engine is still alive after the cancel.
        self.assertTrue(self.engine.alive())
        # A second /v1/cancel says idle (the first one ended the request).
        code2, body2 = self.request("/v1/cancel", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(code2, 200)
        self.assertEqual(body2["status"], "idle")


if __name__ == "__main__":
    unittest.main()
