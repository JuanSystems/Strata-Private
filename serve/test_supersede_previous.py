"""supersede_previous: a generation request that arrives while another one runs cancels it.

With the engine one request at a time, a request that arrives while another one runs is its
replacement: the client stopped the old one (a stop the server may never see - a retry's
proxy or the client itself can leave the old socket open) or sent it again after an error.
The new request cancels every other in-flight generation, so the engine is STOPped at once
instead of finishing a request nobody reads while the new one waits behind it.

    python -m unittest serve.test_supersede_previous -v
"""
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.test_lifecycle import ResidentEngine

ROOT = Path(__file__).resolve().parents[1]


class GatedEngine(ResidentEngine):
    """A ResidentEngine whose generate() blocks on a gate until the test releases it (or a
    cancel arrives), so a request stays in flight holding the engine's turn."""

    def __init__(self, tok):
        super().__init__(tok)
        self.tok = tok
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.cancels_seen = 0

    def generate(self, ids, max_new, sampling, cancel, **kw):
        self.entered.set()
        while not self.gate.is_set() and not cancel.is_set():
            if cancel.wait(0.02):
                break
        if cancel.is_set():
            self.cancels_seen += 1
            return
        for t in self.tok.encode("Hello."):
            yield t


class SupersedePreviousTest(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = GatedEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.ids = tok.encode("hi")
        self.addCleanup(self.engine.gate.set)              # never leave a request stuck
        patch = mock.patch.object(server, "QUEUE_BEAT_S", 0.05)
        patch.start()
        self.addCleanup(patch.stop)

    def run_in_background(self, **kw):
        """svc.run() on a thread: its events, its cancel Event, and the thread."""
        cancel = kw.pop("cancel", threading.Event())
        events, gen = [], self.svc.run(self.ids, False, None, 8, {}, cancel, **kw)
        thread = threading.Thread(target=lambda: events.extend(list(gen)), daemon=True)
        return thread, events, cancel

    def finish_of(self, events):
        return [x for k, x in events if k == "done"][0]

    # ── 1. The new request cancels the one in flight ────────────────────────────────────
    def test_new_request_cancels_the_in_flight_one(self):
        tA, evA, cancelA = self.run_in_background()
        tA.start()
        self.assertTrue(self.engine.entered.wait(timeout=2.0))   # A holds the engine

        tB, evB, _ = self.run_in_background()
        tB.start()
        # B's registration superseded A: its cancel event is set without any socket closing.
        self.assertTrue(cancelA.wait(timeout=2.0), "the new request never cancelled the in-flight one")

        self.engine.gate.set()                                   # B (still queued or just started) may run
        tA.join(timeout=5.0)
        tB.join(timeout=5.0)
        self.assertFalse(tA.is_alive() or tB.is_alive())

        self.assertEqual(self.finish_of(evA)["finish"], "cancel")
        self.assertEqual(self.engine.cancels_seen, 1)             # A was cancelled inside generate
        self.assertEqual(self.finish_of(evB)["finish"], "length")  # B answered normally
        # A is out of the registry; B left it when it ended.
        self.assertEqual(self.svc.gen_requests, set())

    # ── 2. Off: the old queue (the second request waits, the first is not cancelled) ───
    def test_off_restores_the_queue(self):
        self.svc.supersede_previous = False
        tA, evA, cancelA = self.run_in_background()
        tA.start()
        self.assertTrue(self.engine.entered.wait(timeout=2.0))

        tB, evB, _ = self.run_in_background()
        tB.start()
        time.sleep(0.3)                                          # B queues; nothing cancels A
        self.assertFalse(cancelA.is_set())
        with self.svc.status_lock:
            self.assertEqual(self.svc.status["queued"], 1)       # B is really waiting

        self.engine.gate.set()
        tA.join(timeout=5.0)
        tB.join(timeout=5.0)
        self.assertFalse(tA.is_alive() or tB.is_alive())
        self.assertFalse(cancelA.is_set())
        self.assertNotEqual(self.finish_of(evA)["finish"], "cancel")
        self.assertNotEqual(self.finish_of(evB)["finish"], "cancel")

    # ── 3. A queued request is superseded too (the newest one wins) ────────────────────
    def test_a_queued_request_is_superseded_by_a_newer_one(self):
        tA, evA, cancelA = self.run_in_background()
        tA.start()
        self.assertTrue(self.engine.entered.wait(timeout=2.0))   # A holds the engine

        tB, evB, cancelB = self.run_in_background()
        tB.start()                                               # B queues behind A
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.svc.status_lock:
                if self.svc.status["queued"] == 1:
                    break
            time.sleep(0.01)

        tC, evC, _ = self.run_in_background()
        tC.start()                                               # C supersedes A and B
        self.assertTrue(cancelA.wait(timeout=2.0))
        self.assertTrue(cancelB.wait(timeout=2.0), "the queued request was not superseded")

        self.engine.gate.set()
        tA.join(timeout=5.0)
        tB.join(timeout=5.0)
        tC.join(timeout=5.0)
        self.assertFalse(any(t.is_alive() for t in (tA, tB, tC)))
        # B never took the turn: the queue's own cancel path, with no tokens generated.
        self.assertEqual(self.finish_of(evB)["finish"], "cancel")
        self.assertEqual(self.finish_of(evB)["completion_tokens"], 0)
        self.assertEqual(self.finish_of(evC)["finish"], "length")   # C is the one that answered

    # ── 4. Never with --batch: the slots run requests concurrently ──────────────────────
    def test_batch_mode_never_supersedes(self):
        self.engine.batch = 2                                    # "parallel": requests run at once
        tA, evA, cancelA = self.run_in_background()
        tA.start()
        self.assertTrue(self.engine.entered.wait(timeout=2.0))

        tB, evB, _ = self.run_in_background()
        tB.start()
        time.sleep(0.3)
        self.assertFalse(cancelA.is_set())                      # nobody cancelled A

        self.engine.gate.set()
        tA.join(timeout=5.0)
        tB.join(timeout=5.0)
        self.assertFalse(tA.is_alive() or tB.is_alive())
        self.assertFalse(cancelA.is_set())
        self.assertNotEqual(self.finish_of(evA)["finish"], "cancel")
        self.assertNotEqual(self.finish_of(evB)["finish"], "cancel")

    # ── 5. STRATA_SUPERSEDE_PREVIOUS=0 disables it regardless of anything else ─────────
    def test_env_disables_it(self):
        with mock.patch.dict(os.environ, {"STRATA_SUPERSEDE_PREVIOUS": "0"}):
            tok = ByteTokenizer()
            engine = GatedEngine(tok)
            svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            self.assertFalse(svc.supersede_previous)


if __name__ == "__main__":
    unittest.main()
