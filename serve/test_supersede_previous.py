"""supersede_previous: a generation request that arrives while another one runs cancels it.

With the engine one request at a time, a request that arrives while another one runs is its
replacement: the client stopped the old one (a stop the server may never see - a retry's
proxy or the client itself can leave the old socket open) or sent it again after an error.
The new request cancels every other in-flight generation, so the engine is STOPped at once
instead of finishing a request nobody reads while the new one waits behind it.

Under --batch (upstream's slots, a split engine's lanes, "Concurrent requests") the same
holds where the engine is really one request at a time: one slot supersedes on arrival like
the solo path, and with every slot busy only the request that can only wait behind the
holders supersedes them (StrataEngine.on_slot_stall) - an arrival with a free slot supersedes
nobody, because the engine really runs those together.

    python -m unittest serve.test_supersede_previous -v
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve
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

    # ── 4. --batch with 2+ slots: the slots run requests concurrently, an arrival supersedes nobody ──
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

    # ── 4b. --batch 1: one slot is one request at a time - the arrival supersedes all the same ──
    def test_one_batch_slot_supersedes_on_arrival(self):
        self.engine.batch = 1                                    # one slot: still one request at a time
        tA, evA, cancelA = self.run_in_background()
        tA.start()
        self.assertTrue(self.engine.entered.wait(timeout=2.0))

        tB, evB, _ = self.run_in_background()
        tB.start()
        self.assertTrue(cancelA.wait(timeout=2.0), "one slot: the new request never cancelled the in-flight one")

        self.engine.gate.set()
        tA.join(timeout=5.0)
        tB.join(timeout=5.0)
        self.assertFalse(tA.is_alive() or tB.is_alive())
        self.assertEqual(self.finish_of(evA)["finish"], "cancel")
        self.assertEqual(self.finish_of(evB)["finish"], "length")  # B answered normally

    # ── 5. STRATA_SUPERSEDE_PREVIOUS=0 disables it regardless of anything else ─────────
    def test_env_disables_it(self):
        with mock.patch.dict(os.environ, {"STRATA_SUPERSEDE_PREVIOUS": "0"}):
            tok = ByteTokenizer()
            engine = GatedEngine(tok)
            svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            self.assertFalse(svc.supersede_previous)


# The fake batch engine for the slot tests below: speaks the engine's GEN / BGEN / T / BT / DONE /
# BDONE / BADM / BSTOP lines over stdio (as test_parallel.py's does), with a reply whose length the
# prompt picks: one with LONG in it gets --long tokens (plus <|im_end|>, 257), a short one gets 4.
# Tokens stream every --step seconds, so a request that is left alone runs for --long x --step
# seconds and one that is superseded ends a moment after its STOP / BSTOP.
FAKE_SLOTS = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1])
STEP = float(args[args.index("--step") + 1]) if "--step" in args else 0.06
LONG = int(args[args.index("--long") + 1]) if "--long" in args else 60
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print(f"INFO engine=0.1.56 batch_slots={slots}", flush=True)
print("READY 65536 stop", flush=True)
active = {}      # slot -> [tokens left, max_new, produced]
stopped = set()  # BSTOPped slots: they end "cancel" at the next window
def window():
    for b in sorted(active):
        left, max_new, produced = active[b]
        if b in stopped or stop.is_set():
            print(f"BDONE {b} {produced} cancel 1.0", flush=True)
            stopped.discard(b)
            del active[b]
            continue
        t = left.pop(0)
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if not left or produced >= max_new:
            print(f"BDONE {b} {produced} length 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith("BYIELD "):
        continue                                # too late: the read it was for has ended
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = [int(x) for x in f[-1].split(",")]
        n = LONG if b"LONG" in bytes(x & 255 for x in ids) else 4
        toks = ([65] * n) + [257]               # 'A' * n, then <|im_end|>
        stop.clear()
        out = 0                                 # (a short prompt: no PP lines)
        for t in toks[:1 if slot is not None else max_new]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 0", flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class BatchSlotsSupersedeTest(unittest.TestCase):
    """supersede_previous over the real StrataEngine and Service, with the fake batch engine behind them:
    the engine's slots as upstream's --batch / a split's lanes serve them, and the cancel Events doing what
    they do against a real pipe (STOP / BSTOP out, DONE / BDONE back)."""

    LONG_S = 60 * 0.06                                      # a LONG reply left alone: this many seconds

    def start(self, slots, env=None):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_slots.py"
        script.write_text(FAKE_SLOTS, encoding="utf-8")
        real = server.subprocess.Popen
        extra = ["--batch", str(slots), "--step", "0.06", "--long", "60"]
        env_patch = mock.patch.dict(os.environ, env or {})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.engine.unload)
        self.addCleanup(self.tmp.cleanup)

    def chat(self, text, max_tokens=96):
        """POST /v1/chat/completions -> (completion tokens, seconds the whole answer took)."""
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=60) as r:
            answer = json.loads(r.read().decode())
        return answer["usage"]["completion_tokens"], time.monotonic() - t0, answer

    def go(self, out, key, text, max_tokens=96):
        def run():
            out[key] = self.chat(text, max_tokens=max_tokens)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    # ── every slot busy: the request that can only wait behind the holders supersedes them ──
    def test_a_request_that_cannot_get_a_slot_supersedes_the_holders(self):
        self.start(2)
        out = {}
        t1 = self.go(out, "a1", "LONG one")                       # both slots end up held by long
        time.sleep(0.3)
        t2 = self.go(out, "a2", "LONG two")                       # requests nobody is reading any more
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not all(self.engine.slot_busy):
            time.sleep(0.02)
        self.assertTrue(all(self.engine.slot_busy), "both slots are held by the long requests")

        b0 = time.monotonic()
        n_b, took_b, answer_b = self.chat("quick")                # ... and a new one arrives
        waited_b = time.monotonic() - b0
        t1.join(timeout=30.0)
        t2.join(timeout=30.0)
        self.assertFalse(t1.is_alive() or t2.is_alive())

        self.assertEqual(answer_b["choices"][0]["message"]["content"], "AAAA")
        self.assertLess(waited_b, self.LONG_S, "the new request waited behind the replaced ones")
        for k in ("a1", "a2"):
            n, took, answer = out[k]
            self.assertLess(n, 40, f"{k} was not superseded: {n} tokens")
            self.assertLess(took, self.LONG_S, f"{k} ran to its end: {took:.1f} s")

    # ── a free slot: the arrival supersedes nobody (the engine really runs them together) ──
    def test_an_arrival_with_a_free_slot_supersedes_nobody(self):
        self.start(2)
        out = {}
        t1 = self.go(out, "a", "LONG one")
        time.sleep(0.3)                                           # a is in the engine; a slot is still free
        n_b, took_b, answer_b = self.chat("quick")
        t1.join(timeout=30.0)
        self.assertFalse(t1.is_alive())

        n_a, took_a, answer_a = out["a"]
        self.assertEqual(answer_b["choices"][0]["message"]["content"], "AAAA")
        self.assertGreaterEqual(n_a, 55, "the long request was cut short: it had a slot of its own")
        self.assertGreaterEqual(took_a, self.LONG_S * 0.8, "the long request ended too early")

    # ── one slot: one request at a time behind a fancier protocol - the arrival supersedes ──
    def test_one_slot_the_new_request_takes_the_engine(self):
        self.start(1)
        out = {}
        t1 = self.go(out, "a", "LONG one")
        time.sleep(0.5)

        b0 = time.monotonic()
        n_b, took_b, answer_b = self.chat("quick")
        waited_b = time.monotonic() - b0
        t1.join(timeout=30.0)
        self.assertFalse(t1.is_alive())

        n_a, took_a, answer_a = out["a"]
        self.assertEqual(answer_b["choices"][0]["message"]["content"], "AAAA")
        self.assertLess(waited_b, self.LONG_S * 0.5, "the new request waited behind the replaced one")
        self.assertLess(n_a, 40, "the long request was not superseded")

    # ── STRATA_SUPERSEDE_PREVIOUS=0: the queue - the new request waits for the old one's end ──
    def test_env_off_the_new_request_waits_for_the_slot(self):
        self.start(1, env={"STRATA_SUPERSEDE_PREVIOUS": "0"})
        self.assertFalse(self.svc.supersede_previous)
        out = {}
        t1 = self.go(out, "a", "LONG one")
        time.sleep(0.5)

        b0 = time.monotonic()
        n_b, took_b, answer_b = self.chat("quick")
        waited_b = time.monotonic() - b0
        t1.join(timeout=30.0)
        self.assertFalse(t1.is_alive())

        n_a, took_a, answer_a = out["a"]
        self.assertEqual(answer_b["choices"][0]["message"]["content"], "AAAA")
        self.assertGreaterEqual(waited_b, self.LONG_S * 0.6,
                                "the queue must let the new request wait behind the running one")
        self.assertGreaterEqual(n_a, 55, "with the queue the running request is not cancelled")


if __name__ == "__main__":
    unittest.main()
