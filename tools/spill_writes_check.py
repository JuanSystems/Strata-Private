"""Disk-write check for the conversation spill (docs/PLAN_SPILL_ESCRITURAS.md), for a GPU box.

Drives a running Strata server (its OpenAI API) through ONE scripted multi-turn conversation with the
shapes of the 2026-11-08 Gyro-M log: a long system prompt, turns that grow by a few hundred tokens, a
clock line in every user message, and every few turns a retry of the same turn with the clock moved
on. The clock is fake and deterministic, so two runs send byte-identical prompts. Optionally the
server is stopped (a hard kill by default: what the disk must survive) and started again mid-way,
so the next turn restores the conversation from disk.

Afterwards it reads the engine log the run produced: the bytes written to disk by kind (the
`wrote ... MiB to disk` lines), journal appends, gain skips, captures, the restore after the restart,
the tokens each request reused, and every discard / failure line.

    run      one configuration of the server -> a JSON file
    compare  a baseline JSON and a candidate JSON -> a table; exit 1 when the candidate logged a
             failure, and with --strict also when the replies differ

Only the standard library. Example (Kaggle, two runs):

    python tools/spill_writes_check.py run --name baseline --log /kaggle/working/strata.log \\
        --start "bash run_server.sh" --turns 12 --restart-after 6 --out baseline.json
    python tools/spill_writes_check.py run --name lean --log /kaggle/working/strata.log \\
        --start "bash run_server.sh --conversation-spill-lean" --turns 12 --restart-after 6 --out lean.json
    python tools/spill_writes_check.py compare baseline.json lean.json
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

MIB = 1024 * 1024

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu cache layer token expert state "
         "journal record sidecar index prefix turn checkpoint restore spill writer budget").split()


# ── the scripted conversation ───────────────────────────────────────────────────────────────

def filler(seed, words):
    """`words` deterministic words (a fixed linear-congruential walk over WORDS)."""
    out, x = [], (seed * 2654435761 + 12345) & 0xFFFFFFFF
    for _ in range(words):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out)


def clock(step):
    """A fake clock that moves 7 s per request: deterministic across runs."""
    s = 36000 + 7 * step
    return "%02d:%02d:%02d" % (s // 3600 % 24, s // 60 % 60, s % 60)


def system_prompt(words):
    return ("You are a terse assistant. Answer every message in one short paragraph and never ask a "
            "question back. Reference material follows.\n\n" + filler(1, words))


def user_message(turn, step, words):
    return ("Current time: %s\nTurn %d. Read this note and summarise it in one sentence:\n%s"
            % (clock(step), turn, filler(100 + turn, words)))


def plan(turns, retry_every):
    """The request plan: (turn, kind) pairs; a retry repeats its turn with the clock moved on."""
    out = []
    for t in range(1, turns + 1):
        out.append((t, "turn"))
        if retry_every > 0 and t % retry_every == 0:
            out.append((t, "retry"))
    return out


# ── the server ──────────────────────────────────────────────────────────────────────────────

def http(url, payload=None, key=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="GET" if payload is None else "POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(base, key, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            models = http(base + "/v1/models", key=key, timeout=10)
            return [m.get("id") for m in models.get("data", [])]
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(3)
    raise SystemExit("the server at %s did not answer /v1/models within %d s" % (base, timeout))


class Server:
    """The server this run starts (--start) and stops (--hard: SIGKILL its process group; else --stop
    or SIGTERM). Without --start the script only talks to a server someone else runs."""

    def __init__(self, args):
        self.args, self.proc = args, None

    def start(self):
        if not self.args.start:
            return
        log = open(self.args.log, "ab") if self.args.start_to_log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.args.start, shell=True, stdout=log, stderr=log,
                                     start_new_session=True)

    def stop(self):
        if self.args.stop and not self.args.hard:
            subprocess.run(self.args.stop, shell=True, check=False)
        elif self.proc is not None:
            sig = signal.SIGKILL if self.args.hard else signal.SIGTERM
            try:
                os.killpg(os.getpgid(self.proc.pid), sig)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        elif self.args.stop:
            subprocess.run(self.args.stop, shell=True, check=False)
        self.proc = None
        time.sleep(3)


# ── the engine log ──────────────────────────────────────────────────────────────────────────

RE_WROTE = re.compile(r"conversation spill: wrote ([\d.]+) MiB to disk \(bin ([\d.]+), stage ([\d.]+), "
                      r"dlt ([\d.]+), rst ([\d.]+), idx ([\d.]+), pfx ([\d.]+)\)")
RE_APPEND = re.compile(r"conversation spill \[delta\]: \+(\d+) tokens .*appended to conv-(\d+)\.dlt")
RE_GAIN = re.compile(r"conversation spill \[[^\]]+\]: skip \(gain")
RE_CAPTURE = re.compile(r"conversation spill \[[^\]]+\]: captured (\d+)(?: of (\d+))? tokens")
RE_SKIP_HELD = re.compile(r"conversation spill: skip capture \(")
RE_RESTORE = re.compile(r"conversation spill: restored (\d+) tokens from disk .*?(?:read ([\d.]+) MiB in (\d+) ms)?$")
RE_PROMPT = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
RE_FORCED = re.compile(r"writing the backup the gain skipped")
RE_SIDECAR = re.compile(r"state sidecar of conv-\d+ holds (.*)$")
RE_FAIL = re.compile(r"DISCARD|background write FAILED|tail capture \(from \d+ tokens\) dropped|refusing to write|"
                     r"apply failed|is missing or corrupt|^ERR |allocation failed during spill|capture failed")


def parse_log(text):
    """Everything the check reads from one stretch of the engine log."""
    out = {"wrote_mib": 0.0, "wrote": {k: 0.0 for k in ("bin", "stage", "dlt", "rst", "idx", "pfx")},
           "writes": 0, "appends": 0, "gain_skips": 0, "captures": 0, "captured_tokens": 0, "held_skips": 0,
           "forced": 0, "restores": [], "prompts": [], "failures": [], "sidecar": None}
    for line in text.splitlines():
        m = RE_WROTE.search(line)
        if m:
            vals = [float(x) for x in m.groups()]
            out["wrote_mib"] += vals[0]
            out["writes"] += 1
            for k, v in zip(("bin", "stage", "dlt", "rst", "idx", "pfx"), vals[1:]):
                out["wrote"][k] += v
        if RE_APPEND.search(line):
            out["appends"] += 1
        if RE_GAIN.search(line):
            out["gain_skips"] += 1
        m = RE_CAPTURE.search(line)
        if m:
            out["captures"] += 1
            out["captured_tokens"] += int(m.group(1))
        if RE_SKIP_HELD.search(line):
            out["held_skips"] += 1
        if RE_FORCED.search(line):
            out["forced"] += 1
        m = RE_RESTORE.search(line)
        if m:
            out["restores"].append({"tokens": int(m.group(1)),
                                    "mib": float(m.group(2)) if m.group(2) else None,
                                    "ms": int(m.group(3)) if m.group(3) else None})
        m = RE_PROMPT.search(line)
        if m:
            out["prompts"].append({"tokens": int(m.group(1)), "reused": int(m.group(2)),
                                   "read": int(m.group(3)), "ms": int(m.group(4))})
        m = RE_SIDECAR.search(line)
        if m and out["sidecar"] is None:
            out["sidecar"] = m.group(1)
        if RE_FAIL.search(line):
            out["failures"].append(line.strip()[:300])
    return out


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def log_read(path, start):
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ── run / compare ───────────────────────────────────────────────────────────────────────────

def clean_spill(directory):
    """Remove the spill files (conv-* and pfx-*, nothing else) of `directory`; returns how many."""
    n = 0
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if name.startswith("conv-") or name.startswith("pfx-"):
            try:
                os.remove(os.path.join(directory, name))
                n += 1
            except OSError:
                pass
    return n


def run(args):
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    base = args.url.rstrip("/")
    if args.clean_spill:
        print("%s: removed %d spill files from %s" % (args.name, clean_spill(args.clean_spill), args.clean_spill))
    server = Server(args)
    log0 = log_size(args.log)
    server.start()
    models = wait_ready(base, key, args.ready_timeout)
    model = args.model or (models[0] if models else "strata")
    messages = [{"role": "system", "content": system_prompt(args.system_words)}]
    results, step, restart_at, segments = [], 0, None, []
    seg_start = log0
    requests = plan(args.turns, args.retry_every)
    for i, (turn, kind) in enumerate(requests):
        if args.restart_after and restart_at is None and turn == args.restart_after + 1 and kind == "turn":
            # let the idle backup land (and its write), then kill the server and start it again
            time.sleep(args.settle)
            segments.append(log_read(args.log, seg_start))
            server.stop()
            seg_start = log_size(args.log)
            server.start()
            wait_ready(base, key, args.ready_timeout)
            restart_at = i
        step += 1
        msg = {"role": "user", "content": user_message(turn, step, args.grow_words)}
        body = {"model": model, "messages": messages + [msg], "max_tokens": args.max_tokens,
                "temperature": 0, "top_p": 1, "seed": 0, "stream": False}
        t0 = time.time()
        reply = http(base + "/v1/chat/completions", body, key=key, timeout=args.request_timeout)
        dt = time.time() - t0
        choice = reply["choices"][0]
        text = choice["message"].get("content") or ""
        usage = reply.get("usage", {})
        results.append({"i": i, "turn": turn, "kind": kind, "reply": text, "finish": choice.get("finish_reason"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"), "seconds": round(dt, 2),
                        "after_restart": restart_at is not None and i == restart_at})
        # the turn's own request is kept unless a retry of it follows: then the retry's is
        next_is_retry = i + 1 < len(requests) and requests[i + 1] == (turn, "retry")
        if not next_is_retry:
            messages += [msg, {"role": "assistant", "content": text}]
        print("%-8s turn %2d %-5s %6s prompt tokens  %5.1f s  %s" % (args.name, turn, kind,
              usage.get("prompt_tokens"), dt, text[:60].replace("\n", " ")), flush=True)
        time.sleep(args.pause)
    time.sleep(args.settle)
    segments.append(log_read(args.log, seg_start))
    if not args.keep:
        server.stop()
    parsed = [parse_log(s) for s in segments]
    total = parse_log("\n".join(segments))
    out = {"name": args.name, "args": {k: v for k, v in vars(args).items() if k != "func"}, "model": model,
           "results": results, "segments": parsed, "total": total}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("%s: %.1f MiB written to disk in %d writes (%d journal appends, %d gain skips); %d failure lines -> %s"
          % (args.name, total["wrote_mib"], total["writes"], total["appends"], total["gain_skips"],
             len(total["failures"]), args.out))
    return 0


def first_after_restart(doc):
    """The first request after the restart: its log prompt line (reused/read) and the restore line."""
    if len(doc["segments"]) < 2:
        return None, None
    seg = doc["segments"][1]
    return (seg["prompts"][0] if seg["prompts"] else None), (seg["restores"][0] if seg["restores"] else None)


def compare(args):
    with open(args.baseline, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        b = json.load(f)
    ta, tb = a["total"], b["total"]
    rows = [("MiB written to disk", "%.1f" % ta["wrote_mib"], "%.1f" % tb["wrote_mib"])]
    for k in ("bin", "stage", "dlt", "rst", "idx", "pfx"):
        rows.append(("  of which " + k, "%.1f" % ta["wrote"][k], "%.1f" % tb["wrote"][k]))
    for k, label in (("writes", "background writes"), ("appends", "journal appends"),
                     ("captures", "captures (GPU -> RAM)"), ("gain_skips", "gain skips"),
                     ("held_skips", "skip capture (already held)"), ("forced", "forced by the idle timer")):
        rows.append((label, str(ta[k]), str(tb[k])))
    pa, ra = first_after_restart(a)
    pb, rb = first_after_restart(b)
    fmt_r = lambda r: "-" if not r else "%d tok%s" % (r["tokens"], "" if r["mib"] is None else
                                                     ", %.0f MiB in %d ms" % (r["mib"], r["ms"]))
    fmt_p = lambda p: "-" if not p else "%d reused, %d read in %d ms" % (p["reused"], p["read"], p["ms"])
    rows.append(("restore after the restart", fmt_r(ra), fmt_r(rb)))
    rows.append(("first prompt after the restart", fmt_p(pa), fmt_p(pb)))
    same = sum(1 for x, y in zip(a["results"], b["results"]) if x["reply"] == y["reply"])
    n = min(len(a["results"]), len(b["results"]))
    rows.append(("replies identical", "%d / %d" % (same, n), ""))
    rows.append(("failure lines", str(len(ta["failures"])), str(len(tb["failures"]))))
    w = max(len(r[0]) for r in rows)
    print("%-*s  %-28s  %s" % (w, "", a["name"], b["name"]))
    for r in rows:
        print("%-*s  %-28s  %s" % (w, r[0], r[1], r[2]))
    if ta["wrote_mib"] > 0:
        print("\nthe candidate writes %.0f%% of the baseline's bytes" % (100.0 * tb["wrote_mib"] / ta["wrote_mib"]))
    if tb["sidecar"]:
        print("candidate sidecar: " + tb["sidecar"])
    for line in tb["failures"][:20]:
        print("FAILURE (candidate): " + line)
    diffs = [(x["i"], x["turn"], x["kind"]) for x, y in zip(a["results"], b["results"]) if x["reply"] != y["reply"]]
    if diffs:
        print("replies differ at requests: " + ", ".join("#%d (turn %d %s)" % d for d in diffs[:20]))
    bad = bool(tb["failures"]) or (args.strict and bool(diffs))
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="drive one server configuration, write a JSON")
    r.add_argument("--name", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--log", required=True, help="the engine log file this run appends to")
    r.add_argument("--url", default="http://127.0.0.1:8095", help="the server (serve/server.py's default port: 8095)")
    r.add_argument("--clean-spill", default=None, metavar="DIR",
                   help="delete the conv-* / pfx-* spill files in DIR before the start, so a run never restores "
                        "what an earlier run wrote")
    r.add_argument("--model", default=None)
    r.add_argument("--api-key-env", default=None)
    r.add_argument("--start", default=None, help="shell command that starts the server (run in its own session)")
    r.add_argument("--stop", default=None, help="shell command that stops it (default: signal the --start session)")
    r.add_argument("--hard", action="store_true", default=True, help="kill -9 at the restart (the default)")
    r.add_argument("--soft", dest="hard", action="store_false", help="a clean stop at the restart instead")
    r.add_argument("--start-to-log", action="store_true", help="append the --start command's output to --log")
    r.add_argument("--turns", type=int, default=12)
    r.add_argument("--retry-every", type=int, default=4)
    r.add_argument("--restart-after", type=int, default=6, help="restart after this turn (0: never)")
    r.add_argument("--system-words", type=int, default=3000)
    r.add_argument("--grow-words", type=int, default=300)
    r.add_argument("--max-tokens", type=int, default=48)
    r.add_argument("--pause", type=float, default=8.0, help="seconds between requests (past the 5 s idle backup)")
    r.add_argument("--settle", type=float, default=30.0, help="seconds to wait before the restart and at the end")
    r.add_argument("--ready-timeout", type=int, default=1800)
    r.add_argument("--request-timeout", type=int, default=3600)
    r.add_argument("--keep", action="store_true", help="leave the server running at the end")
    r.set_defaults(func=run)
    c = sub.add_parser("compare", help="baseline JSON vs candidate JSON")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--strict", action="store_true", help="exit 1 when the replies differ too")
    c.set_defaults(func=compare)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
