"""The disk-write check's own logic (no GPU): the log parser on the engine's real lines, the request
plan, the deterministic prompts, and the compare verdict."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spill_writes_check as W  # noqa: E402

# lines as the engine prints them (the 2026-11-08 Gyro-M log, and this change's new lines)
LOG = """\
strata serve: conversation spill [idle after done]: captured 30925 tokens (1129.2 MiB, 1 stage image) in 1032.1 ms; conv-9223372036854775826.bin goes to the background disk writer (no request is stalled; a restart restores it)
strata serve: conversation spill [delta]: +444 tokens (6.5 MiB) appended to conv-9223372036854775821.dlt (base 30481 tokens; journal 41.1 MiB, 5 records; state sidecar 452.5 MiB)
strata serve: conversation spill: wrote 459.0 MiB to disk (bin 0.0, stage 0.0, dlt 6.5, rst 452.5, idx 0.01, pfx 0.0); 0.45 GiB since start
strata serve: conversation spill [delta]: state sidecar of conv-9223372036854775821 holds image 0: live 30925 tokens 113.1 MiB + 30592 (turn) 113.1 MiB
strata serve: conversation spill: skip capture (30956 tokens; the disk already holds this conversation equal or longer)
strata serve: conversation spill [park]: skip (gain: the disk resumes this conversation at 30592, 827 behind the live turn 31419; --conversation-spill-gain 2048)
strata serve: conversation spill [idle after done]: captured 31419 of 31539 tokens (cut at the turn checkpoint) (K/V copied from token 30592; 1 stage image) in 210.4 ms; serial 9223372036854775900 goes to the background disk writer
strata serve: conversation spill: 60 s without a request; writing the backup the gain skipped (--conversation-spill-gain-max-seconds)
strata serve: conversation spill: restored 32784 tokens from disk (deeper than the RAM cache's 29415-token match and the live resume's 0); read 1802.4 MiB in 1240 ms
strata serve: prompt 33679 tokens = 32440 reused + 1239 read in 33952 ms (36.5 tok/s), 344 generated in 14581 ms (23.6 tok/s), drafts accepted 239 of 288, 4 checkpoints
[strata diag] DISCARD: main snapshot invalid: conversation snapshot: incompatible runtime geometry
strata serve: conversation spill: tail capture (from 1040 tokens) dropped: no journal base reaches it any more; the next park writes it whole
"""


class Parser(unittest.TestCase):
    def test_lines(self):
        p = W.parse_log(LOG)
        self.assertAlmostEqual(p["wrote_mib"], 459.0)
        self.assertAlmostEqual(p["wrote"]["rst"], 452.5)
        self.assertAlmostEqual(p["wrote"]["dlt"], 6.5)
        self.assertEqual(p["writes"], 1)
        self.assertEqual(p["appends"], 1)
        self.assertEqual(p["gain_skips"], 1)
        self.assertEqual(p["captures"], 2)
        self.assertEqual(p["captured_tokens"], 30925 + 31419)
        self.assertEqual(p["held_skips"], 1)
        self.assertEqual(p["forced"], 1)
        self.assertEqual(p["restores"], [{"tokens": 32784, "mib": 1802.4, "ms": 1240}])
        self.assertEqual(p["prompts"], [{"tokens": 33679, "reused": 32440, "read": 1239, "ms": 33952}])
        self.assertTrue(p["sidecar"].startswith("image 0: live 30925 tokens"))
        self.assertEqual(len(p["failures"]), 2)

    def test_restore_line_of_an_older_engine(self):
        p = W.parse_log("strata serve: conversation spill: restored 100 tokens from disk (deeper than the RAM "
                        "cache's 0-token match and the live resume's 0)\n")
        self.assertEqual(p["restores"], [{"tokens": 100, "mib": None, "ms": None}])

    def test_empty(self):
        p = W.parse_log("")
        self.assertEqual((p["wrote_mib"], p["writes"], p["failures"]), (0.0, 0, []))


class Script(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(W.plan(5, 2), [(1, "turn"), (2, "turn"), (2, "retry"), (3, "turn"), (4, "turn"),
                                        (4, "retry"), (5, "turn")])
        self.assertEqual(W.plan(3, 0), [(1, "turn"), (2, "turn"), (3, "turn")])

    def test_deterministic(self):
        self.assertEqual(W.system_prompt(50), W.system_prompt(50))
        self.assertEqual(W.user_message(3, 7, 40), W.user_message(3, 7, 40))
        self.assertNotEqual(W.user_message(3, 7, 40), W.user_message(3, 8, 40))   # the clock moved
        self.assertEqual(W.clock(0), "10:00:00")
        self.assertEqual(W.clock(9), "10:01:03")
        self.assertEqual(len(W.filler(5, 300).split()), 300)


class Clean(unittest.TestCase):
    def test_only_spill_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("conv-1.bin", "conv-1.dlt", "conv-1.s0-5-ab.st", "pfx-2.bin", "keep.txt", "model.gguf"):
                (Path(d) / name).write_text("x")
            self.assertEqual(W.clean_spill(d), 4)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["keep.txt", "model.gguf"])
        self.assertEqual(W.clean_spill(str(Path(d) / "missing")), 0)


class Compare(unittest.TestCase):
    def doc(self, name, mib, replies, failures=()):
        total = W.parse_log("")
        total["wrote_mib"], total["failures"] = mib, list(failures)
        return {"name": name, "total": total, "segments": [total],
                "results": [{"i": i, "turn": i + 1, "kind": "turn", "reply": r} for i, r in enumerate(replies)]}

    def run_compare(self, a, b, strict=False):
        with tempfile.TemporaryDirectory() as d:
            pa, pb = Path(d) / "a.json", Path(d) / "b.json"
            pa.write_text(json.dumps(a))
            pb.write_text(json.dumps(b))
            argv = ["compare", str(pa), str(pb)] + (["--strict"] if strict else [])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = W.main(argv)
            return code, buf.getvalue()

    def test_verdicts(self):
        base = self.doc("baseline", 8000.0, ["x", "y"])
        code, text = self.run_compare(base, self.doc("lean", 400.0, ["x", "y"]))
        self.assertEqual(code, 0)
        self.assertIn("5% of the baseline", text)
        self.assertEqual(self.run_compare(base, self.doc("lean", 400.0, ["x", "z"]))[0], 0)
        self.assertEqual(self.run_compare(base, self.doc("lean", 400.0, ["x", "z"]), strict=True)[0], 1)
        self.assertEqual(self.run_compare(base, self.doc("lean", 400.0, ["x", "y"], ["DISCARD: x"]))[0], 1)


if __name__ == "__main__":
    unittest.main()
