"""STRATA_PROMPT_DIFF (serve/prompt_diff.py): where a prompt rewrites a recent one.  No model, no jinja2."""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from serve import prompt_diff as pd

T = pd.TURN
SYSTEM = T + "system\n" + "You are an agent. " * 50 + "<|im_end|>\n"
EFFORT = T + "system\nReasoning effort is set to low.<|im_end|>\n"
ANSWER = T + "assistant\n<think>\n"


def user(text, time="2026-11-09T21:27:13Z", tabs="a.html"):
    return (T + f"user\n{text}\n<environment_details>\n# VSCode Open Tabs\n{tabs}\n\n# Current Time\n{time}\n"
            "</environment_details><|im_end|>\n")


def reply(text):
    return T + f"assistant\n<think>\n\n</think>\n\n{text}<|im_end|>\n"


class CommonPrefix(unittest.TestCase):
    def test_lengths(self):
        for a, b, n in (("", "", 0), ("abc", "", 0), ("abc", "abc", 3), ("abcd", "abxd", 2), ("ab", "abc", 2),
                        ("x" * 100000 + "a", "x" * 100000 + "b", 100000)):
            self.assertEqual(pd.common_prefix(a, b), n, (a[:5], b[:5]))

    def test_answer_start(self):
        p = SYSTEM + user("hi") + ANSWER
        self.assertEqual(pd.answer_start(p), len(SYSTEM + user("hi")))
        p = SYSTEM + user("hi") + EFFORT + ANSWER          # "effort_position": "end": the tail turn counts
        self.assertEqual(pd.answer_start(p), len(SYSTEM + user("hi")))
        self.assertEqual(pd.answer_start(SYSTEM + ANSWER), len(SYSTEM), "the first system turn is not a tail")
        self.assertEqual(pd.answer_start("no turns here"), len("no turns here"))


class Check(unittest.TestCase):
    def check(self, diff, prompt, tokens=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lines = diff.check(prompt, tokens)
        return lines, out.getvalue()

    def test_a_new_turn_is_not_reported(self):
        diff = pd.PromptDiff()
        history = SYSTEM + user("do it")
        for tail in ("", EFFORT):
            self.assertEqual(self.check(diff, history + tail + ANSWER), (None, ""))
            history += reply("done") + user("next", time="2026-11-09T21:28:00Z")
            self.assertEqual(self.check(diff, history + tail + ANSWER), (None, ""), tail)

    def test_the_same_prompt_again_is_not_reported_nor_kept_twice(self):
        diff = pd.PromptDiff()
        p = SYSTEM + user("do it") + ANSWER
        self.check(diff, p)
        self.assertEqual(self.check(diff, p), (None, ""))
        self.assertEqual(len(diff.recent), 1)

    def test_a_retry_that_changes_the_last_message_is_reported(self):
        diff = pd.PromptDiff()
        history = SYSTEM + user("do it")
        self.check(diff, history + ANSWER)
        lines, out = self.check(diff, SYSTEM + user("do it", time="2026-11-09T21:29:59Z") + ANSWER, tokens=500)
        self.assertIsNotNone(lines)
        self.assertEqual(out, "\n".join(lines) + "\n")
        at = history.index("21:27") + len("21:2")
        self.assertIn(f"at character {at:,} of", lines[0])
        self.assertIn("in turn 2 of 3 (user)", lines[0])
        self.assertIn("about token", lines[0])
        self.assertTrue(lines[1].endswith("# Current Time\\n2026-11-09T21:2"), lines[1])
        self.assertTrue(lines[2].startswith("[strata] prompt diff: was: 7:13Z"), lines[2])
        self.assertTrue(lines[3].startswith("[strata] prompt diff: now: 9:59Z"), lines[3])

    def test_a_rewrite_far_back_is_reported_against_the_closest_prompt(self):
        diff = pd.PromptDiff()
        other = T + "system\nAnother agent.<|im_end|>\n" + user("unrelated") + ANSWER
        turns = [user("task", tabs="a.html")] + [reply(f"step {i}") + user(f"result {i}") for i in range(6)]
        self.check(diff, SYSTEM + "".join(turns) + ANSWER)
        self.check(diff, other)                              # another conversation in between
        changed = [user("task", tabs="a.html\nsettings")] + turns[1:]
        lines, _ = self.check(diff, SYSTEM + "".join(changed) + reply("step 6") + user("result 6") + ANSWER,
                              tokens=4000)
        self.assertIn("in turn 2 of 15 (user)", lines[0])
        self.assertTrue(lines[2].startswith("[strata] prompt diff: was: \\n# Current Time"), lines[2])
        self.assertTrue(lines[3].startswith("[strata] prompt diff: now: settings\\n\\n# Current Time"), lines[3])

    def test_a_file_target(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "diff.log")
            with mock.patch.dict(os.environ, {pd.ENV: path}):
                diff = pd.from_env()
            diff.check(SYSTEM + user("a") + ANSWER)
            lines, out = self.check(diff, SYSTEM + user("b") + ANSWER)
            self.assertEqual(out, "")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "\n".join(lines) + "\n")

    def test_a_restart_compares_with_the_prompts_from_before_it(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "diff.log")
            before = pd.PromptDiff(path)
            before.check(SYSTEM + user("a") + ANSWER)
            self.assertTrue(os.path.exists(path + ".prompts.json"))
            after = pd.PromptDiff(path)                          # a new server process
            self.assertEqual(list(after.recent), list(before.recent))
            lines = after.check(SYSTEM + user("b") + ANSWER)
            self.assertIsNotNone(lines)
            self.assertIn("was: a\\n<environment_details>", lines[2])
            with open(path + ".prompts.json", "w", encoding="utf-8") as f:
                f.write("{broken")
            self.assertEqual(len(pd.PromptDiff(path).recent), 0, "an unreadable file starts empty")
            self.assertFalse(os.path.exists(os.path.join(d, "1.prompts.json")))
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            cwd = os.getcwd()
            os.chdir(d)
            try:
                pd.PromptDiff("1").check(SYSTEM + ANSWER)
                self.assertEqual(os.listdir(d), [], "printing to the output saves nothing")
            finally:
                os.chdir(cwd)

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(pd.from_env())
        with mock.patch.dict(os.environ, {pd.ENV: "0"}):
            self.assertIsNone(pd.from_env())
        with mock.patch.dict(os.environ, {pd.ENV: "1"}):
            self.assertEqual(pd.from_env().target, "1")

    def test_the_config_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(pd.from_env({"STRATA_TRACE": "1"}))
            self.assertEqual(pd.from_env({pd.ENV: "/tmp/d.log"}).target, "/tmp/d.log")
            self.assertEqual(pd.from_env({pd.ENV: 1}).target, "1")
            self.assertIsNone(pd.from_env({pd.ENV: "0"}))
        with mock.patch.dict(os.environ, {pd.ENV: "0"}):           # the server's own environment wins
            self.assertIsNone(pd.from_env({pd.ENV: "1"}))


if __name__ == "__main__":
    unittest.main()
