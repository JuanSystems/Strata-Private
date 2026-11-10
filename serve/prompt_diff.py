"""Opt-in diagnostic (STRATA_PROMPT_DIFF): say where a prompt rewrites a recent one, and show the text.

The engine resumes a conversation at the last checkpoint before the first token that changed. A client that
rewrites text far back in its history (an old message, its timestamp, a list of open files) makes the engine read
everything after that point again, and the engine's log only shows the position (``diverge@42872``). This module
keeps the last few rendered prompts and, when a new one differs from the most similar of them before that prompt's
answer turn, prints the position and the text on both sides.

A new turn is not reported: the previous prompt's answer turn (``<|im_start|>assistant``, and with
"effort_position": "end" the short system turn in front of it) is where every next request is expected to differ.

STRATA_PROMPT_DIFF=1 prints to the server's output; any other value is a file the lines are appended to.
Off (unset): nothing is kept and nothing changes.
"""
from __future__ import annotations

import collections
import os
import sys
import threading

ENV = "STRATA_PROMPT_DIFF"
TURN = "<|im_start|>"
KEEP = 4            # recent prompts compared (several conversations or agents at once)
CONTEXT = 400       # characters shown on each side of the change


def common_prefix(a: str, b: str) -> int:
    """The length of the longest common prefix of two strings (binary search on C-level slice comparisons)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def answer_start(prompt: str) -> int:
    """Where the prompt's answer turn starts: the last turn marker, or the trailing system turn in front of it
    ("effort_position": "end").  A prompt with no turn marker (another template) has none: its end."""
    last = prompt.rfind(TURN)
    if last <= 0:
        return len(prompt)
    before = prompt.rfind(TURN, 0, last)
    if before > 0 and prompt.startswith(TURN + "system\n", before):
        return before
    return last


def _flat(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n")


def describe(old: str, new: str, at: int, new_tokens: int | None = None, context: int = CONTEXT) -> list[str]:
    """The lines for a change at character `at` (old[:at] == new[:at])."""
    turn = old.rfind(TURN, 0, at + 1)
    if turn >= 0:
        role = old[turn + len(TURN):].split("\n", 1)[0][:24]
        where = (f"in turn {old.count(TURN, 0, turn) + 1} of {old.count(TURN)} ({role}), "
                 f"{at - turn:,} characters into it")
    else:
        where = "before the first turn"
    tokens = ""
    if new_tokens and new:
        tokens = f", about token {round(at * new_tokens / len(new)):,} of {new_tokens:,}"
    return [
        f"[strata] prompt diff: this prompt rewrites a recent one at character {at:,} of {len(old):,} "
        f"({len(old) - at:,} before that prompt's end{tokens}), {where}",
        f"[strata] prompt diff: same before: ...{_flat(old[max(0, at - context):at])}",
        f"[strata] prompt diff: was: {_flat(old[at:at + context])}...",
        f"[strata] prompt diff: now: {_flat(new[at:at + context])}...",
    ]


class PromptDiff:
    """The recent prompts and the check of a new one."""

    def __init__(self, target: str = "1", keep: int = KEEP, context: int = CONTEXT):
        self.target = target
        self.context = context
        self.recent: collections.deque[str] = collections.deque(maxlen=keep)
        self.lock = threading.Lock()

    def check(self, prompt: str, tokens: int | None = None) -> list[str] | None:
        """Compare `prompt` with the most similar recent prompt; report and return the lines when it rewrites text
        before that prompt's answer turn.  `prompt` is kept for the next request."""
        with self.lock:
            best, at = None, -1
            for p in self.recent:
                n = common_prefix(p, prompt)
                if n > at:
                    best, at = p, n
            if best is not None and best == prompt:
                return None                       # the same prompt again (a retry): nothing new to keep
            self.recent.append(prompt)
        if best is None or at >= answer_start(best):
            return None
        lines = describe(best, prompt, at, tokens, self.context)
        self.write(lines)
        return lines

    def write(self, lines: list[str]) -> None:
        text = "\n".join(lines) + "\n"
        if self.target == "1":
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        try:
            with open(self.target, "a", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            print(f"[strata] {ENV}: cannot write {self.target}: {e}", flush=True)


def from_env(config_env: dict | None = None) -> PromptDiff | None:
    """On when STRATA_PROMPT_DIFF is set in the server's environment or, failing that, in the model config's
    "env" (which otherwise only the engine gets)."""
    value = os.environ.get(ENV)
    if value is None and isinstance(config_env, dict):
        value = config_env.get(ENV)
    value = str(value or "").strip()
    return PromptDiff(value) if value and value != "0" else None
