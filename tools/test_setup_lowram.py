"""Tests for setup.py's low-RAM messages and the paging question (#250): the one-GPU warning says why and shows the
RAM math, and an explicit --low-ram off installs with --yes instead of stopping at "Install it anyway?" (default
n).  Mocked input - no GPU, no downloads.

    python -m unittest tools.test_setup_lowram
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text); `stdin` answers input() (None: never asked)."""
    out = io.StringIO()
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    code = None
    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class OneGpuWhy(unittest.TestCase):
    def test_auto_shows_the_ram_math(self):
        lines = setup.low_ram_one_gpu_why("IQ3_XXS", 46, "auto")       # the #250 report: 46 GB, IQ3_XXS
        text = "\n".join(lines)
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        need = arena + setup.LOW_RAM_HEADROOM_GB
        self.assertIn(f"{arena:.0f} + {setup.LOW_RAM_HEADROOM_GB} = {need:.0f} GB", text)
        self.assertIn("this PC has 46 GB", text)
        self.assertIn("layer split", text)                            # why one GPU is recommended
        self.assertIn("--gpus 0,1", text)                             # #364 #384: and the way to use them all
        self.assertIn("OS file cache", text)
        self.assertNotIn("--low-ram off", text)                       # (that pages the experts: not the way)

    def test_an_explicit_choice_says_so(self):
        text = "\n".join(setup.low_ram_one_gpu_why("Q2_0", 64, "resident"))
        self.assertIn("you chose the low-RAM mode (--low-ram resident)", text)


class PagingQuestion(unittest.TestCase):
    MODEL, RAM = "IQ3_XXS", 46                                       # 60 GB wanted, 46 here

    def test_yes_with_low_ram_off_installs(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "off", True)
        self.assertIsNone(code, out)
        self.assertEqual(asked, [])
        self.assertIn("installing IQ3_XXS with 46 GB of RAM, as you chose (--low-ram off)", out)

    def test_yes_with_auto_still_stops(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "auto", True)
        self.assertEqual(code, 1)
        self.assertIn("IQ3_XXS needs about 60 GB of RAM; this PC has 46 GB", out)
        self.assertIn("--low-ram off --yes", out)                    # the way to insist

    def test_interactive(self):
        for choice, answer, want in (("off", "", None), ("auto", "", 1), ("auto", "y", None), ("off", "n", 1)):
            with self.subTest(choice=choice, answer=answer):
                code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, choice, False, stdin=answer)
                self.assertEqual(code, want, out)
                self.assertEqual(len(asked), 1)
                self.assertIn("[y]" if choice == "off" else "[n]", asked[0])


class DraftLayerCountsInTheFit(unittest.TestCase):
    """The draft layer (MTP) takes about 1 GB of graphics memory out of the expert cache: every fit check counts it while it is on."""

    def test_the_card_holds_one_gb_less_with_mtp_on(self):
        on = setup.low_ram_gpu_gb("IQ3_XXS", 12.0)
        off = setup.low_ram_gpu_gb("IQ3_XXS", 12.0, mtp=False)
        self.assertAlmostEqual(off - on, setup.MTP_VRAM_GB)
        self.assertEqual(setup.MTP_VRAM_GB, 1.0)
        self.assertAlmostEqual(setup.low_ram_gpu_share("IQ3_XXS", 12.0, mtp=False) - setup.low_ram_gpu_share("IQ3_XXS", 12.0),
                               1.0 / setup.MODELS["IQ3_XXS"]["arena_gb"])

    def test_the_edge_of_what_fits_moves_by_the_draft_layer(self):
        arena = setup.MODELS["IQ1_M"]["arena_gb"]
        ram = arena + 6 - 7.0 + 0.5            # a card of 12 GB holds 7 GB beside the dense part; the draft layer needs one of them
        self.assertFalse(setup.low_ram_fits("IQ1_M", ram, 12.0))
        self.assertTrue(setup.low_ram_fits("IQ1_M", ram, 12.0, mtp=False))


class PunchedGguflsVerify(unittest.TestCase):
    """A GGUF whose expert ranges were hole-punched reads them as zeros, so its SHA-256 can never match the
    published one again.  The .done mark says `trimmed:` and the apparent size still has to be right: a shard
    cut short must still be caught.  verify_sha256 is exercised as it is, no production code is replaced."""

    SIZE = 4096
    SHA = "0" * 64

    def _shard(self, tmp: Path, size: int):
        p = tmp / "Qwen-test-00001-of-00002.gguf"
        p.write_bytes(b"\0" * size)
        return p

    def test_a_trimmed_mark_skips_the_hash_when_the_size_matches(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            p = self._shard(Path(d), self.SIZE)
            p.with_name(p.name + ".done").write_text(
                "trimmed: gguf punched, experts.bin built\n", encoding="utf-8")
            calls = []
            with mock.patch.object(setup, "_hash_file", lambda *a, **k: calls.append(a) or self.SHA):
                setup.verify_sha256(p, self.SIZE, self.SHA)
            self.assertEqual(calls, [], "111 GB were hashed although the mark says trimmed")

    def test_a_trimmed_mark_with_the_wrong_size_is_refused(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            p = self._shard(Path(d), self.SIZE - 100)          # cut short: must not be accepted
            p.with_name(p.name + ".done").write_text(
                "trimmed: gguf punched, experts.bin built\n", encoding="utf-8")
            calls = []
            with mock.patch.object(setup, "_hash_file", lambda *a, **k: calls.append(a) or self.SHA):
                with self.assertRaises(SystemExit):
                    setup.verify_sha256(p, self.SIZE, self.SHA)
            self.assertEqual(calls, [], "a truncated GGUF reached the hash instead of being refused first")

    def test_a_mark_with_the_pinned_sha_needs_no_hash_at_all(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            p = self._shard(Path(d), self.SIZE)
            p.with_name(p.name + ".done").write_text(f"sha256 {self.SHA}\n", encoding="utf-8")
            calls = []
            with mock.patch.object(setup, "_hash_file", lambda *a, **k: calls.append(a) or self.SHA):
                setup.verify_sha256(p, self.SIZE, self.SHA)
            self.assertEqual(calls, [])


class NoExpertsBinGate(unittest.TestCase):
    """STRATA_NO_EXPERTS_BIN is read once, before the space arithmetic, and again in the write and the
    resident/mapped choice.  These read setup.py's source: the gates live inside install(), which downloads
    tens of GB, so they are asserted where they are written."""

    SRC = (ROOT / "setup.py").read_text(encoding="utf-8")

    def test_the_gate_is_read_before_real_is_summed(self):
        gate = self.SRC.index('_no_ebin = os.environ.get("STRATA_NO_EXPERTS_BIN") == "1"')
        real = self.SRC.index('real = to_fetch + prep_gb + mtp_fetch +')
        self.assertLess(gate, real)
        self.assertIn("(0 if _no_ebin else", self.SRC[real:real + 400])

    def test_the_write_of_experts_bin_is_skipped_when_the_gate_is_on(self):
        at = self.SRC.index('if low_ram and not ready and not (pack / "experts.bin").exists() and')
        self.assertIn('os.environ.get("STRATA_NO_EXPERTS_BIN") != "1"', self.SRC[at:at + 200])
        self.assertIn("STRATA_NO_EXPERTS_BIN=1: not writing experts.bin", self.SRC)

    def test_both_variants_stay_available_without_the_file(self):
        self.assertIn('_force_res = os.environ.get("STRATA_RESIDENT_VARIANT") == "1"', self.SRC)
        at = self.SRC.index('if os.environ.get("STRATA_NO_EXPERTS_BIN") == "1":')
        block = self.SRC[at:at + 1400]
        self.assertIn('args += ["--resident-experts"]', block)
        self.assertIn('args += ["--mmap-experts"]', block)
        self.assertIn('args += ["--resident-experts" if resident else "--mmap-experts"]', block)

    def test_the_default_path_is_unchanged(self):
        # the else branch is the old one line, byte for byte
        self.assertIn('        else:\n            args += ["--resident-experts" if resident else "--mmap-experts"]',
                      self.SRC)


if __name__ == "__main__":
    unittest.main()
