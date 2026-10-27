"""Tests for tools/strata_extract_experts_bin.py: the concatenation, the hole-punch guards and the local
offsets of an intermediate.

The phases that read a GGUF (extract_from_shard, finalize) need a real model and the vendored gguf-py, which a
fresh checkout does not have; they are checked by their invariant on the source instead (see LocalOffsets), so
a regression of the global-offset bug is caught without 70 GB of GGUFs.

    python tools/test_extract_experts_bin.py
"""
from __future__ import annotations

import errno
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_extract_experts_bin as X  # noqa: E402


class Concatenation(unittest.TestCase):
    """Phase 3: the parts end up back to back, with no hole at the head of any of them."""

    def test_two_equal_parts_give_their_sum(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "expert-01.bin").write_bytes(b"A" * 3000)
            (out / "expert-02.bin").write_bytes(b"B" * 3000)
            self.assertEqual(X.concat_intermediates(out), 0)
            data = (out / "experts.bin").read_bytes()
            self.assertEqual(len(data), 6000, "the concatenation is not the sum of the parts")
            self.assertEqual(data[:3000], b"A" * 3000)
            self.assertEqual(data[3000:], b"B" * 3000)
            self.assertFalse((out / "expert-01.bin").exists())
            self.assertFalse((out / "expert-02.bin").exists())

    def test_one_part_is_a_rename(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "expert-01.bin").write_bytes(b"Z" * 1234)
            self.assertEqual(X.concat_intermediates(out), 0)
            self.assertEqual((out / "experts.bin").stat().st_size, 1234)

    def test_no_parts_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(X.concat_intermediates(Path(d)), 1)

    def test_the_final_file_is_not_taken_as_a_part(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "experts.bin").write_bytes(b"old")
            (out / "expert-01.bin").write_bytes(b"N" * 100)
            self.assertEqual(X.concat_intermediates(out), 0)
            self.assertEqual((out / "experts.bin").stat().st_size, 100)


class HolePunchOnThisSystem(unittest.TestCase):
    """Where fallocate does not exist the tool must say so instead of pretending 35 GB were freed."""

    def test_punch_outside_posix_reports_false(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(b"\0" * 8192)
            with mock.patch.object(X.os, "name", "nt"):
                self.assertFalse(X.punch_hole(p, 0, 4096))
            self.assertEqual(p.stat().st_size, 8192)

    def test_real_disk_usage_outside_posix_is_the_file_size(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "y.bin"
            p.write_bytes(b"\0" * 5000)
            with mock.patch.object(X.os, "name", "nt"):
                self.assertEqual(X.real_disk_usage(p), 5000)

    def test_the_preflight_says_yes_outside_posix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "z.bin"
            p.write_bytes(b"\0" * 8192)
            with mock.patch.object(X.os, "name", "nt"):
                ok, why = X.check_punch_supported(p)
            self.assertTrue(ok)
            self.assertIn("keeps its full size", why)

    def test_a_filesystem_without_punch_is_refused(self):
        """EOPNOTSUPP from the probe: no extract, the user keeps the GGUF whole and must be told."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.bin"
            p.write_bytes(b"\0" * 8192)
            fake = mock.Mock()
            fake.fallocate.return_value = -1
            with mock.patch.object(X, "_load_libc", lambda: fake), \
                    mock.patch.object(X.os, "name", "posix"), \
                    mock.patch.object(X.ctypes, "get_errno", lambda: errno.EOPNOTSUPP):
                ok, why = X.check_punch_supported(p)
            self.assertFalse(ok)
            self.assertIn("does not support hole-punch", why)


class LocalOffsets(unittest.TestCase):
    """The bug this port fixed: a shard whose first layer is not layer 0 must write at 0 in its own intermediate,
    not at the absolute offset of the final experts.bin.  With the global offset, expert-02.bin carries a hole at
    its head, the concatenation adds it to the end, and FileExpertSource refuses the pack."""

    SRC = (ROOT / "tools" / "strata_extract_experts_bin.py").read_text(encoding="utf-8")

    def test_the_intermediate_is_sized_without_the_global_offset(self):
        at = self.SRC.index("need_size = ")
        code = self.SRC[at:self.SRC.index("\n", at)].split("#", 1)[0]     # the trailing comment names the bug
        self.assertIn("sum(entry[4] * n_expert for entry in my_layers)", code)
        self.assertNotIn("global_offset", code)

    def test_the_write_seeks_to_the_local_offset(self):
        self.assertIn("fo.seek(off - global_offset)", self.SRC)
        self.assertNotIn("fo.seek(off)\n", self.SRC)

    def test_the_unused_index_variable_is_gone(self):
        self.assertNotIn("my_first_layer_idx", self.SRC)

    def test_the_sidecar_is_the_one_iq_pack_writes(self):
        self.assertIn("sidecar = experts_source(model, text, total)", self.SRC)
        self.assertIn("from iq_pack import Model, expert_layout, experts_source", self.SRC)

    def test_windows_uses_shutil_and_not_statvfs(self):
        self.assertNotIn("os.statvfs(", self.SRC)                  # the call, not the comment that explains it
        self.assertIn("shutil.disk_usage(out_dir).free", self.SRC)


if __name__ == "__main__":
    unittest.main()
