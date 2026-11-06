"""Tests for tools/strata_extract_experts_bin.py: the concatenation, the hole-punch guards, the local
offsets of an intermediate, the shard resolution (a single-file GGUF such as Gyro-M) and an end-to-end
`--phase all` over a hand-built GGUF (no download, no gguf-py).

The phases that read a GGUF (extract_from_shard, finalize) are also checked end to end against a minimal
but valid single-file GGUF written here byte by byte - the Gyro-M shape: Q8_0 experts, a prism.hadamard
metadata block, one file, no -NNNNN-of-NNNNN name - so a regression of the global-offset bug or of the
sidecar's v5 parity is caught without 70 GB of GGUFs.

    python tools/test_extract_experts_bin.py
"""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_extract_experts_bin as X  # noqa: E402
import iq_pack  # noqa: E402    (on X's sys.path already; named import for the parity checks below)
import gguf_reader  # noqa: E402


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


class ResolveShards(unittest.TestCase):
    """--gguf resolves a -NNNNN-of-NNNNN family from any of its shards, and takes any other name as a
    single-file GGUF (Gyro-S / Gyro-M and their variants ship as one file)."""

    def test_any_shard_of_a_family_gives_the_whole_family(self):
        for i in (1, 2):
            with tempfile.TemporaryDirectory() as d:
                shard = Path(d) / f"M-0000{i}-of-00002.gguf"
                shards, what = X.resolve_shards(shard)
                self.assertEqual([s.name for s in shards],
                                 ["M-00001-of-00002.gguf", "M-00002-of-00002.gguf"])
                self.assertIn("2 shards", what)

    def test_a_plain_name_is_a_single_file_model(self):
        name = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0-noPLE.gguf"   # the file this fix is for
        shards, what = X.resolve_shards(Path("/models") / name)
        self.assertEqual(shards, [Path("/models") / name])
        self.assertIn("single", what)

    def test_the_single_file_itself_is_the_only_shard_even_beside_a_family(self):
        with tempfile.TemporaryDirectory() as d:
            one = Path(d) / "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
            (Path(d) / "other-00001-of-00002.gguf").touch()
            shards, _ = X.resolve_shards(one)
            self.assertEqual(shards, [one])


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


class IqPackContract(unittest.TestCase):
    """The two contracts with iq_pack.py without which an extracted experts.bin is unusable: the sidecar's
    hash is of the v5 table on a prism.hadamard model (Gyro), and iq_pack's reuse check does not compare
    blobs with a hole-punched GGUF (they read as zeros)."""

    EXTRACT = (ROOT / "tools" / "strata_extract_experts_bin.py").read_text(encoding="utf-8")
    IQ = (ROOT / "tools" / "iq_pack.py").read_text(encoding="utf-8")

    def test_finalize_hashes_the_same_text_iq_pack_hashes(self):
        self.assertIn("had = hadamard_spec(model.files[0].metadata, layout", self.EXTRACT)
        self.assertIn("text = native_experts_v5(text)", self.EXTRACT)
        self.assertIn("sidecar = experts_source(model, text, total)", self.EXTRACT)

    def test_iq_pack_skips_the_blob_comparison_for_a_punched_gguf(self):
        self.assertIn("def gguf_is_punched(", self.IQ)
        self.assertIn("if reuse and any(gguf_is_punched(p) for p in model.paths):", self.IQ)


# ------------------------------------------------------------------------------ a hand-built single-file GGUF
# minimal but valid GGUF v3: the Gyro-M shape (one file, Q8_0 experts, F32 routers, prism.hadamard), small
# enough to build and read in milliseconds.  2 layers, 4 experts: blob = 3 * (256/32*34/4) = 204 bytes.

N_EXPERT, LAYERS = 4, 2
Q8_0, F32 = 8, 0
GYRO_M_NAME = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0-noPLE.gguf"


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def kv_str(key: str, value: str) -> bytes:
    return gstr(key.encode()) + struct.pack("<I", 8) + gstr(value.encode())


def kv_u32(key: str, value: int) -> bytes:
    return gstr(key.encode()) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_str_array(key: str, values) -> bytes:
    return gstr(key.encode()) + struct.pack("<IIQ", 9, 8, len(values)) + b"".join(gstr(v.encode()) for v in values)


def tensor_entry(name: str, dims, type_id: int, offset: int) -> bytes:
    return (gstr(name.encode()) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", type_id, offset))


def build_gyro_m(path: Path) -> bytes:
    """Write the fixture and return the experts.bin its layers must produce: per layer, per expert,
    [68 B gate | 68 B up | 68 B down] (Q8_0: 256 values = 8 blocks of 34 B, 4 experts of 68 B each)."""
    meta = (kv_str("general.architecture", "qwen4exp")
            + kv_u32("prism.hadamard.version", 1)
            + kv_u32("prism.hadamard.block_size", 32)
            + kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
            + kv_str("prism.hadamard.axis", "input-last-dimension")
            + kv_str("prism.hadamard.sign_mode", "identity")
            + kv_str_array("prism.hadamard.weight_names",
                           [f"blk.{l}.ffn_{r}_exps.weight" for l in range(LAYERS) for r in ("gate", "up", "down")]))
    entries, expected = [], b""
    for l in range(LAYERS):
        entries.append((f"blk.{l}.ffn_gate_inp.weight", [2, N_EXPERT], F32, bytes(range(32))))
        exps = {}
        for k, r in enumerate(("gate", "up", "down")):
            # 272 bytes, distinct per tensor and per position, so a misplaced slice cannot pass the checks
            raw = bytes(((i * 7 + (l * 3 + k) * 31) % 251) + 1 + k for i in range(272))
            exps[r] = raw
            entries.append((f"blk.{l}.ffn_{r}_exps.weight", [32, 2, N_EXPERT], Q8_0, raw))
        per = 272 // N_EXPERT
        expected += b"".join(exps["gate"][per * e:per * (e + 1)] + exps["up"][per * e:per * (e + 1)]
                             + exps["down"][per * e:per * (e + 1)] for e in range(N_EXPERT))
    directory, offset = b"", 0
    for name, dims, type_id, raw in entries:
        directory += tensor_entry(name, dims, type_id, offset)
        offset = (offset + len(raw) + 31) // 32 * 32          # GGUF aligns every tensor's data
    head = struct.pack("<IIQQ", 0x46554747, 3, len(entries), 7) + meta + directory
    data_start = (len(head) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(head)
        f.write(bytes(data_start - len(head)))
        at = 0
        for name, dims, type_id, raw in entries:
            f.seek(data_start + at)
            f.write(raw)
            at = (at + len(raw) + 31) // 32 * 32
        f.truncate(data_start + at)
    return expected


class SingleFileEndToEnd(unittest.TestCase):
    """--phase all over a single-file GGUF (the Gyro-M shape): the exact case the -NNNNN-of-NNNNN check used
    to refuse with 'does not follow the pattern'.  Everything is checked for real: the extracted bytes, the
    v5 sidecar (the same dict iq_pack computes), the .done mark and the punch ranges."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gguf = self.dir / GYRO_M_NAME
        self.expected = build_gyro_m(self.gguf)
        self.out = self.dir / "pack"

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra) -> tuple[int, str]:
        argv = ["strata_extract_experts_bin.py", "--gguf", str(self.gguf), "--out", str(self.out),
                "--phase", "all", *extra]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
            rc = X.main()
        return rc, buf.getvalue()

    def test_phase_all_extracts_and_writes_the_v5_sidecar(self):
        rc, printed = self.run_main("--no-punch")
        self.assertEqual(rc, 0, printed)
        self.assertIn("single GGUF", printed)
        self.assertIn("no shard 2", printed)                  # phase 2 reports and does nothing
        self.assertFalse(iq_pack.gguf_is_punched(self.gguf))
        self.assertEqual((self.out / "experts.bin").read_bytes(), self.expected)
        self.assertFalse(list(self.out.glob("expert-*.bin")), "the intermediates must be gone")
        self.assertFalse(list(self.out.glob("*.progress.json")))
        # the sidecar is exactly the dict iq_pack computes for this model: shards, bytes and the hash of the
        # v5 table (a prism.hadamard model), so iq_pack reuses this experts.bin instead of rewriting it
        model = iq_pack.Model(self.gguf)
        layout, text, n_expert, total = iq_pack.expert_layout(model, self.gguf)
        t0 = layout[0][5]
        had = iq_pack.hadamard_spec(model.files[0].metadata, layout, int(t0[0].shape[0]), int(t0[2].shape[0]))
        self.assertTrue(str(had).startswith("# strata hadamard"), "the fixture must be a Hadamard model")
        want = iq_pack.experts_source(model, iq_pack.native_experts_v5(text), total)
        self.assertEqual(total, len(self.expected))
        self.assertEqual(json.loads((self.out / "experts.bin.src.json").read_text(encoding="utf-8")), want)
        self.assertIn("v5", printed)
        # the .done mark that makes setup accept the (punched) GGUF without re-hashing it
        done = self.gguf.with_name(self.gguf.name + ".done")
        self.assertIn("trimmed: gguf punched, experts.bin built", done.read_text(encoding="utf-8"))

    @unittest.skipIf(os.name != "posix", "the punch path probes a POSIX fallocate")
    def test_the_punch_ranges_are_the_expert_tensors_own_bytes(self):
        fake = mock.Mock()
        fake.fallocate.return_value = 0                       # every fallocate "succeeds" without touching the file
        with mock.patch.object(X, "_load_libc", lambda: fake):
            rc, printed = self.run_main()
        self.assertEqual(rc, 0, printed)
        g = gguf_reader.GGUFFile(self.gguf)
        want = {(g.data_start + t.offset, t.expected_bytes()) for t in g.tensors
                if t.name.endswith("_exps.weight")}
        got = {(c.args[2].value, c.args[3].value) for c in fake.fallocate.call_args_list}
        # check_punch_supported probes 4096 B at the tail only for files of a page or more; this fixture is
        # smaller, so every call is one expert tensor's own range - nothing else may ever be punched
        self.assertLess(self.gguf.stat().st_size, 4096)
        self.assertEqual(got, want, "nothing but the expert tensors' own byte ranges may be punched")
        self.assertTrue(all(c.args[1].value == X.FALLOC_FL_PUNCH_HOLE | X.FALLOC_FL_KEEP_SIZE
                            for c in fake.fallocate.call_args_list))
        # and the extract itself is unchanged by the (mocked) punch: the bytes still come from the GGUF
        self.assertEqual((self.out / "experts.bin").read_bytes(), self.expected)


if __name__ == "__main__":
    unittest.main()
