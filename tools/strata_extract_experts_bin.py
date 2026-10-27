#!/usr/bin/env python3
"""strata_extract_experts_bin.py

Build `experts.bin` from the GGUF shards of Qwen3.8-Flash-Next by extracting the
experts layer by layer and hole-punching the ranges they came from.  This keeps
the disk use CONSTANT at about 70 GB during the whole operation, instead of
growing to about 105 GB as the standard setup does (it writes experts.bin
WITHOUT punching the GGUF).

Made for PCs with a tight disk (e.g. 90 GB free with 70 GB of GGUFs already
downloaded: the standard setup needs 70 + 35.5 = 105.5 GB and fails there).

How it works (the disk arithmetic):
  - the GGUFs always LOOK like 70 GB (the file size does not change)
  - the GGUFs on disk really start at 70 GB and end at about 32.5 GB (the
    ~35.5 GB of expert tensors become sparse / holes)
  - experts.bin grows from 0 to 35.5 GB
  - the real sum on disk is always about 70 GB while it runs, about 68 GB at the end

WHERE THE SAVING IS REAL: hole-punch is `fallocate(FALLOC_FL_PUNCH_HOLE)`,
which exists on Linux only.  On Windows and macOS this tool does NOT punch, so
the GGUF keeps its full size and the extract only rebuilds experts.bin.  On
these systems use `tools/iq_pack.py --experts-bin` (the standard path, which
also copies, with no space saved) or run the setup with
`STRATA_NO_EXPERTS_BIN=1` (which does not write experts.bin at all, but needs
the 70 GB of GGUFs untouched).

Phases (`--phase`):
  1  extract the experts of shard 1 -> expert-01.bin (intermediate), punch shard 1
  2  extract the experts of shard 2 -> expert-02.bin (intermediate), punch shard 2
  3  concatenate expert-01.bin + expert-02.bin -> experts.bin (streaming; the
     first intermediate is deleted as it is consumed)
  4  cleanup: delete the second intermediate, write the sidecar
     experts.bin.src.json, update the .done marks

`--phase all` runs 1+2+3+4.  A run can also be done step by step, which is the
recommended way on a very tight disk.

Requirements:
  - Linux for the hole-punch (elsewhere it extracts without punching, as
    iq_pack.py does)
  - Python 3.10+, numpy
  - the model's GGUFs already downloaded

Typical use (from inside the Strata repo):
  python3 tools/strata_extract_experts_bin.py \\
      --gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \\
      --out /path/to/data/packs/iq2xs \\
      --phase all

Resuming: if interrupted, phases 1 and 2 skip the tensors already extracted
(recorded in expert-01.bin.progress.json and expert-02.bin.progress.json).
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ---- locate iq_pack.py (the GGUF reader and the pack logic live there)
SCRIPT_PATH = Path(__file__).resolve()
CANDIDATES = [
    SCRIPT_PATH.parent.parent / "tools" / "iq_pack.py",       # <repo>/scripts/foo.py -> <repo>/tools/
    SCRIPT_PATH.parent / "tools" / "iq_pack.py",             # <repo>/foo.py -> <repo>/tools/
    SCRIPT_PATH.parent / "iq_pack.py",                       # <repo>/tools/foo.py -> <repo>/tools/
    Path.cwd() / "tools" / "iq_pack.py",                     # from cwd
    Path.cwd() / "iq_pack.py",                               # from cwd/tools/
]
_strata_repo = os.environ.get("STRATA_REPO_PATH")
if _strata_repo:
    CANDIDATES.insert(0, Path(_strata_repo) / "tools" / "iq_pack.py")

IQ_PACK_PATH = next((c for c in CANDIDATES if c.exists()), None)
if IQ_PACK_PATH is None:
    print("[X] no iq_pack.py found. Options:")
    print("    1) run this script from the root of the Strata repo")
    print("    2) set STRATA_REPO_PATH=/path/to/Strata")
    print("    3) copy this script into <repo>/tools/ and run it from there")
    sys.exit(1)

sys.path.insert(0, str(IQ_PACK_PATH.parent))
import numpy as np
import iq_pack
from iq_pack import Model, expert_layout, experts_source


# ---- fallocate constants (Linux)
FALLOC_FL_KEEP_SIZE = 0x01
FALLOC_FL_PUNCH_HOLE = 0x02


def _load_libc():
    """libc with `fallocate` reachable.  `libc.so.6` is glibc's soname; musl
    (Alpine) and other libcs do not ship it, so fall back to the process' own
    handle (CDLL(None)) before giving up."""
    for name in ("libc.so.6", None):
        try:
            lib = ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
        if not hasattr(lib, "fallocate"):
            continue
        lib.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
        lib.fallocate.restype = ctypes.c_int
        return lib
    return None


def punch_hole_linux(fd: int, offset: int, length: int) -> bool:
    """Punch a range of the file (turning it sparse).  Linux only.  True on success."""
    libc = _load_libc()
    if libc is None:
        print("  [!] libc without fallocate: cannot punch holes here", file=sys.stderr)
        return False
    ret = libc.fallocate(
        ctypes.c_int(fd),
        ctypes.c_int(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE),
        ctypes.c_longlong(offset),
        ctypes.c_longlong(length),
    )
    if ret == 0:
        return True
    err = ctypes.get_errno()
    print(f"  [!] fallocate failed (errno={err}), range not punched", file=sys.stderr)
    return False


def punch_hole(path: Path, offset: int, length: int) -> bool:
    """Punch a range of the file.  Linux uses fallocate; other systems do
    nothing (and say so).  True when the range was released."""
    if os.name != "posix":
        print(f"  [!] {os.name}: hole-punch not supported.  The GGUF keeps its original size.", file=sys.stderr)
        return False
    try:
        # PUNCH_HOLE needs a writable fd (O_RDWR or O_WRONLY); O_RDONLY gives
        # EBADF.  fallocate does not read or write, it only releases the blocks.
        fd = os.open(str(path), os.O_RDWR)
        try:
            return punch_hole_linux(fd, offset, length)
        finally:
            os.close(fd)
    except OSError as e:
        print(f"  [!] cannot open {path} for punching: {e}", file=sys.stderr)
        return False


def check_punch_supported(gguf_path: Path) -> tuple[bool, str]:
    """Pre-flight: try to punch 4096 bytes at the very end of the shard.  If the
    filesystem says EOPNOTSUPP/ENOSYS, the whole run would write the ~35 GB of
    experts.bin WITHOUT freeing them from the GGUF, so fail loudly and early
    instead of letting the user believe 35 GB were saved.

    Returns (ok, message)."""
    if os.name != "posix":
        return True, ("no hole-punch on this system: the GGUF keeps its full size; "
                      "the extract only rebuilds experts.bin")
    libc = _load_libc()
    if libc is None:
        return False, ("libc has no fallocate: this filesystem/toolchain cannot hole-punch. "
                       "Use `tools/iq_pack.py --experts-bin` (which also copies the experts) "
                       "or run with --no-punch")
    size = gguf_path.stat().st_size
    probe_len = 4096
    if size < probe_len:
        return True, "shard smaller than one page: nothing to probe"
    fd = os.open(str(gguf_path), os.O_RDWR)
    try:
        ret = libc.fallocate(
            ctypes.c_int(fd),
            ctypes.c_int(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE),
            ctypes.c_longlong(size - probe_len),
            ctypes.c_longlong(probe_len),
        )
    finally:
        os.close(fd)
    if ret == 0:
        return True, "hole-punch supported"
    err = ctypes.get_errno()
    if err in (errno.EOPNOTSUPP, errno.ENOSYS, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)):
        return False, ("this filesystem does not support hole-punch: the GGUF would keep its "
                       "full size.  Use `tools/iq_pack.py --experts-bin` (which also copies "
                       "the experts) or run with --no-punch")
    return False, f"hole-punch probe failed (errno={err}): the GGUF would keep its full size"


def real_disk_usage(path: Path) -> int:
    """Bytes really used on disk (less than st_size when the file is sparse)."""
    try:
        if os.name != "posix":
            # Windows does not fill st_blocks: every file reports 0, so the
            # report would read "0 B".  There is no hole-punch there either.
            return os.path.getsize(path)
        return os.stat(path).st_blocks * 512   # st_blocks is always 512-byte units
    except OSError:
        return 0


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# ----------------------------------------------------------------------------
# Phases 1 + 2: extract one shard's experts into an intermediate file
# ----------------------------------------------------------------------------

def extract_from_shard(
    gguf_path: Path,
    out_bin: Path,
    punch: bool = True,
    progress_file: Path | None = None,
) -> int:
    """Extract the expert tensors of `gguf_path` into `out_bin`, punching the
    GGUF over the ranges that were extracted.

    0 on success, non-zero on error.

    Each layer has three expert tensors:
      - blk.{L}.ffn_gate_exps.weight
      - blk.{L}.ffn_up_exps.weight
      - blk.{L}.ffn_down_exps.weight

    In `out_bin` the order is: for each layer, the three tensors as
    [gate rows | up rows | down rows] per expert.

    OFFSETS ARE LOCAL TO THIS INTERMEDIATE FILE.  The per-layer `off` that
    expert_layout() returns is absolute inside the final experts.bin, so a
    shard whose first layer is not layer 0 would otherwise write its first
    layer at `off` and leave `off` zero bytes at the start of the intermediate.
    concat_intermediates() appends those files back to back, so the zeros end
    up inside experts.bin, its size becomes total + off, and
    FileExpertSource::open refuses it with "is not the pack this geometry came
    from" (src/core/expert_source.cpp:1455).
    """
    print(f"\n=== Extracting experts from {gguf_path.name} -> {out_bin.name} ===")
    print(f"  GGUF apparent size: {human_bytes(gguf_path.stat().st_size)}")
    print(f"  GGUF real size:     {human_bytes(real_disk_usage(gguf_path))}")

    model = Model(gguf_path)
    g = None
    for p, gf in zip(model.paths, model.files):
        if p.name == gguf_path.name:
            g = gf
            break
    if g is None:
        print(f"  [X] {gguf_path.name} is not one of the model's shards")
        return 1

    got = expert_layout(model, gguf_path)
    if isinstance(got, str):
        print(f"  [X] {got}")
        return 1
    layout, text, n_expert, total_bytes = got
    print(f"  model: {len(layout)} layers, {n_expert} experts per layer, experts.bin = {human_bytes(total_bytes)}")

    # Keep only the layers whose three tensors all live in THIS shard.
    my_layers = []
    for entry in layout:
        l, gt, dt, off, blob, ts = entry
        names_here = [t.name for t in ts if model.where[t.name][3].name == gguf_path.name]
        if len(names_here) == 3:
            my_layers.append(entry)
        elif names_here:
            print(f"  [X] layer {l}: its tensors are split across shards, which is not expected "
                  "for a two-shard IQ2_XS")
            print(f"      in this shard:  {names_here}")
            print(f"      in another one: {[t.name for t in ts if model.where[t.name][3].name != gguf_path.name]}")
            return 1
    print(f"  layers to extract from this shard: {len(my_layers)} of {len(layout)}")

    if not my_layers:
        print(f"  [!] no experts in {gguf_path.name}; skipping")
        return 0

    # Local offset of the first layer this shard contributes: everything the
    # shard writes goes at (absolute off - this base), so the intermediate has
    # no hole at its head.
    global_offset = my_layers[0][3]
    print(f"  first layer here: {my_layers[0][0]}; its blobs start at {human_bytes(global_offset)} "
          "in experts.bin, and at 0 in this intermediate")

    extracted_layers = set()
    if progress_file and progress_file.exists():
        try:
            prog = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prog = {}
        if prog.get("shard") == gguf_path.name:
            extracted_layers = set(prog.get("extracted_layers", []))
            print(f"  resuming: {len(extracted_layers)} layers already extracted")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    need_size = sum(entry[4] * n_expert for entry in my_layers)   # local, no global_offset
    open_mode = "r+b" if out_bin.exists() else "wb"
    with open(out_bin, open_mode) as fo:
        if open_mode == "wb":
            fo.truncate(need_size)
        for entry in my_layers:
            l, gt, dt, off, blob, ts = entry
            if l in extracted_layers:
                continue
            parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)
            if chunk.shape != (n_expert, blob):
                print(f"  [X] layer {l}: shape {chunk.shape} != ({n_expert}, {blob})")
                return 1
            fo.seek(off - global_offset)
            fo.write(chunk.tobytes())
            fo.flush()
            os.fsync(fo.fileno())
            if punch:
                for t in ts:
                    g_, t_, mm_, p_ = model.where[t.name]
                    if p_.name != gguf_path.name:
                        continue
                    tensor_off = g_.data_start + t_.offset
                    tensor_size = t_.expected_bytes()
                    if not punch_hole(p_, tensor_off, tensor_size):
                        print(f"  [!] could not punch {t.name} in {p_.name}", file=sys.stderr)
            extracted_layers.add(l)
            if progress_file:
                progress_file.write_text(json.dumps({
                    "shard": gguf_path.name,
                    "extracted_layers": sorted(extracted_layers),
                }, indent=2), encoding="utf-8")
            print(f"  layer {l:2d}: blob={human_bytes(blob)}, at {human_bytes(off - global_offset)} local, "
                  f"GGUF real now {human_bytes(real_disk_usage(gguf_path))}")

    print(f"  done: {len(extracted_layers)} layers extracted from {gguf_path.name}")
    print(f"  GGUF real size at the end: {human_bytes(real_disk_usage(gguf_path))}")
    return 0


# ----------------------------------------------------------------------------
# Phase 3: concatenate expert-01.bin + expert-02.bin -> experts.bin
# ----------------------------------------------------------------------------

def concat_intermediates(out_dir: Path, final_name: str = "experts.bin") -> int:
    """Concatenate the intermediates in order (expert-01.bin, expert-02.bin, ...)
    into `experts.bin`.

    No disk spike:
      - one part: a direct rename (instant, 0 extra bytes).
      - N parts: rename the first to experts.bin, then append the others one at
        a time, deleting each intermediate as soon as it is appended.  The peak
        is ONE intermediate, not all of them.
    """
    parts = []
    for p in out_dir.glob("expert-*.bin"):
        if p.name == final_name:
            continue
        if re.match(r"expert-\d+\.bin$", p.name):
            parts.append(p)
    parts.sort()
    if not parts:
        print(f"  [X] no expert-NN.bin files in {out_dir}")
        return 1

    out = out_dir / final_name
    print(f"\n=== Concatenating {len(parts)} parts -> {final_name} ===")
    for p in parts:
        print(f"  {p.name}: {human_bytes(p.stat().st_size)}")

    # Same convention as setup.py's free_gb(): shutil.disk_usage works on every
    # platform; os.statvfs does not exist on Windows at all.
    try:
        free = shutil.disk_usage(out_dir).free
        need = 0 if len(parts) == 1 else max(p.stat().st_size for p in parts[1:])
        print(f"  free: {human_bytes(free)}, needed (estimated): {human_bytes(need)}")
        if free < need + 1024 ** 3:
            print("  [!] LOW SPACE: if the filesystem fills up it may go read-only.")
            print("      Consider deleting temporary files before continuing.")
    except OSError:
        pass

    if len(parts) == 1:
        try:
            # Path.replace(), not Path.rename(): on Windows rename fails when the destination exists
            # (WinError 183) and phase 3 is run again after an interrupted one; os.replace overwrites on both.
            parts[0].replace(out)
            print(f"  [ok] {parts[0].name} -> {out.name} (rename, nothing duplicated)")
            print(f"  {out.name}: {human_bytes(out.stat().st_size)} total")
            return 0
        except OSError as e:
            print(f"  [X] rename failed: {e}")
            print("      If the filesystem is read-only, free some space and start again.")
            return 1

    try:
        parts[0].replace(out)
        print(f"  [ok] {parts[0].name} -> {out.name} (base rename)")
    except OSError as e:
        print(f"  [X] cannot rename {parts[0].name}: {e}")
        return 1

    chunk = 64 * 1024 * 1024   # 64 MiB per iteration
    with open(out, "r+b") as fo:
        fo.seek(0, os.SEEK_END)
        for p in parts[1:]:
            written = 0
            try:
                with open(p, "rb") as fi:
                    while True:
                        buf = fi.read(chunk)
                        if not buf:
                            break
                        fo.write(buf)
                        written += len(buf)
                fo.flush()
                os.fsync(fo.fileno())
                print(f"  {p.name} -> {out.name}: {human_bytes(written)} appended")
                p.unlink()
                print(f"  {p.name} deleted")
            except OSError as e:
                print(f"  [X] error while appending {p.name}: {e}")
                print(f"      What is already in {out.name} is kept.")
                return 1

    print(f"  {out.name}: {human_bytes(out.stat().st_size)} total")
    return 0


# ----------------------------------------------------------------------------
# Phase 4: cleanup + sidecar + .done marks
# ----------------------------------------------------------------------------

def finalize(gguf_path: Path, out_dir: Path) -> int:
    """Write the sidecar experts.bin.src.json and update the GGUFs' .done marks
    so the setup neither re-downloads them nor re-hashes them (the holes read as
    zeros, so their SHA-256 cannot match the published one any more)."""
    print("\n=== Finalizing ===")
    experts_bin = out_dir / "experts.bin"
    if not experts_bin.exists():
        print(f"  [X] {experts_bin} does not exist")
        return 1
    print(f"  experts.bin: {human_bytes(experts_bin.stat().st_size)}")

    model = Model(gguf_path)
    got = expert_layout(model, gguf_path)
    if isinstance(got, str):
        print(f"  [X] {got}")
        return 1
    layout, text, n_expert, total = got

    # Exactly the sidecar iq_pack writes: the engine compares it (names, sizes
    # and the native_experts.txt hash) to decide whether an experts.bin belongs
    # to this model.  Anything else is "not the pack this geometry came from".
    sidecar = experts_source(model, text, total)
    sidecar_path = out_dir / "experts.bin.src.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"  sidecar: {sidecar_path}")

    # REPO_PINS lives in setup.py (not in iq_pack.py).  Reading it gives the
    # pinned SHA to write into each .done mark so verify_sha256 short-circuits.
    pinned_shas = {}
    repo_root = IQ_PACK_PATH.parent.parent      # tools/ -> repo root
    setup_path = repo_root / "setup.py"
    if setup_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("strata_setup", setup_path)
            setup_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(setup_mod)
            for repo_files in getattr(setup_mod, "REPO_PINS", {}).values():
                if isinstance(repo_files, dict):
                    for fname, info in repo_files.items():
                        if isinstance(info, tuple) and len(info) == 2:
                            size, sha = info
                            pinned_shas[fname] = (size, sha)
        except Exception as e:
            print(f"  [!] could not import REPO_PINS from setup.py: {e}")
            print("      Continuing without pinned SHAs (setup.py may re-hash the GGUFs).")

    for p in model.paths:
        done = p.with_name(p.name + ".done")
        old = done.read_text(encoding="utf-8", errors="replace") if done.exists() else ""

        kept = [line for line in old.splitlines()
                if line.strip() and not line.startswith("sha256 ") and not line.startswith("trimmed:")]

        if p.name in pinned_shas:
            _, sha = pinned_shas[p.name]
            kept.append(f"sha256 {sha}")
            print(f"  {p.name}: pinned SHA found -> verify_sha256 will short-circuit")
        else:
            print(f"  [!] {p.name}: no pinned SHA in REPO_PINS")
            print("      setup.py may try to re-hash it and fail; pass --gguf-dir or set STRATA_SKIP_SHA256=1")

        kept.append("trimmed: gguf punched, experts.bin built")
        done.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"  {done.name}: updated")

    for f in out_dir.glob("expert-*.bin.progress.json"):
        f.unlink()
    print("\nDone. experts.bin is built and the GGUFs are punched.")
    print(f"  disk really used: {human_bytes(real_disk_usage(experts_bin))} (experts.bin)")
    for p in model.paths:
        print(f"  {p.name}: {human_bytes(real_disk_usage(p))} of {human_bytes(p.stat().st_size)} apparent")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build experts.bin by extracting the experts from the GGUF and punching the source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gguf", required=True, help="path to shard 1 of the model's GGUF")
    ap.add_argument("--out", required=True, help="pack directory (where experts.bin is written)")
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "all"],
                    help="phase to run (1: shard 1, 2: shard 2, 3: concat, 4: finalize, all: 1+2+3+4)")
    ap.add_argument("--no-punch", action="store_true",
                    help="do not punch the GGUF (extract only, as plain iq_pack.py does)")
    args = ap.parse_args()

    gguf = Path(args.gguf).resolve()
    out_dir = Path(args.out).resolve()
    punch = not args.no_punch

    if not gguf.exists():
        print(f"[X] does not exist: {gguf}")
        return 1

    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", gguf.name)
    if not m:
        print(f"[X] {gguf.name} does not follow the -NNNNN-of-NNNNN.gguf pattern")
        return 1
    total_shards = int(m.group(2))
    shards = [gguf.with_name(gguf.name[:m.start()] + f"-{i:05d}-of-{total_shards:05d}.gguf")
              for i in range(1, total_shards + 1)]
    missing = [s for s in shards if not s.exists()]
    if missing:
        print(f"[X] missing shards: {[s.name for s in missing]}")
        return 1
    print(f"Model: {total_shards} shards")
    for s in shards:
        print(f"  {s.name}: {human_bytes(s.stat().st_size)}")

    # The pre-flight runs once, before anything is written: without hole-punch
    # the extract frees nothing and the user would believe 35 GB were saved.
    if punch:
        ok, why = check_punch_supported(shards[0])
        if not ok:
            print(f"[X] {why}")
            return 1
        print(f"  punch: {why}")
    else:
        print("  punch: disabled (--no-punch); the GGUF keeps its full size")

    out_dir.mkdir(parents=True, exist_ok=True)

    phases = ["1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase in ("1", "2"):
            shard_idx = int(phase) - 1
            if shard_idx >= len(shards):
                print(f"\n[!] phase {phase}: there is no shard {phase} (only {len(shards)})")
                continue
            shard = shards[shard_idx]
            inter = out_dir / f"expert-0{shard_idx + 1}.bin"
            prog = out_dir / f"expert-0{shard_idx + 1}.bin.progress.json"
            rc = extract_from_shard(shard, inter, punch=punch, progress_file=prog)
            if rc != 0:
                return rc
        elif phase == "3":
            rc = concat_intermediates(out_dir)
            if rc != 0:
                return rc
        elif phase == "4":
            rc = finalize(gguf, out_dir)
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
