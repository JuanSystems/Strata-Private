// Delta journal codec for the disk-tier spill store: the record that extends (or rewinds to a
// checkpoint of) a base conv-<serial>.bin without rewriting it. See plan-diario-deltas-spill.md.
//
// A conversation on disk is the pair (base .bin, journal .dlt): the base is a v1 session file
// (session_file_write, atomic temp+rename), the journal is an append-only sequence of records,
// one per park that found the base as its delta base. Applying every record to the base yields
// byte-exactly the image a full write of the newest park would have stored.
//
// Record layout (little-endian, fixed header 117 B + small region + K/V table + K/V payload +
// trailer 20 B):
//   u8[4]  "SDLT"          u32 version = 1
//   u64 record_bytes       total size of this record, header..trailer included
//   u64 applies_to_len     live ids length of the state this record was computed against (the
//                          chain rule: the first record must match the base .bin's live length,
//                          each later one the previous record's new_tokens)
//   u64 keep_tokens        how many leading ids of that state are kept (the growth case keeps
//                          them all; the re-render case keeps the deepest checkpoint's cut)
//   u64 new_tokens         live ids length after applying
//   u64 small_bytes        size of the small region that follows the header
//   u64 geometry_hash      FNV-1a over geometry[18] + layer_lo + layer_hi (an echo: a mismatch
//                          refuses the record before any payload moves)
//   u8  cvec               must equal the base's
//   u32 n_layers           image.kv.size()
//   u64 n_imgs             live image count after applying
//   u64 ckpt_tokens        deepest checkpoint's ids size after applying (0: no checkpoint)
//   u64 state_bytes[5]     live gdn/ple/tails/dead/block_pos sizes
//   small region:
//   i32[new_tokens-keep_tokens]                ids suffix
//   (i64 start, u64 hash) x n_imgs             live images, the complete list
//   u8[state_bytes[0..4]]                      live running state, complete
//   u32 n_checkpoints
//   per checkpoint: u64 n_ids; i32 x n_ids; u64 n_cimgs; (i64,u64) x n_cimgs; u64 sz[5];
//                   u8 x sz[0..4]; u64 used
//   K/V table (so a scanner can size the record without reading the payload):
//   per layer: 7 x i64  format, cells, heads, head_dim, page_size, pooled_rows, idx_dim
//   per layer x 5 parts: u64 at_bytes; u64 n_bytes
//   K/V payload: per layer x 5 parts, n_bytes of raw bytes (k, v, k_scale, v_scale, pooled)
//   trailer: u64 hash (FNV-1a over everything above); u64 record_bytes; u8[4] "EDLT"
//
// The at_bytes boundary mirrors conversation_kv_save's reuse rule: completed pages (and
// completed indexer rows) strictly below the kept tokens are byte-identical between the base
// and the newer image, so the record only re-sends from the last completed page of the kept
// prefix on. The pooled boundary needs idx_block, which no image carries; it is derived from
// the NEW image's own pooled_rows as an UPPER bound (an over-estimate only ever re-sends more
// bytes, never keeps bytes that differ), so correctness never depends on the derivation.
//
// All reader entry points are single-pass, never throw, and stop at the last fully valid
// record: a torn or unchainable tail is reported so the caller can truncate the file, and a
// journal whose first record does not chain onto the base (a compaction that crashed after
// publishing the new .bin but before deleting the journal, or a .bin replaced under a reused
// serial) is reported with zero applied records so the caller can drop it whole.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// FNV-1a over the geometry array and the layer carve — the anti-misalignment echo a record
/// carries and the reader compares against the base image's own values.
uint64_t conversation_delta_geometry_hash(const std::array<int64_t, 18>& geometry, int64_t layer_lo,
                                          int64_t layer_hi);

/// One K/V part's byte span inside a record: the bytes [at, at+n) replace the same span of
/// the chaining state's buffer (at is always a completed-page boundary of the kept prefix,
/// so everything below it is byte-identical and stays in the base).
struct DeltaPart {
    uint64_t at = 0, n = 0;
};

/// The writer's plan: which bytes of the new image the record must carry, given that the first
/// `keep_tokens` ids of the chaining state are identical (see the boundary rule above).
struct DeltaPlan {
    uint64_t applies_to_len = 0;   // live ids length of the state this record chains from
    uint64_t keep_tokens = 0;
    uint64_t new_tokens = 0;
    uint64_t geometry_hash = 0;
    bool cvec = true;
    size_t n_layers = 0;
    std::vector<std::array<DeltaPart, 5>> parts;   // [layer][k/v part]
    uint64_t record_bytes = 0;                    // exact on-disk size of the record
};

/// Compute the plan for `image` against a chaining state of `base_len` live ids, keeping
/// `keep_tokens` of them. False (no delta possible) when the image's K/V geometry is not
/// self-consistent with the boundary rule — the caller then writes the full .bin as today.
bool conversation_delta_plan(DeltaPlan& plan, const SavedConversation& image, uint64_t base_len,
                             uint64_t keep_tokens);

/// Append one record for `image` to the journal at `path` (opened in append mode; the caller
/// has already verified the file's size matches the manifest's journal_bytes). Streams the K/V
/// payloads straight from the image's buffers, fsyncs the file, and on success reports the
/// record's exact byte count. False leaves a possibly partial record behind — the reader's
/// hash check truncates it, and the manifest's size check falls the next write back to the
/// full path.
bool conversation_delta_append(const std::string& path, const SavedConversation& image, const DeltaPlan& plan,
                               uint64_t& record_bytes);

/// What a journal did to the image it was applied to.
struct DeltaFold {
    size_t applied = 0;            // records applied (0 with a non-empty file: drop it whole)
    uint64_t valid_end = 0;        // file offset just past the last applied record
    uint64_t journal_bytes = 0;    // == valid_end; the manifest's journal accounting
    uint32_t records = 0;          // same as applied, the compaction trigger's unit
    bool had_journal = false;      // the file existed
    bool torn = false;             // the file is longer than valid_end (truncate it)
    bool dropped = false;          // nothing chains: the whole journal is stale for this base
    bool failed = false;           // an I/O error mid-apply: the image is suspect, discard it
};

/// Apply the journal at `path` to `image` (the base .bin already parsed into it). Every record
/// is FULLY validated before anything is mutated: the chain rule (applies_to_len), the geometry
/// echo, the per-layer invariants, and the record's hash over a streaming read of all its
/// bytes; only then are the ids, the running state and the K/V payload applied (the payload is
/// re-read from its offset — the file cannot move under us, the writer is drained first). A
/// torn or corrupted record therefore leaves the image at the LAST GOOD record's state, and
/// the fold reports where the valid chain ends so the caller can truncate. Never throws; an
/// I/O error during the apply marks the fold `failed` — the image is then suspect and the
/// caller must discard it (as a corrupt .bin today).
DeltaFold conversation_delta_apply(const std::string& path, SavedConversation& image);

/// The journal's magic values, exposed for tests.
inline constexpr char kDeltaMagic[4] = {'S', 'D', 'L', 'T'};
inline constexpr char kDeltaEndMagic[4] = {'E', 'D', 'L', 'T'};
inline constexpr uint32_t kDeltaVersion = 1;

} // namespace strata::core
