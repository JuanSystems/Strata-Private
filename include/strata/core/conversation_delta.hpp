// Delta journal codec for the disk-tier spill store: the record that extends (or rewinds to a
// checkpoint of) a base conv-<serial>.bin without rewriting it. See plan-diario-deltas-spill.md.
//
// A conversation on disk is the pair (base .bin, journal .dlt): the base is a v1 session file
// (session_file_write, atomic temp+rename), the journal is an append-only sequence of records,
// one per park that found the base as its delta base. Applying every record to the base yields
// byte-exactly the image a full write of the newest park would have stored.
//
// v2 (layer split): a record covers EVERY image of the spill — the main conv-<serial>.bin AND
// its conv-<serial>.stage<k>.bin files, one SECTION per image inside the same record. One
// append + one hash makes the record atomic across images: a crash mid-append leaves main and
// every stage at the same record boundary (per-file journals would strand the main one record
// ahead of a stage). The ids and the image list are stored ONCE (a capture gives every image
// the same live ids/imgs — spill_live_to_disk builds them from one `live`); the running state,
// the checkpoints and the K/V are per-section (each stage carries its own carve's portion).
//
// v3 (light records + state sidecar): every v2 section re-sent the image's COMPLETE live
// running state and its checkpoint, so the record was O(model state) per turn — the journal
// grew with the state × turns, the very amplification the parent/children design avoids, and
// the writer copied the whole state into its front buffer (an O(state) RAM peak). A v3 record
// carries ONLY the tables and the new K/V; the running state and the checkpoint of EVERY image
// live in ONE sidecar, conv-<serial>.rst, replaced atomically (temp + fsync + rename) by every
// append. Only the newest state matters — the fold applies the records for the ids/dims/K/V and
// loads the state ONCE, from the sidecar of the last record it applied — so one sidecar per
// journal is enough, and the record drops to O(turn tokens). The sidecar binds to its record
// through (seq, record_hash): seq is the record's 1-based index in the journal and record_hash
// the hash the record's trailer carries, so a sidecar from another generation is detected.
//
// Record layout (little-endian, fixed header 129 B + shared region + per-image sections +
// K/V payload + trailer 20 B):
//   u8[4]  "SDLT"          u32 version = 2 | 3
//   u64 record_bytes       total size of this record, header..trailer included
//   u64 applies_to_len     live ids length of the state this record was computed against (the
//                          chain rule: the first record must match the base .bin's live length,
//                          each later one the previous record's new_tokens)
//   u64 keep_tokens        how many leading ids of that state are kept (the growth case keeps
//                          them all; the re-render case keeps the deepest checkpoint's cut)
//   u64 new_tokens         live ids length after applying
//   u64 small_bytes        size of the shared region that follows the header
//   u64 geometry_hash      FNV-1a over geometry[18] + layer_lo + layer_hi of the MAIN image
//                          (the quick reject; every section repeats its own)
//   u8  cvec               must equal the base's
//   u32 n_images           1 + stage_images.size() (main first, then the stages in order)
//   u32 n_layers           the main image's kv.size()
//   u64 n_imgs             live image count after applying
//   u64 ckpt_tokens        the MAIN image's deepest checkpoint ids size after applying (0: none)
//   u64 state_bytes[5]     v2: the MAIN image's live gdn/ple/tails/dead/block_pos sizes.
//                          v3: state_bytes[0] is the record's seq (its 1-based index within the
//                          journal); state_bytes[1..4] are zero. The state itself is in the .rst.
//   u64 sections_bytes     size of every section's front matter combined
//   shared region:
//   i32[new_tokens-keep_tokens]                ids suffix (applies to every image)
//   (i64 start, u64 hash) x n_imgs             live images, the complete list
//   per image section (0 = the main, then each stage):
//   v2:
//   u32 n_layers_i                             image.kv.size()
//   u64 geometry_hash_i                        this image's own echo (its layer carve)
//   u64 ckpt_tokens_i                          this image's deepest checkpoint after applying
//   u64 state_bytes_i[5]                       this image's live running state sizes
//   u8[state_bytes_i[0..4]]                    this image's live running state, complete
//   u32 n_checkpoints_i (0 or 1: the ONE checkpoint the record keeps — spill_resume_checkpoint's
//       selection, the chain's DEEPEST, ignoring `pinned`: the turn boundary a re-rendering
//       client (zoocode/roocode) still contains, which is the resume point the engine's
//       post-restore pass takes; the pinned system prompt is persisted apart as a
//       pfx-<serial>.bin shared-prefix snapshot and must not steal the slot, or the fold
//       replaces the chain with it and every delta-restored resume falls to the system
//       prompt. Readers accept any count, records written by older builds may carry more) + per
//       checkpoint: u64 n_ids; i32 x n_ids; u64 n_cimgs; (i64,u64) x
//       n_cimgs; u64 sz[5]; u8 x sz[0..4]; u64 used
//   v3:
//   u32 n_layers_i                             image.kv.size()
//   u64 geometry_hash_i                        this image's own echo (its layer carve)
//   K/V table (so a scanner can size the record without reading the payload):
//   per layer: 7 x i64  format, cells, heads, head_dim, page_size, pooled_rows, idx_dim
//   per layer x 5 parts: u64 at_bytes; u64 n_bytes
//   K/V payload: per layer x 5 parts, n_bytes of raw bytes (k, v, k_scale, v_scale, pooled)
//   trailer: u64 hash (FNV-1a over everything above); u64 record_bytes; u8[4] "EDLT"
//
// Sidecar conv-<serial>.rst layout (little-endian; the state of EVERY image of the spill —
// the main first, then the stages — for the LAST record the journal holds):
//   u8[4] "SRST"   u32 version = 2   u64 seq   u64 record_hash   u32 n_images   u32 reserved (32 B)
//   per image:
//   u64 geometry_hash                  this image's own echo (misalignment reject)
//   u64 state_sz[5] + the 5 live state byte arrays (gdn, ple, tails, dead, block_pos)
//   u32 n_ckpt (0..4) and, per checkpoint, spill_resume_checkpoints' SELECTION (the resume
//       set: the chain's deepest of any kind, its two deepest `turn` boundaries and its
//       deepest `periodic` — deduplicated by length, ascending):
//       u64 n_ids; i32 x n_ids; u64 n_cimgs; (i64,u64) x n_cimgs;
//       u64 sz[5]; u8 x sz[0..4]; u64 used; u8 kind (CheckpointKind)
//   trailer: u64 fnv (everything above); u64 total_bytes; u8[4] "ERST"               (20 B)
//
// v1 (every sidecar a pre-set build wrote) carries ONE checkpoint per image (the chain's
// deepest) and no kind byte; the probe accepts both generations, a v1 checkpoint loads as
// `periodic`, and a v2 kind byte an old reader cannot parse makes it reject the whole sidecar
// (fail-soft: the journal's v3 records read as torn tails and the base restores alone).
// The set is what a restore can offer as resume points — see spill_resume_checkpoints
// (conversation_file.hpp) for why the deepest alone loses every restart resume of a
// --tail-role-token client.
//
// The at_bytes boundary mirrors conversation_kv_save's reuse rule: completed pages (and
// completed indexer rows) strictly below the kept tokens are byte-identical between the base
// and the newer image, so the record only re-sends from the last completed page of the kept
// prefix on — computed PER IMAGE (each stage's own page_size and buffers). The pooled boundary
// needs idx_block, which no image carries; it is derived from the NEW image's own pooled_rows
// as an UPPER bound (an over-estimate only ever re-sends more bytes, never keeps bytes that
// differ), so correctness never depends on the derivation.
//
// All reader entry points are single-pass, never throw, and stop at the last fully valid
// record: a torn or unchainable tail is reported so the caller can truncate the file, and a
// journal whose first record does not chain onto the base (a compaction that crashed after
// publishing the new .bin but before deleting the journal, or a .bin replaced under a reused
// serial) is reported with zero applied records so the caller can drop it whole. A record
// whose n_images does not match the images the reader holds (a stage file lost or replaced)
// is a torn tail the same way. A v3 record additionally requires the sidecar to chain: the
// probe must succeed, the record's seq must be the next index, and the sidecar's seq must
// cover it — a v3 record past the sidecar's coverage is a torn tail (the crash window between
// the record append and the sidecar's rename), and after the loop the sidecar's (seq,
// record_hash) must land exactly on the last APPLIED record or the fold is `failed` (the
// images are suspect: the state and the K/V would disagree). A journal where NO v3 record
// applied while a sidecar sits beside it reports `state_stale` so the caller can delete the
// orphan (a v2 record appended after v3 ones does not: the earlier records still chain
// through the sidecar, and a tail tear could yet land the fold on one of them).
//
// The codec writes v2 when the plan's seq is 0 (the store's STRATA_SPILL_JOURNAL_V3=0 escape,
// and the shape every pre-v3 test expects), so one .dlt can mix v2 and v3 records; a build
// that only knows v2 rejects a v3 header as a torn tail (never misreads it).
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

/// One image's slice of the plan (the main image or one stage's).
struct DeltaImagePlan {
    uint64_t geometry_hash = 0;                      // this image's own echo
    size_t n_layers = 0;                             // image.kv.size()
    std::vector<std::array<DeltaPart, 5>> parts;     // [layer][k/v part]
};

/// The writer's plan: which bytes of each image the record must carry, given that the first
/// `keep_tokens` ids of the chaining state are identical (see the boundary rule above).
struct DeltaPlan {
    uint64_t applies_to_len = 0;   // live ids length of the state this record chains from
    uint64_t keep_tokens = 0;
    uint64_t new_tokens = 0;
    uint64_t geometry_hash = 0;    // the main image's (the header's quick-reject echo)
    bool cvec = true;
    uint32_t version = 0;          // kDeltaVersionV2 when seq == 0, kDeltaVersion otherwise
    uint64_t seq = 0;              // the record's 1-based index within the journal (0: v2)
    std::vector<DeltaImagePlan> images;   // [0] = the main image, [1..] = its stage images
    uint64_t record_bytes = 0;            // exact on-disk size of the record
};

/// Compute the plan for `image` (its stage_images included) against a chaining state of
/// `base_len` live ids, keeping `keep_tokens` of them. `seq` is the record's 1-based index
/// within the journal: seq > 0 plans a v3 record (no state in the record, the sidecar
/// carries it), seq == 0 plans the v2 shape (the state rides in the record, as before).
/// False (no delta possible) when any image's K/V geometry is not self-consistent with the
/// boundary rule, or when the images do not share one live state (ids and image list) —
/// the caller then writes the full .bin as today.
bool conversation_delta_plan(DeltaPlan& plan, const SavedConversation& image, uint64_t base_len,
                             uint64_t keep_tokens, uint64_t seq = 0);

/// Append one record for `image` (main + stage images) to the journal at `path` (opened in
/// append mode; the caller has already verified the file's size matches the manifest's
/// journal_bytes). Streams the K/V payloads straight from the images' buffers, fsyncs the
/// file, and on success reports the record's exact byte count. When `record_hash` is not
/// null it receives the record's trailer hash — the value the state sidecar binds to (see
/// conversation_delta_state_write). False leaves a possibly partial record behind — the
/// reader's hash check truncates it, and the manifest's size check falls the next write back
/// to the full path.
bool conversation_delta_append(const std::string& path, const SavedConversation& image, const DeltaPlan& plan,
                               uint64_t& record_bytes, uint64_t* record_hash = nullptr);

/// The exact on-disk size of the state sidecar for `image` (main + stage images): the live
/// running state and the RESUME SET of every image (spill_resume_checkpoints' selection,
/// each checkpoint with its kind byte), one file. The store's budget accounting needs it
/// BEFORE the write (the sidecar replaces its previous self, so the net cost is the
/// difference).
uint64_t conversation_delta_state_bytes(const SavedConversation& image);

/// Write the state sidecar for `image` to `path` (conv-<serial>.rst): the live state and the
/// resume-set checkpoints of EVERY image, stamped with `seq` (the record's 1-based journal
/// index) and `record_hash` (the trailer hash conversation_delta_append reported for that
/// record) so the reader can bind the sidecar to exactly one record. Streams straight from the
/// images' state vectors (no O(state) front buffer), then publishes atomically: temp + fflush
/// + fsync + rename + directory fsync — the rename IS the commit. On any failure the temp is
/// removed and `path` keeps its previous contents. `bytes` receives the file's exact size.
bool conversation_delta_state_write(const std::string& path, const SavedConversation& image, uint64_t seq,
                                    uint64_t record_hash, uint64_t& bytes);

/// One line of the per-image record breakdown (MiB units) for STRATA_SPILL_DIAG: the live
/// state, the selected checkpoint, the K/V the plan sends vs what the window expects, and
/// the full K/V — the Phase 0 instrument that says where the record's bytes are. Empty when
/// the plan covers no image.
std::string conversation_delta_describe(const SavedConversation& image, const DeltaPlan& plan);

/// What a journal did to the images it was applied to.
struct DeltaFold {
    size_t applied = 0;            // records applied (0 with a non-empty file: drop it whole)
    uint64_t valid_end = 0;        // file offset just past the last applied record
    uint64_t journal_bytes = 0;    // == valid_end; the manifest's journal accounting
    uint32_t records = 0;          // same as applied, the compaction trigger's unit
    bool had_journal = false;      // the file existed
    bool torn = false;             // the file is longer than valid_end (truncate it)
    bool dropped = false;          // nothing chains: the whole journal is stale for this base
    bool failed = false;           // an I/O error mid-apply, or a v3 sidecar that does not
                                   // land on the last applied record: the image is suspect,
                                   // discard it
    bool state_stale = false;      // NO v3 record applied while a sidecar sits beside the
                                   // journal: it describes nothing reachable — delete it. (A
                                   // v2 record appended after v3 ones does not trigger this:
                                   // the earlier records still chain through the sidecar.)
};

/// Apply the journal at `path` to `image` (the base .bin already parsed into it, its
/// stage_images filled with the parsed conv-<serial>.stage<k>.bin files). Every record is
/// FULLY validated before anything is mutated: the chain rule (applies_to_len), the geometry
/// echoes (header + every section's), the image count, the per-layer invariants, and the
/// record's hash over a streaming read of all its bytes; only then are the ids, the running
/// states, the checkpoints and the K/V payloads applied (the payload is re-read from its
/// offset — the file cannot move under us, the writer is drained first). A torn or corrupted
/// record therefore leaves every image at the LAST GOOD record's state, and the fold reports
/// where the valid chain ends so the caller can truncate. Never throws; an I/O error during
/// the apply (or a v3 sidecar that does not chain: see the header comment) marks the fold
/// `failed` — the images are then suspect and the caller must discard them (as a corrupt .bin
/// today). `state_path` is the conv-<serial>.rst sidecar the v3 records' state lives in; an
/// empty path (or an unreadable sidecar) makes every v3 record a torn tail, so a journal that
/// opens with one is dropped whole and the base restores alone.
DeltaFold conversation_delta_apply(const std::string& path, SavedConversation& image,
                                   const std::string& state_path = std::string());

/// The journal's magic values, exposed for tests.
inline constexpr char kDeltaMagic[4] = {'S', 'D', 'L', 'T'};
inline constexpr char kDeltaEndMagic[4] = {'E', 'D', 'L', 'T'};
/// The record format this build writes when the store's v3 journal is on. v2 is the
/// pre-sidecar shape (state + checkpoint inside every record); one .dlt may mix both.
inline constexpr uint32_t kDeltaVersionV2 = 2;
inline constexpr uint32_t kDeltaVersion = 3;
/// The state sidecar's magic values (conv-<serial>.rst).
inline constexpr char kStateMagic[4] = {'S', 'R', 'S', 'T'};
inline constexpr char kStateEndMagic[4] = {'E', 'R', 'S', 'T'};
/// v2: the sidecar carries spill_resume_checkpoints' SET per image (up to 4 checkpoints:
/// the deepest, the two deepest turn boundaries, the deepest periodic), each with a `kind`
/// byte after `used`. v1 (one checkpoint, no kind) is still accepted on read — every sidecar
/// a pre-set build wrote restores exactly as before.
inline constexpr uint32_t kStateVersion = 2;

} // namespace strata::core
