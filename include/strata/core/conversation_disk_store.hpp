// Disk-tier spill for parked conversations: when ConversationCache evicts an entry, this store
// writes the SavedConversation to a file (the same v1 session-file format SAVE uses), so the
// next turn restores it in ~1-2 s instead of re-processing 72K tokens (~210 s). The
// conversation never disappears while there is disk.
//
// Pure host code: no CUDA. Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; the engine links
// it into strata_engine when STRATA_SPILL is enabled (Phase 1b). A nullptr spill_ pointer in
// ConversationCache reproduces today's behavior exactly (no spill, conversation dropped on
// eviction) for A/B tests and regression safety.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_delta.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::core {

/// A manifest entry for a spilled conversation. Holds the cheap prefix-match data in RAM
/// (ids + imgs + geometry + layer carve + cvec flag); the heavy payload (K/V + running
/// state, ~1.4 GiB at 72K tokens) lives on disk until a request matches the prefix.
struct DiskEntry {
    uint64_t serial = 0;                              // unique, mirrors ConversationCache's serials_
    std::string path;                                  // <spill_dir>/conv-<serial>.bin
    // The checkpoint prefix the next turn's `best()` matches against — `live` and the deepest
    // checkpoint of a parked conversation (the rest are discarded by session_checkpoints_to_save).
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    // The deepest checkpoint's ids/imgs (the turn boundary — what the next request's prompt
    // actually matches, since the client re-renders the history without the assistant's
    // thinking tokens). When the live ids don't match the prompt (because of reasoning tokens
    // the client strips), best() falls back to matching against ckpt_ids.
    std::vector<int32_t> ckpt_ids;
    std::vector<ConversationImageKey> ckpt_imgs;
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
    uint64_t file_bytes = 0;                           // for LRU byte-budget accounting
    uint64_t used = 0;                                 // LRU stamp (monotonic)
    // A shared-prefix snapshot (the pinned system prompt's K/V, written as pfx-<serial>.bin):
    // a cross-conversation asset, not a branch of one conversation. Set by put()/put_async()
    // (the caller says which kind this image is) and re-derived from the file name on reload.
    // The ids-based prefix_snapshot() predicate (live ids == the deepest checkpoint's ids)
    // still recognizes prefix-shaped files written by older builds under the conv- name, so
    // every protection keeps BOTH signals: e.prefix || prefix_snapshot(e).
    bool prefix = false;
    // ── Delta journal (STRATA_SPILL_JOURNAL=1) ── the .dlt appended beside the .bin: its
    // bytes (counted in file_bytes too) and its record count (the compaction trigger). A
    // conversation on disk is then the pair (base .bin + journal .dlt): take() applies the
    // journal to the base, and a park that finds this entry as its delta base appends one
    // record instead of rewriting the whole .bin.
    uint64_t journal_bytes = 0;
    uint32_t journal_records = 0;
    // ── Layer-split stages (conv-<serial>.stage<k>.bin) ── the count and each stage file's
    // layer carve, so find_delta_base can match a new capture's stage images against this
    // entry's: a record carries one section per image, and a stage count or carve mismatch
    // cannot chain. An entry whose stage files exist but whose carve lists are empty (a
    // pre-v4 .idx that has not been regenerated yet) is never used as a delta base — the
    // full write replaces it once and fills them in.
    uint32_t n_stages = 0;
    std::vector<int64_t> stage_layer_lo;
    std::vector<int64_t> stage_layer_hi;
};

/// Disk-tier spill store. RAM manifest + payload-on-disk. Take = remove: when a conversation
/// comes back from disk, its file is deleted (no orphan copies of what's already in RAM).
/// All operations are fail-soft: a full or corrupt disk degrades to today's behavior, never
/// worse.
///
/// Phase 2 (async writes): the request path CAPTURES the snapshot synchronously (the GPU→host
/// copy must happen before the next request rewrites the session) but the disk write — hundreds
/// of MiB on a slow disk, tens of seconds on Kaggle-class hardware — runs on a background
/// writer thread. Queue depth is 1 (plus the image in the writer's hands): each image is
/// hundreds of MiB of host RAM, and the back-to-back parks that overflow the queue are the
/// retry loop's earlier capture of the SAME conversation, superseded by the newer one anyway.
/// Readers (take/best/size/…) and the sync put() first DRAIN the queue via flush_pending(), so
/// after any public call returns, the manifest is exactly what is on disk and no two threads
/// touch entries_ — the writer only mutates it between flushes, while the main thread is
/// blocked inside flush_pending().
class ConversationDiskStore {
public:
    /// `dir` is created (with parents) if missing. `budget_bytes = 0` disables the store
    /// (put() returns false; best() returns no match). `identity = nullptr` skips the
    /// model/config fingerprint check on read (test fixture only).
    ConversationDiskStore(std::string dir, uint64_t budget_bytes,
                          const SessionFileIdentity* identity = nullptr);
    ~ConversationDiskStore();

    // The store is held by unique_ptr everywhere (make_disk_store); it is NOT movable since
    // Phase 2 — the background writer thread, its mutex and its condition_variable are
    // non-movable, and no call site ever moved the store by value.
    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;

    bool enabled() const { return budget_ != 0; }
    const std::string& dir() const { return dir_; }

    // ---- Park path (the new evict_oldest calls this) ----
    // Writes `image` (captured in RAM by ConversationCache::take) to <dir>/conv-<serial>.bin
    // using session_file_write (atomic temp+rename+fsync). On a full disk or I/O error returns
    // false: the caller then drops the image as today (the spill only ever improves the
    // current state). `serial` is the cache's serial for this entry — the file name is
    // derived from it, so two parked conversations never collide.
    // Sync version: drains any in-flight background write first, then writes on the CALLING
    // thread. Tests and rare paths use this; the engine's park path uses put_async().
    // `prefix = true` marks the image as a shared-prefix snapshot (pinned system prompt K/V):
    // it is written as pfx-<serial>.bin, survives the same-conversation cleanups and the
    // reload dedupe, is never dropped in favor of a queued conversation image, and is NOT
    // consumed by take() (many conversations branch from it).
    bool put(SavedConversation&& image, uint64_t serial, const SessionFileIdentity& id,
             const SessionWriteOptions& options = {}, bool prefix = false);

    // ---- Async park path (the engine's park_current_body uses this) ----
    // Enqueues the image for the background writer and returns immediately (the request's
    // prompt read is not stalled by the disk write). Returns false only when the store is
    // disabled or the enqueue fails (caller logs and drops the image, as with sync put).
    // After a successful write the writer also runs the superseded cleanup (same-branch
    // older spills are deleted), so a retry loop does not pile up conv-<serial>.bin files.
    // `prefix = true`: a shared-prefix snapshot (see put()); the queue holds ONE conversation
    // image plus ONE prefix image, and a prefix is never dropped in favor of a conversation
    // image (its loss is unrecoverable this run; a dropped conversation capture is usually
    // the superseded older capture of the same conversation anyway).
    bool put_async(SavedConversation&& image, uint64_t serial, const SessionFileIdentity& id,
                   const SessionWriteOptions& options = {}, bool prefix = false);

    // Wait until no background write is pending or in flight. After this returns, the
    // manifest reflects every put_async() the store has accepted. Const so best()/size()
    // can drain before reading entries_ (the queue members are mutable).
    void flush_pending() const;

    // ---- Skip-check for the park path ----
    // True when the disk already holds THIS conversation equal or longer (same branch, same
    // cvec, images lined up), so a new capture+write would add nothing. best() cannot express
    // this: conversation_prefix() is a STRICT-prefix match (n >= prompt.size() returns 0), so
    // an entry exactly the length of the conversation — the retry-loop case — never matches.
    // Also consults ckpt_ids: when the client strips the assistant's thinking tokens, the
    // entry's deepest checkpoint is what equals the conversation the next park would write.
    // Drains the writer first, like every reader.
    bool holds(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec) const;

    // ---- Shared-prefix snapshots ----
    // The token lengths of every prefix-kind entry in the manifest (drains the writer first,
    // like every reader). The engine asks once after reload_from_disk() and keeps the answer
    // as this run's "already on disk" set, so the first request of every new conversation can
    // skip re-capturing (GPU->host, tens of MiB) a prefix the disk already holds.
    std::vector<int64_t> prefix_pin_lengths() const;

    // ---- Restore path (the new conversations.best() falls back to this) ----
    // Same contract as ConversationCache::best(): the longest prefix among the manifest's
    // entries. O(manifest size) in RAM — does not touch the disk payload.
    // Templated on the prompt's token type, mirroring ConversationCache::best(): the engine
    // holds ids as std::vector<long> on Linux x86_64, and a fixed int32_t signature would force
    // a copy at the call site.
    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt,
                                   const std::vector<ConversationImageKey>& images, bool cvec) const;

    // Reads, validates (session_file_read's full check: header, identity, bounded parse, hash)
    // the entry's base .bin — and applies its delta journal on top (each record hash-checked
    // as its bytes stream in; a torn or unchainable tail truncates the journal at the last
    // good record, and a journal whose first record does not chain is dropped whole). The
    // entry STAYS in the manifest (its file too): that is what lets the next park of the same
    // conversation APPEND a delta to this base instead of rewriting the .bin, and what makes
    // holds() skip the re-capture of an unchanged conversation (the retry case). A
    // shared-prefix entry is not consumed either (many conversations branch from it).
    // nullopt: I/O error, corrupt file, identity mismatch, or RAM preflight refusal (the file
    // is kept on a RAM refusal so a later turn with more free RAM can retry — the entry is
    // then dropped from the manifest as before, since nothing about it is usable right now).
    // Phase 2: if the entry is still being written by the background thread, take() waits
    // (condvar) for the write to finish before reading.
    std::optional<SavedConversation> take(const ConversationCache::Match& m, const SessionFileIdentity& id,
                                          const SessionReadLimits& limits = {});

    // ---- Management ----
    size_t size() const {
        flush_pending();
        return entries_.size();
    }
    uint64_t bytes() const {
        flush_pending();
        return bytes_;
    }
    uint64_t budget() const { return budget_; }
    size_t writes() const { return writes_; }
    size_t restores() const { return restores_; }
    size_t drops() const { return drops_; }    // LRU evictions from the disk tier itself
    size_t lost() const { return lost_; }       // RAM-and-disk-both-refused count (logged)

    // LRU eviction of the disk tier itself: when put() would exceed budget_, the least
    // recently used entry is deleted. Returns false only when the manifest is empty (or every
    // candidate is pinned or protected). Called by put() internally — NO flush here (put() and
    // the writer already hold the single-thread invariant; flushing would deadlock the writer).
    // protect_serial: an append's own base never pays for its own record (its `used` stamp is
    // the newest anyway; the guard covers the only-entry case). UINT64_MAX: today's behaviour.
    bool evict_oldest(uint64_t protect_serial = UINT64_MAX);

    // #879 (non-finite logits poison): purge every spill file and clear the manifest.
    // Mirrors ConversationCache::clear() — the disk tier must not keep a poisoned conversation.
    size_t clear();

    // Phase 1: spill does not survive engine restarts. Called once at startup; deletes
    // every conv-*.bin in dir_ and any leftover .tmp files. Phase 3 will replace this with a
    // sidecar .idx scan to keep spills across restarts.
    void purge_on_startup();

    // Reload every valid spill file from disk on engine startup, so a request can restore a
    // conversation the previous engine run spilled. Scans the spill dir for conv-*.bin files,
    // validates each with session_file_read (header + identity + payload hash + full parse),
    // captures the cheap manifest data (ids, imgs, geometry, layer carve, cvec, file_bytes) and
    // DROPS the heavy K/V from RAM right after.
    //
    // The disk budget is enforced by evict_oldest() (the real LRU): files that don't fit are
    // deleted with proper accounting (drops_++). The spill dir can hold conversations from
    // multiple sessions, so we no longer assume the highest serial "includes" the others —
    // every valid file is loaded and the LRU decides what stays.
    //
    // Files that fail validation (corruption, identity mismatch, unsupported version) are
    // deleted: they can never match this engine. A SessionError::memory (RAM preflight refused)
    // KEEPS the file — the next engine start with more free RAM may succeed. A std::bad_alloc
    // mid-read stops the loop: the remaining files stay on disk for a later reload (NOT deleted
    // — they were never validated, and deleting unvalidated files was the bug).
    //
    // Iteration order: ASCENDING by serial (oldest first), so entries_ ends up oldest-first
    // (LRU front = oldest), matching put()'s order. Under budget pressure, evict_oldest()
    // pops the OLDEST spill (the right one) — never the newest.
    //
    // Leftover .tmp files (an interrupted write) are cleaned. Returns the count of files that
    // were read and pushed into the manifest (NOT the count of survivors — see size()).
    size_t reload_from_disk(const SessionFileIdentity& identity);

    // The same drop_superseded contract as ConversationCache's: when a fresh park supersedes
    // an old branch (same deepest checkpoint), the old spill file is deleted. Called by the
    // engine after ConversationCache::drop_superseded.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                            const std::vector<ConversationCheckpoint>& checkpoints, bool cvec);

    // ── Manifest inspection (for diagnostics) ──
    // Returns a pointer to entry i's ids, or nullptr if i is out of range. Used by the
    // restore_from_disk_or_ram diagnostic log to show what the manifest holds vs the prompt.
    const std::vector<int32_t>* entry_ids(size_t i) const {
        return i < entries_.size() ? &entries_[i].ids : nullptr;
    }
    const std::vector<ConversationImageKey>* entry_imgs(size_t i) const {
        return i < entries_.size() ? &entries_[i].imgs : nullptr;
    }
    uint64_t entry_serial(size_t i) const {
        return i < serials_.size() ? serials_[i] : 0;
    }

private:
    std::string dir_;
    uint64_t budget_ = 0;
    uint64_t bytes_ = 0;
    uint64_t next_used_ = 1;        // LRU stamp
    size_t writes_ = 0, restores_ = 0, drops_ = 0, lost_ = 0;
    SessionFileIdentity identity_;  // when bound, written files carry this; reads must match
    bool identity_bound_ = false;
    std::deque<DiskEntry> entries_;   // oldest-first (LRU)
    std::deque<uint64_t> serials_;   // entries_[i]'s serial, mirrors ConversationCache

    // ── Background writer (Phase 2) ── queue + one thread. The queue holds at most 2 images;
    // every public reader drains it first (flush_pending), so entries_ is single-threaded.
    struct PendingWrite {
        SavedConversation image;
        uint64_t serial = 0;
        SessionFileIdentity id;
        SessionWriteOptions options;
        bool prefix = false;   // a shared-prefix snapshot: queue-priority and never dropped for a conversation
        // Cheap copies for the post-write superseded cleanup (the checkpoints are NOT copied:
        // they hold K/V; the cleanup's prefix branch only needs ids/imgs/cvec).
        std::vector<int32_t> sup_ids;
        std::vector<ConversationImageKey> sup_imgs;
        bool sup_cvec = true;
    };
    mutable std::mutex writer_mu_;
    mutable std::condition_variable writer_cv_;   // writer waits for work; flush waits for idle
    std::thread writer_;
    bool writer_stop_ = false;                    // set by the destructor
    bool writer_busy_ = false;                    // writer moved an image out and is writing it
    std::deque<PendingWrite> pending_;
    size_t async_dropped_ = 0;                    // queue-overflowed images (logged)
    // ── Delta journal (read once in the constructor): DEFAULT ON since the v2 multi-image
    // record — a park that finds a delta base appends a conv-<serial>.dlt record instead of
    // rewriting the .bin and its stage files. STRATA_SPILL_JOURNAL=0 opts out (the A/B escape
    // for the pre-journal on-disk shape); the constructor decides, this default only matters
    // before it runs.
    bool journal_ = true;
    // Fresh-serial source for the put() collision guard: reloaded spills keep the serials of
    // the runs that wrote them, while the RAM cache's park counter restarts at 1 every engine
    // start — a colliding park must not replace a DIFFERENT conversation's files. Lifted above
    // every serial seen at reload, and above every serial written since.
    uint64_t next_fresh_serial_ = 1;
    // The identity this run's writes (put/append) and reload carried — the shutdown compaction
    // needs one to re-read the .bin files, and the engine hands it per call instead of holding
    // it here. Set on the first write or reload; unset means nothing on disk is ours to fold.
    SessionFileIdentity write_id_{};
    bool write_id_set_ = false;

    // ── Delta journal: compaction triggers, evaluated on the writer thread when the queue is
    // idle. A journal past either bound is folded back into its .bin (one full rewrite, off
    // every request's path) and the .dlt starts over.
    static constexpr uint64_t kJournalCompactBytes = 64ull * 1024 * 1024;   // or 20% of the .bin
    static constexpr uint32_t kJournalCompactRecords = 16;

    void writer_loop();
    // The sync put()'s body, shared by put() and the writer thread. NO locking inside.
    // A prefix=true write whose exact prefix (ids + images + cvec) is already held SUCCEEDS
    // as a no-op (returns true, nothing written): a restart's first request must not write a
    // duplicate pfx-<serial>.bin of what a previous run left on disk.
    bool put_impl(SavedConversation&& image, uint64_t serial, const SessionFileIdentity& id,
                  const SessionWriteOptions& options, bool prefix);
    // drop_superseded()'s body, shared by the public wrapper and the writer thread.
    // protect_serial: the delta base the writer just appended to — the exact-match, prefix and
    // re-render branches all describe "this conversation's older spill", and after an append
    // the base entry IS the newest state of that conversation, so without the guard the very
    // cleanup that follows every write would delete it (with .bin and .dlt) one step after the
    // append. UINT64_MAX (the default) protects nothing: today's behaviour.
    size_t drop_superseded_impl(const std::vector<int32_t>& ids,
                                const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                                uint64_t protect_serial = UINT64_MAX);
    // Post-reload cleanup: same-conversation duplicates left by earlier runs (one serial per
    // park, none of them superseded because the client re-renders the assistant thinking).
    // The newest serial of each conversation wins; the older files are deleted. Called at
    // the end of reload_from_disk(), so the FIRST start after this fix collapses the
    // accumulated duplicates (measured: 18 files for 1 conversation) back to one set.
    size_t dedupe_loaded();
    // Post-write cleanup the writer runs: same-branch older spills go (prefix of the new live).
    // protect_serial forwards to drop_superseded_impl (the delta base survives its own append).
    size_t drop_superseded_light(const std::vector<int32_t>& ids,
                                 const std::vector<ConversationImageKey>& images, bool cvec,
                                 uint64_t protect_serial = UINT64_MAX) {
        return drop_superseded_impl(ids, images, {}, cvec, protect_serial);
    }
    // ── Delta journal: the writer-side selection and append ──
    // The manifest entry a new image can extend by a delta: same cvec, geometry and layer
    // carve, not a shared prefix, no stage files (a layer split's spill is written whole),
    // and holding a prefix of the image's live ids — either its own ids (the conversation
    // grew) or its deepest checkpoint's (the client re-rendered the history). The candidate
    // keeping the most tokens wins; the base must carry at least half the new image (the
    // codec's own rule) or the record would not save a rewrite.
    struct DeltaBase {
        size_t index = 0;
        uint64_t keep_tokens = 0;
    };
    std::optional<DeltaBase> find_delta_base(const SavedConversation& image) const;
    // Append one record for `image` to base's journal, update the entry (ids, imgs, ckpt,
    // file_bytes, journal accounting) and refresh the .idx. Any anomaly (the file's size does
    // not match the entry's journal_bytes, a plan the codec refuses, an I/O failure) returns
    // false with nothing landed in the manifest — the caller falls back to the full write,
    // whose superseded cleanup then replaces this entry wholesale.
    bool append_delta(size_t base_index, uint64_t keep_tokens, const SavedConversation& image,
                      const SessionFileIdentity& id);
    // Fold one entry's journal back into its .bin (read + apply + rewrite + drop the .dlt).
    // Fail-soft: on any failure the entry is left exactly as it was. Runs on the writer thread
    // only. compact_idle() picks at most one candidate per idle pass; compact_all() is the
    // shutdown drain's sweep (it uses the identity remembered from this run's writes/reload —
    // the engine passes the identity per put_async call, it does not live in the store).
    void compact_journal(DiskEntry& e, const SessionFileIdentity& id);
    bool compact_idle(const SessionFileIdentity& id);
    void compact_all();

    // <dir>/conv-<serial>.bin — a whole conversation. `prefix`: <dir>/pfx-<serial>.bin — a
    // shared-prefix snapshot (the pinned system prompt's K/V). One serial space (the cache's
    // low half, the direct spills' high half): the name prefix tells the kinds apart.
    std::string path_for(uint64_t serial, bool prefix = false) const;
    // <dir>/conv-<serial>.dlt — a conversation's delta journal (never a prefix snapshot's).
    std::string path_for_dlt(uint64_t serial, bool prefix = false) const;
    // A layer split's spill: one extra self-contained session file per later stage
    // (conv-<serial>.stage<k>.bin; a shared-prefix snapshot's is pfx-<serial>.stage<k>.bin),
    // written by put() BEFORE the main .bin (the main is the commit point of the put) and
    // read back by take() right after the main image. The .idx sidecar does not describe
    // them; a stage file's bytes count toward DiskEntry::file_bytes.
    std::string path_for_stage(uint64_t serial, size_t k, bool prefix = false) const;
    // Total size of serial's stage files on disk (0 when the spill has none; a stat error
    // ends the sum, never throws). reload_from_disk's slow path adds this to the entry's
    // file_bytes: the .bin parse covers the main image only, and the .idx (when fresh)
    // already carries the put-time total, which includes the stage files.
    uint64_t stage_files_bytes(uint64_t serial, bool prefix = false) const;
    // Sidecar .idx: <dir>/conv-<serial>.idx (or pfx-<serial>.idx for a prefix snapshot) —
    // small manifest metadata without parsing the .bin.
    std::string path_for_idx(uint64_t serial, bool prefix = false) const;
    // Write the .idx atomically (temp + rename). Best-effort: on failure returns false, the
    // .bin is still valid and the next start falls back to a full parse of it.
    bool write_idx(const DiskEntry& e, const std::string& bin_path, const SessionFileIdentity& id) const;
    // Read and validate the .idx. Returns false on any I/O/parse/checksum/identity/freshness
    // error — never throws, never asserts. On success, fills `out` with the manifest data
    // (path set to the .bin). The caller sets serial, used, and pushes into the manifest.
    bool read_idx(const std::string& idx_path, DiskEntry& out, const SessionFileIdentity& id) const;
    // Remove a spill's .bin and its .idx (and the .idx.tmp) — best-effort, all three with
    // std::error_code. Used by every site that today does fs::remove(.bin).
    void remove_spill_files(const std::string& bin_path);
    void erase(size_t index);
    DiskEntry* find_by_serial(uint64_t serial);
    const DiskEntry* find_by_serial(uint64_t serial) const;
};

// ── Free-function helpers (testable without GPU/runtime) ─────────────────────────────────
// These wrap the integration points the engine calls: they take simple types (the cache, the
// store, plain id lists) so a test can exercise the cache+disk cooperation without loading any
// model. The engine's generate.cpp calls these (one or two lines per integration point), and
// the bulk of the spill logic is testable in src/core/conversation_spill_integration_test.cpp.

/// Configuration for the disk-tier spill store. Mirrors the engine's Options fields; passed
/// as a plain struct so the test does not depend on the engine's Options type.
struct SpillConfig {
    std::string path;            // empty: use data_dir/strata-spill/
    uint64_t budget_gib = 0;     // 0: disabled (today's behavior, A/B safe)
    int64_t min_tokens = 0;      // 0: inherit conversation-cache-min-tokens
    std::string data_dir;         // the model's data dir, used when path is empty
};

/// Build a ConversationDiskStore from SpillConfig + identity. Returns a disabled store
/// (budget = 0) when config.budget_gib == 0, so the engine's "if (disk_store.enabled())"
/// guards skip the spill path with zero overhead. The directory is created if missing.
/// The identity pointer is stored (non-owning); pass nullptr for tests that don't bind it.
std::unique_ptr<ConversationDiskStore> make_disk_store(const SpillConfig& cfg,
                                                        const SessionFileIdentity* identity);

/// Bind a disk store to a conversation cache (the engine's startup wiring, punta 1).
/// After this, every ConversationCache::evict_oldest() writes the evicted entry to disk
/// instead of dropping it. No-op when store is nullptr or disabled.
void attach_spill(ConversationCache& cache, ConversationDiskStore* store,
                  const SessionFileIdentity* identity);

/// Restore-path fallback (puntas 3+4): when the RAM cache has no usable prefix (parked.tokens
/// <= max(resume, slot_tokens)), consult the disk store. If the disk has a better match, take
/// it. Returns the restored image (or nullopt on miss / I/O error / RAM refusal). The caller
/// (engine) then runs the existing conversation_snapshot_validate / restore path on the image,
/// unchanged. The image's K/V is fully materialized in host RAM by this call.
///
/// `parked` is the RAM cache's best() result; `disk` is the spill store (may be nullptr or
/// disabled — returns nullopt in that case). `limits` carries the engine's RAM preflight.
/// Templated on the prompt's token type to mirror ConversationCache::best(): the engine holds
/// ids as std::vector<long> on Linux x86_64.
template<class Token>
std::optional<SavedConversation> restore_from_disk_or_ram(
    const ConversationCache::Match& parked, const ConversationDiskStore* disk,
    const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
    int64_t resume, int64_t slot_tokens, const SessionFileIdentity& identity,
    const SessionReadLimits& limits = {});

/// Drop superseded entries from the disk store (punta 5). Mirrors ConversationCache::
/// drop_superseded — when a fresh park supersedes an old branch (same deepest checkpoint),
/// the old spill file is deleted. No-op when disk is nullptr or disabled. Returns the count
/// dropped. Kept as a free function so the engine can call it after the cache's drop_superseded
/// with the same arguments, and tests can exercise it directly.
size_t spill_drop_superseded(ConversationDiskStore* disk,
                             const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                             const std::vector<ConversationCheckpoint>& checkpoints, bool cvec);

/// Clear the disk store (punta 6, the #879 non-finite poison case). Mirrors ConversationCache::
/// clear — after NaN logits, every parked conversation (RAM and disk) goes. No-op when disk is
/// nullptr. Returns the count of disk entries removed.
size_t spill_clear(ConversationDiskStore* disk);

/// Purge the spill directory at engine startup (punta 7). Phase 1: spills do not survive
/// restarts; this deletes every conv-*.bin and leftover .tmp from a prior interrupted write.
/// No-op when disk is nullptr.
void spill_purge_on_startup(ConversationDiskStore* disk);

/// Reload the spill manifest from disk at engine startup. Validates and loads EVERY valid
/// conv-*.bin in the spill dir; the disk budget is enforced by evict_oldest() (the real LRU),
/// so files that don't fit are deleted with proper accounting (drops_++). The spill dir can
/// hold conversations from multiple sessions — we no longer assume the highest serial
/// "includes" the others.
///
/// Called after attach_spill binds the engine's SessionFileIdentity. Files that fail
/// validation (corruption, identity mismatch) are deleted; SessionError::memory keeps the
/// file for a later start with more free RAM; a std::bad_alloc mid-read stops the loop and
/// leaves the remaining files on disk (NOT deleted — they were never validated). No-op when
/// disk is nullptr. Returns the count of files that were read and pushed (NOT the survivors —
/// see ConversationDiskStore::size()).
size_t spill_reload_from_disk(ConversationDiskStore* disk, const SessionFileIdentity& identity);

// ── Template implementations ──────────────────────────────────────────────────────────────
// Templates cannot live in conversation_disk_store.cpp; their definitions must be visible at
// the call site (the engine instantiates them with std::vector<long>, the tests with
// std::vector<int32_t>). Both delegate to non-template helpers in the .cpp that take
// std::vector<int32_t> after a cheap conversion, so the disk payload stays in the .cpp.

namespace detail {

// Non-template helper used by the template best(): builds a ConversationCheckpoint view from
// the manifest entry and calls conversation_prefix(). Defined in conversation_disk_store.cpp.
int64_t best_match_tokens(const std::vector<int32_t>& entry_ids,
                          const std::vector<ConversationImageKey>& entry_imgs,
                          const std::vector<int32_t>& prompt,
                          const std::vector<ConversationImageKey>& prompt_imgs);

} // namespace detail

template<class Token>
ConversationCache::Match ConversationDiskStore::best(const std::vector<Token>& prompt,
                                                     const std::vector<ConversationImageKey>& images,
                                                     bool cvec) const {
    ConversationCache::Match best;
    if (!enabled() || prompt.empty()) return best;
    flush_pending();   // a queued (not yet written) spill is invisible until it lands
    // Convert the prompt to int32_t once: the engine holds ids as `long` on Linux x86_64, but
    // tokens are always <= INT32_MAX (a 32-bit model vocab fits in int32_t). The conversion is
    // a single pass and avoids forcing the caller to copy.
    std::vector<int32_t> prompt32;
    prompt32.reserve(prompt.size());
    for (const auto& t : prompt) prompt32.push_back(static_cast<int32_t>(t));
    for (size_t i = entries_.size(); i-- > 0;) {
        const auto& e = entries_[i];
        if (e.cvec != cvec) continue;
        // Try matching against the live ids first (the full conversation state including
        // reasoning tokens). If that fails (the client strips thinking tokens), try the
        // deepest checkpoint's ids (the turn boundary — what the next request's prompt
        // actually contains).
        int64_t n = detail::best_match_tokens(e.ids, e.imgs, prompt32, images);
        if (n <= 0 && !e.ckpt_ids.empty()) {
            n = detail::best_match_tokens(e.ckpt_ids, e.ckpt_imgs, prompt32, images);
        }
        if (n > best.tokens) {
            best.tokens = n;
            best.serial = e.serial;
            best.index = i;
            best.live = true;
        }
    }
    return best;
}

template<class Token>
std::optional<SavedConversation> restore_from_disk_or_ram(
    const ConversationCache::Match& parked, const ConversationDiskStore* disk,
    const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
    int64_t resume, int64_t slot_tokens, const SessionFileIdentity& identity,
    const SessionReadLimits& limits) {
    const int64_t ram_floor = std::max<int64_t>(resume, slot_tokens);
    if (parked.tokens > ram_floor) return std::nullopt;   // RAM wins; engine takes from cache
    if (disk == nullptr || !disk->enabled()) return std::nullopt;
    // Convert prompt to int32_t for the disk store's best() (which iterates the manifest in RAM).
    std::vector<int32_t> prompt32;
    prompt32.reserve(prompt.size());
    for (const auto& t : prompt) prompt32.push_back(static_cast<int32_t>(t));
    auto disk_match = const_cast<ConversationDiskStore*>(disk)->best(prompt32, images, cvec);
    // ── DIAGNOSTIC: when the disk has entries but no match (or a match below the floor), say so.
    // This pinpoints whether the issue is (a) no matching prefix (the manifest entries' ids don't
    // line up with the prompt's) or (b) a matching prefix that the floor caps out below.
    if (disk_match.tokens <= ram_floor) {
        std::fprintf(stderr, "[strata diag] restore_from_disk_or_ram: no usable disk match "
                              "(disk_size=%zu, disk_best=%lld, ram_floor=%lld, prompt=%zu, cvec=%d, prompt_imgs=%zu)\n",
                     disk->size(), (long long) disk_match.tokens, (long long) ram_floor,
                     prompt.size(), (int) cvec, images.size());
        // Print the first/last few ids of each manifest entry vs the prompt, to see where the
        // mismatch is (when the manifest has entries but best() returns 0).
        if (disk->size() > 0) {
            std::fprintf(stderr, "[strata diag]   prompt first 5 ids:");
            for (size_t i = 0; i < prompt.size() && i < 5; ++i)
                std::fprintf(stderr, " %d", (int) prompt32[i]);
            std::fprintf(stderr, " | last 5:");
            for (size_t i = prompt32.size() >= 5 ? prompt32.size() - 5 : 0; i < prompt32.size(); ++i)
                std::fprintf(stderr, " %d", (int) prompt32[i]);
            std::fprintf(stderr, "\n");
            // Show the manifest entries' ids too, so we can compare against the prompt
            for (size_t i = 0; i < disk->size() && i < 4; ++i) {   // first 4 entries
                const std::vector<int32_t>* e_ids = disk->entry_ids(i);
                if (e_ids && !e_ids->empty()) {
                    std::fprintf(stderr, "[strata diag]   manifest[%zu] serial=%llu n_ids=%zu first 5:",
                                 i, (unsigned long long) disk->entry_serial(i), e_ids->size());
                    for (size_t k = 0; k < e_ids->size() && k < 5; ++k)
                        std::fprintf(stderr, " %d", (int) (*e_ids)[k]);
                    std::fprintf(stderr, " | last 5:");
                    for (size_t k = e_ids->size() >= 5 ? e_ids->size() - 5 : 0; k < e_ids->size(); ++k)
                        std::fprintf(stderr, " %d", (int) (*e_ids)[k]);
                    const std::vector<ConversationImageKey>* e_imgs = disk->entry_imgs(i);
                    std::fprintf(stderr, " imgs=%zu cvec=%d\n", e_imgs ? e_imgs->size() : 0, (int) cvec);
                    // Show where the manifest and prompt diverge: compare byte by byte until
                    // the first mismatch. This tells us exactly which token broke the prefix.
                    const size_t cmp_n = std::min(e_ids->size(), prompt32.size());
                    size_t diverge = 0;
                    while (diverge < cmp_n && (*e_ids)[diverge] == prompt32[diverge]) ++diverge;
                    if (diverge < cmp_n) {
                        std::fprintf(stderr, "[strata diag]     diverge at token %zu: manifest=%d prompt=%d\n",
                                     diverge, (int) (*e_ids)[diverge], (int) prompt32[diverge]);
                        // Show 3 tokens around the divergence point
                        const size_t show_start = diverge >= 3 ? diverge - 3 : 0;
                        const size_t show_end = std::min(diverge + 4, cmp_n);
                        std::fprintf(stderr, "[strata diag]     around divergence:");
                        for (size_t k = show_start; k < show_end; ++k) {
                            const char* mark = (k == diverge) ? ">>" : "  ";
                            std::fprintf(stderr, " %s[%zu]m=%d p=%d", mark, k, (int) (*e_ids)[k], (int) prompt32[k]);
                        }
                        std::fprintf(stderr, "\n");
                    } else if (e_ids->size() < prompt32.size()) {
                        std::fprintf(stderr, "[strata diag]     manifest is a strict prefix of prompt "
                                     "(all %zu tokens match; prompt has %zu more)\n",
                                     e_ids->size(), prompt32.size() - e_ids->size());
                    } else if (e_ids->size() == prompt32.size()) {
                        std::fprintf(stderr, "[strata diag]     manifest and prompt have the same length "
                                     "and all %zu tokens match — but best() still returned 0?\n", e_ids->size());
                    }
                }
            }
        }
        return std::nullopt;
    }
    // ── Escalation: the best disk match is not always readable — a multi-hundred-MiB whole-
    // conversation image can fail under RAM pressure (session_file_read's preflight / bad_alloc)
    // or a corrupt .bin. take() removes the failed entry from the manifest (its file stays on
    // disk), so instead of falling straight back to a full recompute, try the NEXT-best match:
    // a smaller pinned-prefix snapshot usually still fits and covers the system prompt, which is
    // the part worth saving. Bounded: three attempts, each a manifest scan and (on failure) a
    // cheap refused read — never a long drain of a big manifest.
    for (int attempt = 0; attempt < 3 && disk_match.tokens > ram_floor; ++attempt) {
        if (auto image = const_cast<ConversationDiskStore*>(disk)->take(disk_match, identity, limits))
            return image;
        disk_match = const_cast<ConversationDiskStore*>(disk)->best(prompt32, images, cvec);
    }
    return std::nullopt;
}

} // namespace strata::core
