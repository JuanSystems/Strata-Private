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
#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <optional>
#include <string>
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
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
    uint64_t file_bytes = 0;                           // for LRU byte-budget accounting
    uint64_t used = 0;                                 // LRU stamp (monotonic)
};

/// Disk-tier spill store. RAM manifest + payload-on-disk. Take = remove: when a conversation
/// comes back from disk, its file is deleted (no orphan copies of what's already in RAM).
/// All operations are fail-soft: a full or corrupt disk degrades to today's behavior, never
/// worse. Phase 1 is synchronous (put blocks the calling park path). Phase 2 will add a
/// background writer (StageThreads pattern does not apply; needs a queue-based thread).
class ConversationDiskStore {
public:
    /// `dir` is created (with parents) if missing. `budget_bytes = 0` disables the store
    /// (put() returns false; best() returns no match). `identity = nullptr` skips the
    /// model/config fingerprint check on read (test fixture only).
    ConversationDiskStore(std::string dir, uint64_t budget_bytes,
                          const SessionFileIdentity* identity = nullptr);
    ~ConversationDiskStore();

    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;
    ConversationDiskStore(ConversationDiskStore&&) noexcept;
    ConversationDiskStore& operator=(ConversationDiskStore&&) noexcept;

    bool enabled() const { return budget_ != 0; }
    const std::string& dir() const { return dir_; }

    // ---- Park path (the new evict_oldest calls this) ----
    // Writes `image` (captured in RAM by ConversationCache::take) to <dir>/conv-<serial>.bin
    // using session_file_write (atomic temp+rename+fsync). On a full disk or I/O error returns
    // false: the caller then drops the image as today (the spill only ever improves the
    // current state). `serial` is the cache's serial for this entry — the file name is
    // derived from it, so two parked conversations never collide. Phase 2: this becomes the
    // enqueue point for the background writer.
    bool put(SavedConversation&& image, uint64_t serial, const SessionFileIdentity& id,
             const SessionWriteOptions& options = {});

    // ---- Restore path (the new conversations.best() falls back to this) ----
    // Same contract as ConversationCache::best(): the longest prefix among the manifest's
    // entries. O(manifest size) in RAM — does not touch the disk payload.
    // Templated on the prompt's token type, mirroring ConversationCache::best(): the engine
    // holds ids as std::vector<long> on Linux x86_64, and a fixed int32_t signature would force
    // a copy at the call site.
    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt,
                                   const std::vector<ConversationImageKey>& images, bool cvec) const;

    // Reads, validates (session_file_read's full check: header, identity, bounded parse, hash),
    // and REMOVES the entry from the manifest and its file from disk. nullopt: I/O error,
    // corrupt file, identity mismatch, or RAM preflight refusal (the file is kept on a RAM
    // refusal so a later turn with more free RAM can retry). Phase 2: if the entry is still
    // being written by the background thread, take() waits (condvar) for the write to finish
    // before reading.
    std::optional<SavedConversation> take(const ConversationCache::Match& m, const SessionFileIdentity& id,
                                          const SessionReadLimits& limits = {});

    // ---- Management ----
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t budget() const { return budget_; }
    size_t writes() const { return writes_; }
    size_t restores() const { return restores_; }
    size_t drops() const { return drops_; }    // LRU evictions from the disk tier itself
    size_t lost() const { return lost_; }       // RAM-and-disk-both-refused count (logged)

    // LRU eviction of the disk tier itself: when put() would exceed budget_, the oldest entry
    // is deleted. Returns false only when the manifest is empty (or all entries are pinned,
    // which Phase 1 does not have). Called by put().
    bool evict_oldest();

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

    std::string path_for(uint64_t serial) const;
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
    // Convert the prompt to int32_t once: the engine holds ids as `long` on Linux x86_64, but
    // tokens are always <= INT32_MAX (a 32-bit model vocab fits in int32_t). The conversion is
    // a single pass and avoids forcing the caller to copy.
    std::vector<int32_t> prompt32;
    prompt32.reserve(prompt.size());
    for (const auto& t : prompt) prompt32.push_back(static_cast<int32_t>(t));
    for (size_t i = entries_.size(); i-- > 0;) {
        const auto& e = entries_[i];
        if (e.cvec != cvec) continue;
        const int64_t n = detail::best_match_tokens(e.ids, e.imgs, prompt32, images);
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
                }
            }
        }
        return std::nullopt;
    }
    return const_cast<ConversationDiskStore*>(disk)->take(disk_match, identity, limits);
}

} // namespace strata::core
