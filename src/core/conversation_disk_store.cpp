// Disk-tier spill store. See include/strata/core/conversation_disk_store.hpp for the design.
// Phase 1: synchronous put/take. Reuses session_file_write/read for the actual file format
// (v1 little-endian, atomic temp+rename+fsync, identity check, payload hash) — same format
// as /slots/0?action=save, so a spilled conversation is byte-identical to a manually saved one.
#include "strata/core/conversation_disk_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <utility>

namespace strata::core {
namespace fs = std::filesystem;

// Out-of-line spill helper invoked by ConversationCache::evict_oldest() (defined in
// conversation_cache.hpp). The cache header only forward-declares ConversationDiskStore to
// avoid a circular include with conversation_file.hpp; this translation unit includes the
// full definition, so it can call put() directly. Fail-soft: a full disk or I/O error
// leaves the entry lost (today's behavior — the spill only ever improves the state).
void spill_evict_one(ConversationDiskStore& store, const SessionFileIdentity& identity,
                     SavedConversation&& image, uint64_t serial) {
    (void) store.put(std::move(image), serial, identity);
}

ConversationDiskStore::ConversationDiskStore(std::string dir, uint64_t budget_bytes,
                                             const SessionFileIdentity* identity)
    : dir_(std::move(dir)), budget_(budget_bytes) {
    if (identity) { identity_ = *identity; identity_bound_ = true; }
    if (budget_ == 0 || dir_.empty()) { budget_ = 0; return; }
    // Create the spill dir (idempotent). 0700: only the engine user can read parked
    // conversations (they may carry prompt contents).
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (ec) {
        std::fprintf(stderr, "strata serve: conversation spill: could not create %s: %s; spilling disabled\n",
                     dir_.c_str(), ec.message().c_str());
        budget_ = 0;
        return;
    }
    fs::permissions(dir_, fs::perms::owner_all, ec);   // best-effort; ignore if unsupported
}

ConversationDiskStore::~ConversationDiskStore() = default;

ConversationDiskStore::ConversationDiskStore(ConversationDiskStore&&) noexcept = default;
ConversationDiskStore& ConversationDiskStore::operator=(ConversationDiskStore&&) noexcept = default;

std::string ConversationDiskStore::path_for(uint64_t serial) const {
    return dir_ + "/conv-" + std::to_string(serial) + ".bin";
}

void ConversationDiskStore::erase(size_t index) {
    if (index >= entries_.size()) return;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    serials_.erase(serials_.begin() + static_cast<std::ptrdiff_t>(index));
}

DiskEntry* ConversationDiskStore::find_by_serial(uint64_t serial) {
    for (size_t i = 0; i < serials_.size(); ++i)
        if (serials_[i] == serial) return &entries_[i];
    return nullptr;
}

const DiskEntry* ConversationDiskStore::find_by_serial(uint64_t serial) const {
    for (size_t i = 0; i < serials_.size(); ++i)
        if (serials_[i] == serial) return &entries_[i];
    return nullptr;
}

bool ConversationDiskStore::put(SavedConversation&& image, uint64_t serial,
                                const SessionFileIdentity& id, const SessionWriteOptions& options) {
    if (!enabled()) return false;
    // If this serial is already spilled (re-park of a conversation that came back from disk
    // and got evicted again), replace it: delete the old file and re-add the entry.
    if (DiskEntry* existing = find_by_serial(serial)) {
        std::error_code ec;
        if (fs::remove(existing->path, ec) && !ec) {
            bytes_ -= existing->file_bytes;
            const size_t idx = static_cast<size_t>(existing - &entries_.front());
            erase(idx);
        }
    }
    // Capture the cheap manifest data before we move the image out (kv + checkpoints live on).
    // We mirror what ConversationCache's best() matches against: the live state and the deepest
    // checkpoint. session_checkpoints_to_save keeps the deepest one when writing, so the file's
    // image.checkpoints will hold it (and best() below consults both).
    DiskEntry e;
    e.serial = serial;
    e.path = path_for(serial);
    e.ids = image.live.ids;                  // copy: the manifest keeps these in RAM
    e.imgs = image.live.imgs;
    e.geometry = image.geometry;
    e.layer_lo = image.layer_lo;
    e.layer_hi = image.layer_hi;
    e.cvec = image.cvec;

    // Make room in the disk budget before we write. Same pattern as ConversationCache::make_room.
    // We don't know the on-disk bytes yet, so estimate from the in-memory image: SavedConversation::bytes()
    // is the host footprint of the K/V + running state, which is what the v1 file format stores
    // (the file's overhead is ~80 B header + per-array length fields, <1% of the K/V at 72K tokens).
    const uint64_t est_file = image.bytes();
    while (!entries_.empty() && bytes_ + est_file > budget_) {
        if (!evict_oldest()) break;   // nothing to evict: keep going (best effort)
    }
    // Re-check after eviction: still over budget with an empty manifest? Skip the write to
    // avoid filling the disk past the budget. The caller drops the image as today.
    if (est_file > budget_) return false;

    size_t written = 0;
    std::string error;
    SessionStatus status;
    SessionWriteOptions opts = options;
    if (opts.min_free_bytes == 0) {
        // Leave a sane default headroom: 256 MiB. A full disk degrades to today's behavior.
        opts.min_free_bytes = 256ull * 1024 * 1024;
    }
    const bool ok = session_file_write(e.path, image, id, written, error, opts, &status);
    // image is consumed conceptually by the write (it reads its K/V); we move it to discard.
    // The disk store does NOT keep a copy in RAM — that's the whole point of the disk tier.
    { SavedConversation discard = std::move(image); (void) discard; }
    if (!ok) {
        // Disk full / I/O error / RAM refusal (shouldn't happen on the write side, but preflight
        // can refuse on min_free_bytes). The file was never published (rename did not happen).
        // Clean up any leftover temp file.
        std::error_code ec;
        fs::remove(e.path + ".tmp", ec);   // best-effort
        return false;
    }
    e.file_bytes = written;
    e.used = next_used_++;
    bytes_ += written;
    entries_.push_back(std::move(e));
    serials_.push_back(serial);
    ++writes_;
    return true;
}

// Non-template helper for the templated ConversationDiskStore::best<Token>(): builds a
// ConversationCheckpoint view from the manifest entry and calls conversation_prefix(). The
// template (in the header) converts the prompt to int32_t once, then calls this for each entry.
namespace detail {
int64_t best_match_tokens(const std::vector<int32_t>& entry_ids,
                          const std::vector<ConversationImageKey>& entry_imgs,
                          const std::vector<int32_t>& prompt,
                          const std::vector<ConversationImageKey>& prompt_imgs) {
    ConversationCheckpoint live_view;
    live_view.ids = entry_ids;
    live_view.imgs = entry_imgs;
    return conversation_prefix(live_view, prompt, prompt_imgs);
}
} // namespace detail

std::optional<SavedConversation> ConversationDiskStore::take(const ConversationCache::Match& m,
                                                              const SessionFileIdentity& id,
                                                              const SessionReadLimits& limits) {
    if (!enabled() || m.tokens <= 0) return std::nullopt;
    // Resolve the manifest entry by serial (m.serial survives cache changes; m.index does not).
    const DiskEntry* e = find_by_serial(m.serial);
    if (!e) return std::nullopt;
    std::string path = e->path;
    SavedConversation back;
    size_t read = 0;
    std::string error;
    SessionStatus status;
    const bool ok = session_file_read(path, id, back, read, error, limits, &status);
    if (!ok) {
        // Identity mismatch or corruption: the file is unusable. Drop it from the manifest
        // (it can never match this engine) and try to delete it. RAM preflight refusal is the
        // one case where we KEEP the file (the caller's admit said "no RAM now, retry later").
        const bool keep_file = (status.error == SessionError::memory);
        if (!keep_file) {
            std::error_code ec;
            fs::remove(path, ec);
        }
        // Remove from manifest regardless: a corrupt or wrong-identity file is dead weight.
        for (size_t i = 0; i < serials_.size(); ++i) {
            if (serials_[i] == m.serial) { bytes_ -= entries_[i].file_bytes; erase(i); break; }
        }
        return std::nullopt;
    }
    // Success: remove the entry and delete the file (the conversation is back in RAM).
    for (size_t i = 0; i < serials_.size(); ++i) {
        if (serials_[i] == m.serial) { bytes_ -= entries_[i].file_bytes; erase(i); break; }
    }
    std::error_code ec;
    fs::remove(path, ec);   // best-effort; an orphan file is harmless (purge_on_startup cleans)
    ++restores_;
    return back;
}

bool ConversationDiskStore::evict_oldest() {
    if (entries_.empty()) return false;
    // entries_ is oldest-first (LRU). The front is the one to drop.
    const DiskEntry e = std::move(entries_.front());
    bytes_ -= e.file_bytes;
    entries_.pop_front();
    serials_.pop_front();
    std::error_code ec;
    if (!fs::remove(e.path, ec) && ec) {
        // File already gone (manual cleanup, restart with a stale manifest) — silently move on.
    }
    ++drops_;
    return true;
}

size_t ConversationDiskStore::clear() {
    const size_t n = entries_.size();
    std::error_code ec;
    for (const auto& e : entries_) fs::remove(e.path, ec);   // best-effort
    entries_.clear();
    serials_.clear();
    bytes_ = 0;
    return n;
}

void ConversationDiskStore::purge_on_startup() {
    entries_.clear();
    serials_.clear();
    bytes_ = 0;
    if (dir_.empty()) return;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        // conv-<serial>.bin and any leftover .tmp from an interrupted write.
        if (name.rfind("conv-", 0) == 0 && (name.size() > 4 && name.compare(name.size() - 4, 4, ".bin") == 0)) {
            fs::remove(entry.path(), ec);
        } else if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            fs::remove(entry.path(), ec);
        }
    }
}

size_t ConversationDiskStore::reload_from_disk(const SessionFileIdentity& identity) {
    // Phase 3: spills survive engine restarts. Scan the spill dir for conv-*.bin files written
    // by a previous engine run, validate each one with session_file_read (header + identity +
    // payload hash + full parse), capture the cheap manifest data (ids, imgs, geometry, layer
    // carve, cvec, file_bytes), then DROP the heavy K/V image from RAM. The next request's
    // best() finds the entry in the manifest and take() re-reads the file.
    //
    // Files that fail validation (corruption, identity mismatch, unsupported version) are
    // deleted: they can never match this engine. A SessionError::memory (RAM preflight
    // refused) keeps the file — the next engine start with more free RAM may succeed.
    //
    // Leftover .tmp files (an interrupted write) are also cleaned: they were never renamed
    // to .bin, so they are not valid spills.
    if (!enabled() || dir_.empty()) return 0;
    // Clear any stale manifest first (we are about to rebuild it from disk).
    entries_.clear();
    serials_.clear();
    bytes_ = 0;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return 0;
    // Collect conv-*.bin files sorted by serial (filename sorts lexicographically the same
    // as numeric for the decimal form std::to_string produces, so the LRU order entries_ is
    // built oldest-first — mirrors the order put() inserts at).
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.rfind("conv-", 0) == 0 && name.size() > 9 &&
            name.compare(name.size() - 4, 4, ".bin") == 0) {
            files.push_back(entry.path());
        } else if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            // Leftover .tmp from an interrupted write: clean it up.
            fs::remove(entry.path(), ec);
        }
    }
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().string() < b.filename().string();
              });
    size_t loaded = 0;
    for (const auto& path : files) {
        const std::string name = path.filename().string();   // conv-<serial>.bin
        // Parse the serial from the filename: conv-<decimal>.bin
        uint64_t serial = 0;
        try {
            serial = std::stoull(name.substr(5, name.size() - 9));
        } catch (...) {
            std::error_code ec2;
            fs::remove(path, ec2);   // not a valid conv-<serial>.bin name: drop it
            continue;
        }
        SavedConversation image;
        size_t bytes_read = 0;
        std::string error;
        SessionStatus status;
        if (!session_file_read(path.string(), identity, image, bytes_read, error, {}, &status)) {
            // Identity mismatch, corruption, or RAM refusal. For memory (RAM preflight): keep
            // the file, the next start with more RAM may succeed. Anything else: delete.
            if (status.error != SessionError::memory) {
                std::error_code ec2;
                fs::remove(path, ec2);
            }
            continue;
        }
        // Capture the manifest data; the heavy K/V is dropped right after.
        DiskEntry e;
        e.serial = serial;
        e.path = path.string();
        e.ids = image.live.ids;
        e.imgs = image.live.imgs;
        e.geometry = image.geometry;
        e.layer_lo = image.layer_lo;
        e.layer_hi = image.layer_hi;
        e.cvec = image.cvec;
        e.file_bytes = bytes_read;
        e.used = next_used_++;
        { SavedConversation discard = std::move(image); (void) discard; }   // free the K/V
        // LRU evict if over budget: drop the oldest entries (which we just loaded first).
        while (!entries_.empty() && bytes_ + e.file_bytes > budget_) {
            if (!evict_oldest()) break;
        }
        if (e.file_bytes > budget_) continue;   // single file bigger than the whole budget
        entries_.push_back(std::move(e));
        serials_.push_back(serial);
        bytes_ += bytes_read;
        ++loaded;
    }
    return loaded;
}

size_t ConversationDiskStore::drop_superseded(const std::vector<int32_t>& ids,
                                              const std::vector<ConversationImageKey>& images,
                                              const std::vector<ConversationCheckpoint>& checkpoints,
                                              bool cvec) {
    auto held = [&](const std::vector<int32_t>& cid, const std::vector<ConversationImageKey>& cimgs) {
        if (cid == ids && cimgs == images) return true;
        for (const auto& k : checkpoints)
            if (k.ids == cid && k.imgs == cimgs) return true;
        return false;
    };
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        const auto& e = entries_[i];
        if (e.cvec != cvec) { ++i; continue; }
        // The deepest checkpoint the spill file carries is the one ConversationCache handed to
        // session_checkpoints_to_save. The manifest only stores the live prefix; if the live
        // prefix itself is what the new conversation supersedes, drop. (Phase 2: store the
        // deepest checkpoint's ids/imgs in the manifest too for a more accurate match.)
        if (!e.ids.empty() && held(e.ids, e.imgs)) {
            std::error_code ec;
            fs::remove(e.path, ec);
            bytes_ -= e.file_bytes;
            erase(i);
            ++dropped;
            continue;
        }
        ++i;
    }
    return dropped;
}

// ── Free-function helpers (engine integration points; testable without GPU) ───────────────
// These are the wrappers generate.cpp calls. They take simple types so the engine's edits are
// 1-2 lines each, and the integration tests in conversation_spill_integration_test.cpp can
// exercise them with synthetic ConversationCache + ConversationDiskStore instances.

std::unique_ptr<ConversationDiskStore> make_disk_store(const SpillConfig& cfg,
                                                        const SessionFileIdentity* identity) {
    // Resolve the directory: explicit path wins; otherwise <data_dir>/strata-spill/.
    std::string dir = cfg.path;
    if (dir.empty()) {
        if (cfg.data_dir.empty()) return nullptr;   // nothing to default to: spill off
        dir = cfg.data_dir + "/strata-spill";
    }
    const uint64_t budget_bytes = cfg.budget_gib * 1024ull * 1024 * 1024;
    if (budget_bytes == 0) {
        // Disabled store: still create the directory so purge_on_startup() can clean a stale
        // dir from a previous run with the spill on. The store's enabled() returns false.
        std::error_code ec;
        fs::create_directories(dir, ec);
        // Best-effort: ignore failures — the engine starts anyway, just without spill.
    }
    return std::make_unique<ConversationDiskStore>(dir, budget_bytes, identity);
}

void attach_spill(ConversationCache& cache, ConversationDiskStore* store,
                  const SessionFileIdentity* identity) {
    if (store == nullptr || !store->enabled() || identity == nullptr) {
        // Detach: pass nullptr to revert to today's behavior (entry dropped on eviction).
        cache.set_spill_store(nullptr, nullptr);
        return;
    }
    cache.set_spill_store(store, identity);
}

// restore_from_disk_or_ram is templated and lives in the header (templates cannot have their
// definition in a .cpp). It delegates to the templated ConversationDiskStore::best<Token>().

size_t spill_drop_superseded(ConversationDiskStore* disk,
                             const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                             const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    if (disk == nullptr || !disk->enabled()) return 0;
    return disk->drop_superseded(ids, images, checkpoints, cvec);
}

size_t spill_clear(ConversationDiskStore* disk) {
    if (disk == nullptr || !disk->enabled()) return 0;
    return disk->clear();
}

void spill_purge_on_startup(ConversationDiskStore* disk) {
    if (disk == nullptr) return;
    disk->purge_on_startup();   // safe on a disabled store too (cleans stale files)
}

size_t spill_reload_from_disk(ConversationDiskStore* disk, const SessionFileIdentity& identity) {
    if (disk == nullptr) return 0;
    return disk->reload_from_disk(identity);   // no-op on a disabled store
}

} // namespace strata::core
