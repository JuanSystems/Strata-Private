// Disk-tier spill store. See include/strata/core/conversation_disk_store.hpp for the design.
// Phase 1: synchronous put/take. Reuses session_file_write/read for the actual file format
// (v1 little-endian, atomic temp+rename+fsync, identity check, payload hash) — same format
// as /slots/0?action=save, so a spilled conversation is byte-identical to a manually saved one.
#include "strata/core/conversation_disk_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace fs = std::filesystem;

// Out-of-line spill helper invoked by ConversationCache::evict_oldest() (defined in
// conversation_cache.hpp). The cache header only forward-declares ConversationDiskStore to
// avoid a circular include with conversation_file.hpp; this translation unit includes the
// full definition, so it can call put() directly. Fail-soft: a full disk or I/O error
// leaves the entry lost (today's behavior — the spill only ever improves the state).
// Phase 2: the eviction path also writes in the BACKGROUND — make_room() runs inside the
// park, which is on the request path, and a ~hundreds-of-MiB sync write there stalls the
// next request's prompt read exactly like the park-spill did.
void spill_evict_one(ConversationDiskStore& store, const SessionFileIdentity& identity,
                     SavedConversation&& image, uint64_t serial) {
    (void) store.put_async(std::move(image), serial, identity);
}

// ── Background writer (Phase 2) ────────────────────────────────────────────────────────────

// ── Prefix snapshots ───────────────────────────────────────────────────────────────────────
// An image captured at a PINNED shared prefix (pin=N): its live ids ARE its deepest (and only)
// checkpoint — the state was cut exactly at the pin boundary. A whole-conversation park NEVER
// has this shape: the live state always runs past every checkpoint by at least the response
// generated after them. Such an entry is a shared asset that many conversations branch from,
// not a branch of one conversation, so the same-conversation cleanups (drop_superseded's prefix
// rules, the reload dedupe) must KEEP it: to them its ids look like "an earlier turn of this
// conversation" — a strict prefix of every conversation sharing the system prompt — and it would
// be deleted by the first park after its creation. The disk LRU also prefers to keep it.
static bool prefix_snapshot(const DiskEntry& e) {
    return !e.ids.empty() && e.ids.size() == e.ckpt_ids.size() &&
           std::equal(e.ids.begin(), e.ids.end(), e.ckpt_ids.begin());
}

// Every protection's one predicate: a shared-prefix asset by EITHER signal — the caller said
// so at write time (e.prefix, the pfx- name), or the entry has the prefix SHAPE (its live ids
// ARE its deepest checkpoint's; a whole-conversation park never does) which also recognizes
// prefix files written by older builds under the conv- name.
static bool is_prefix_entry(const DiskEntry& e) {
    return e.prefix || prefix_snapshot(e);
}

// The file-name prefix of a spill kind: conv- (a whole conversation) or pfx- (a shared
// prefix). reload_from_disk parses the kind back out of the name; everything else flows from
// DiskEntry::prefix.
static const char* kind_name(bool prefix) { return prefix ? "pfx-" : "conv-"; }

bool ConversationDiskStore::holds(const std::vector<int32_t>& ids,
                                  const std::vector<ConversationImageKey>& imgs, bool cvec) const {
    if (!enabled() || ids.empty()) return false;
    flush_pending();
    auto ids_equal = [&](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    auto ids_prefix = [&](const std::vector<int32_t>& pre, const std::vector<int32_t>& full) {
        return pre.size() < full.size() &&
               std::equal(pre.begin(), pre.end(), full.begin());
    };
    auto imgs_prefix = [&](const std::vector<ConversationImageKey>& pre,
                           const std::vector<ConversationImageKey>& full) {
        if (pre.size() > full.size()) return false;
        for (size_t i = 0; i < pre.size(); ++i)
            if (!(pre[i] == full[i])) return false;
        return true;
    };
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.ids.empty()) continue;
        const bool equal = ids_equal(e.ids, ids);
        const bool longer = ids_prefix(ids, e.ids);
        if (!equal && !longer) continue;
        // the conversation's own images must line up with the entry's, in order (the
        // conversation `ids` is the SHORTER or equal side of the branch)
        if (imgs_prefix(imgs, e.imgs)) return true;
    }
    // The client may re-render the history without the assistant's thinking tokens: the
    // entry's deepest checkpoint is then what the next park would write. Equal-length counts
    // (the checkpoint IS the conversation); longer does not apply through ckpt_ids.
    for (const auto& e : entries_) {
        if (e.cvec != cvec || e.ckpt_ids.empty()) continue;
        if (ids_equal(e.ckpt_ids, ids) && imgs_prefix(imgs, e.ckpt_imgs)) return true;
    }
    return false;
}

std::vector<int64_t> ConversationDiskStore::prefix_pin_lengths() const {
    std::vector<int64_t> out;
    if (!enabled()) return out;
    flush_pending();
    for (const auto& e : entries_)
        if (is_prefix_entry(e) && !e.ids.empty()) out.push_back((int64_t) e.ids.size());
    return out;
}

void ConversationDiskStore::flush_pending() const {
    std::unique_lock<std::mutex> lk(writer_mu_);
    writer_cv_.wait(lk, [&] { return pending_.empty() && !writer_busy_; });
}

bool ConversationDiskStore::put_async(SavedConversation&& image, uint64_t serial,
                                      const SessionFileIdentity& id, const SessionWriteOptions& options,
                                      bool prefix) {
    if (!enabled()) return false;
    PendingWrite p;
    p.serial = serial;
    p.id = id;
    p.options = options;
    p.prefix = prefix;
    p.sup_ids = image.live.ids;        // cheap copies for the post-write superseded cleanup
    p.sup_imgs = image.live.imgs;
    p.sup_cvec = image.cvec;
    p.image = std::move(image);
    bool inline_write = false;   // thread creation failed: write on the calling thread
    {
        std::lock_guard<std::mutex> lk(writer_mu_);
        // Queue: at most ONE conversation image plus ONE prefix image (plus the one in the
        // writer's hands). Each queued conversation image is hundreds of MiB of host RAM and
        // Kaggle-class boxes run close to the floor; a prefix snapshot is tens of MiB. The
        // back-to-back parks that overflow the queue are the retry loop's earlier capture of
        // the SAME conversation, superseded by the newer one anyway — but a prefix snapshot's
        // loss is unrecoverable this run (the engine spills once per pin length), so a queued
        // prefix is NEVER dropped for an arriving conversation image: each kind only replaces
        // its own.
        for (auto it = pending_.begin(); it != pending_.end(); ++it)
            if (it->prefix == prefix) {
                ++async_dropped_;
                std::fprintf(stderr, "strata serve: conversation spill: background write queue "
                                     "full; dropping the older queued %s image (%.1f MiB) for "
                                     "this one\n",
                             prefix ? "prefix" : "conversation",
                             (double) it->image.bytes() / (1024.0 * 1024.0));
                pending_.erase(it);
                break;
            }
        if (!writer_.joinable()) {
            try {
                writer_ = std::thread([this] { writer_loop(); });
            } catch (...) {
                // resource exhaustion (a cgroup thread cap): degrade to a synchronous write so
                // the spill still lands; the queue stays empty in this mode
                inline_write = true;
            }
        }
        if (!inline_write) pending_.push_back(std::move(p));
    }
    if (inline_write) {
        // ── Delta path first, exactly as the writer thread does it (the append before the
        // protected cleanup); the thread-start failure that landed here must not change the
        // on-disk shape between the two modes.
        bool ok = false;
        const size_t superseded = [&] {
            if (journal_ && !p.prefix)
                if (auto base = find_delta_base(p.image)) {
                    const uint64_t base_serial = entries_[base->index].serial;
                    if (append_delta(base->index, base->keep_tokens, p.image, p.id)) {
                        ok = true;
                        { SavedConversation discard = std::move(p.image); (void) discard; }
                        return drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec, base_serial);
                    }
                }
            const size_t n = drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec);
            ok = put_impl(std::move(p.image), p.serial, p.id, p.options, p.prefix);
            return n;
        }();
        if (ok && superseded > 0)
            std::fprintf(stderr, "strata serve: conversation spill: write landed (serial %llu) "
                                 "and dropped %zu superseded spill file%s\n",
                         (unsigned long long) p.serial, superseded, superseded == 1 ? "" : "s");
        return ok;
    }
    writer_cv_.notify_all();
    return true;
}

void ConversationDiskStore::writer_loop() {
    std::unique_lock<std::mutex> lk(writer_mu_);
    for (;;) {
        writer_cv_.wait(lk, [&] { return writer_stop_ || !pending_.empty(); });
        if (pending_.empty()) {
            if (writer_stop_) return;   // idle and stopping: exit (the dtor drains leftovers)
            continue;                   // spurious wakeup
        }
        PendingWrite p = std::move(pending_.front());
        pending_.pop_front();
        writer_busy_ = true;
        lk.unlock();
        // ── Delta path first (STRATA_SPILL_JOURNAL=1): a park whose conversation is already
        // on disk as (base + journal) appends ONE record instead of rewriting the .bin. The
        // append runs BEFORE the superseded cleanup — the base entry holds a strict PREFIX of
        // this image's ids, so the cleanup's prefix branch would delete it (Trampa 1); and
        // AFTER the append the entry's ids EQUAL the cleanup's ids, so the exact-match branch
        // would delete it too (Trampa 2) — hence the protect_serial the cleanup gets then.
        // Any anomaly falls through to today's full write, whose cleanup replaces the entry.
        bool ok = false;
        size_t superseded = 0;
        if (journal_ && !p.prefix) {
            if (auto base = find_delta_base(p.image)) {
                const uint64_t base_serial = entries_[base->index].serial;   // copied: the cleanup may erase
                if (append_delta(base->index, base->keep_tokens, p.image, p.id)) {
                    ok = true;
                    superseded = drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec, base_serial);
                    { SavedConversation discard = std::move(p.image); (void) discard; }
                }
            }
        }
        if (!ok) {
            // drop the same-branch older spills FIRST (the RAM cache's order: drop_superseded
            // runs before put). AFTER the write the new entry's ids equal the cleanup's `ids`
            // and the exact-match branch would delete the file we just wrote.
            superseded = drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec);
            ok = put_impl(std::move(p.image), p.serial, p.id, p.options, p.prefix);
        }
        // ── Compaction: only with the queue idle (never on a request's way) and only one
        // candidate per pass — the fold is a full read+rewrite of one conversation. The
        // writer stays busy through it, so a take() that arrives meanwhile waits inside
        // flush_pending() instead of reading a half-compacted entry.
        bool compact = false;
        lk.lock();
        compact = journal_ && pending_.empty() && !writer_stop_;
        if (!compact) {
            writer_busy_ = false;
            if (!ok)
                std::fprintf(stderr, "strata serve: conversation spill: background write FAILED "
                                     "(serial %llu); the conversation stays in RAM only\n",
                             (unsigned long long) p.serial);
            else if (superseded > 0)
                std::fprintf(stderr, "strata serve: conversation spill: background write landed "
                                     "(serial %llu) and dropped %zu superseded spill file%s\n",
                             (unsigned long long) p.serial, superseded, superseded == 1 ? "" : "s");
            writer_cv_.notify_all();   // a flusher may be waiting for this write to land
        }
        lk.unlock();
        if (compact) {
            (void) compact_idle(p.id);
            lk.lock();
            writer_busy_ = false;
            writer_cv_.notify_all();
            lk.unlock();
        }
        lk.lock();
    }
}

ConversationDiskStore::ConversationDiskStore(std::string dir, uint64_t budget_bytes,
                                             const SessionFileIdentity* identity)
    : dir_(std::move(dir)), budget_(budget_bytes) {
    if (identity) { identity_ = *identity; identity_bound_ = true; }
    // The delta journal's flag, read once. DEFAULT ON since the v2 multi-image record: one
    // append per continued turn instead of a full .bin + stage rewrite is strictly better, and
    // the format is exercised by the whole suite. STRATA_SPILL_JOURNAL=0 is the opt-out (the
    // A/B escape for a run that must reproduce the pre-journal on-disk shape exactly); any
    // other value (unset, 1, garbage) leaves the default on.
    journal_ = true;
    if (const char* env = std::getenv("STRATA_SPILL_JOURNAL")) journal_ = std::strcmp(env, "0") != 0;
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
    if (journal_)
        std::fprintf(stderr, "strata serve: conversation spill: delta journal ON (default; "
                             "STRATA_SPILL_JOURNAL=0 disables it): a continued turn appends "
                             "conv-<serial>.dlt instead of rewriting the .bin and its stage files "
                             "(compaction at %llu MiB / %u records)\n",
                     (unsigned long long) (kJournalCompactBytes / (1024ull * 1024)),
                     (unsigned) kJournalCompactRecords);
}

// The dtor DRAINS: the engine's shutdown auto-spill is queued microseconds before main()
// returns, and it is the whole point of that spill — it must land on disk. The writer thread
// processes queued images even after writer_stop_ (it exits only when the queue is empty), so
// join() usually waits for the last write; the direct put_impl below is the belt-and-braces
// path for an image the thread never picked up (a thread-start failure, an exception).
ConversationDiskStore::~ConversationDiskStore() {
    if (writer_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(writer_mu_);
            writer_stop_ = true;
        }
        writer_cv_.notify_all();
        writer_.join();
    }
    while (!pending_.empty()) {
        PendingWrite p = std::move(pending_.front());
        pending_.pop_front();
        // the writer thread's order (delta first, then the protected cleanup; the fallback
        // full write with today's cleanup) — same code, run on the draining thread
        bool ok = false;
        if (journal_ && !p.prefix)
            if (auto base = find_delta_base(p.image)) {
                const uint64_t base_serial = entries_[base->index].serial;
                if (append_delta(base->index, base->keep_tokens, p.image, p.id)) {
                    ok = true;
                    (void) drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec, base_serial);
                    { SavedConversation discard = std::move(p.image); (void) discard; }
                }
            }
        if (!ok) {
            // same order as the writer: the superseded cleanup runs BEFORE the write
            (void) drop_superseded_light(p.sup_ids, p.sup_imgs, p.sup_cvec);
            if (!put_impl(std::move(p.image), p.serial, p.id, p.options, p.prefix))
                std::fprintf(stderr, "strata serve: conversation spill: shutdown drain write FAILED "
                                     "(serial %llu)\n", (unsigned long long) p.serial);
        }
    }
    // Shutdown compaction (the plan's third trigger): fold every journal past its bounds back
    // into its .bin, so the next run restores from one file per conversation. Fail-soft — a
    // journal left over compacts on the next run's idle pass instead. Needs the identity this
    // run's writes carried (the engine passes it per call; remembered in put_impl/append_delta
    // and at reload) — without one there is nothing this run owns on disk to fold.
    if (journal_ && write_id_set_) compact_all();
}

// Not movable since Phase 2 (the writer thread + mutex + cv); no call site ever moved the
// store by value — everything holds it as unique_ptr.

std::string ConversationDiskStore::path_for(uint64_t serial, bool prefix) const {
    return dir_ + "/" + kind_name(prefix) + std::to_string(serial) + ".bin";
}

std::string ConversationDiskStore::path_for_dlt(uint64_t serial, bool prefix) const {
    return dir_ + "/" + kind_name(prefix) + std::to_string(serial) + ".dlt";
}

// ── Stage files (a layer split's spill) ─────────────────────────────────────────────────────
// Each later stage's image is a self-contained SavedConversation (its own layer carve and K/V),
// so each one is written as its own v1 session file next to the main .bin. The main .bin is the
// commit point: it is renamed into place LAST, so a crash mid-put leaves orphan stage files
// without a main (cleaned by reload_from_disk), never a main whose stages are missing.
std::string ConversationDiskStore::path_for_stage(uint64_t serial, size_t k, bool prefix) const {
    return dir_ + "/" + kind_name(prefix) + std::to_string(serial) + ".stage" + std::to_string(k) + ".bin";
}

uint64_t ConversationDiskStore::stage_files_bytes(uint64_t serial, bool prefix) const {
    uint64_t total = 0;
    for (size_t k = 0; ; ++k) {
        std::error_code ec;
        const std::string spath = path_for_stage(serial, k, prefix);
        if (!fs::exists(spath, ec) || ec) break;
        const uint64_t sb = fs::file_size(spath, ec);
        if (ec) break;
        total += sb;
    }
    return total;
}

// ── Sidecar .idx: cheap manifest metadata without parsing the .bin ──────────────────────────
// The .idx is a small (few hundred bytes) sidecar file written next to each conv-<serial>.bin.
// It carries exactly the fields DiskEntry needs (ids, imgs, geometry, layer_lo/hi, cvec,
// file_bytes) plus a freshness hint (bin_size, bin_mtime_ns) so reload_from_disk() can decide
// whether to trust the .idx or fall back to a full session_file_read of the .bin. The layout is
// little-endian (matching the .bin's v1 format). Writes are atomic (temp + rename), reads never
// throw — a corrupt or stale .idx degrades to the slow path, never aborts the engine start.

namespace {

// Magic + version. Bumping the version invalidates every old .idx (the slow path rewrites them).
constexpr char kIdxMagic[4] = {'S', 'I', 'D', 'X'};
constexpr uint32_t kIdxVersion = 4;   // v4: adds the stage count + each stage file's carve

// Layout (little-endian, no struct memcpy — explicit byte writes):
//   4   magic "SIDX"
//   4   uint32 version
//   8   uint64 model        (SessionFileIdentity.model)
//   8   uint64 config       (SessionFileIdentity.config)
//   144 18 × int64 geometry (std::array<int64_t, 18>)
//   8   int64  layer_lo
//   8   int64  layer_hi
//   1   uint8  cvec
//   8   uint64 file_bytes    (the entry's disk footprint: .bin + stages + journal)
//   8   uint64 bin_size      (fs::file_size of the .bin at write time — freshness check)
//   8   int64  bin_mtime_ns  (fs::last_write_time as nanoseconds since epoch — freshness check)
//   8   uint64 journal_size    (the .dlt's size at write time; 0: none — freshness check)
//   8   int64  journal_mtime_ns (the .dlt's mtime; 0 when there is none)
//   4   uint32 journal_records (the compaction trigger's unit)
//   4   uint32 n_stages       (the layer split's stage files this entry owns)
//   8 × n_stages int64 stage_layer_lo[]  (each stage file's carve — delta-base matching)
//   8 × n_stages int64 stage_layer_hi[]
//   4   uint32 n_ids
//   n_ids × 4   int32 ids[]
//   4   uint32 n_imgs
//   n_imgs × 16 (int64 start, uint64 hash) imgs[]
//   8   uint64 checksum      (session_hash64 of everything above)

void write_le_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(uint8_t(v & 0xFF));
    buf.push_back(uint8_t((v >> 8) & 0xFF));
    buf.push_back(uint8_t((v >> 16) & 0xFF));
    buf.push_back(uint8_t((v >> 24) & 0xFF));
}
void write_le_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(uint8_t((v >> (i * 8)) & 0xFF));
}
void write_le_i64(std::vector<uint8_t>& buf, int64_t v) { write_le_u64(buf, uint64_t(v)); }
void write_le_u8(std::vector<uint8_t>& buf, uint8_t v) { buf.push_back(v); }

bool read_le_u32(const std::vector<uint8_t>& buf, size_t& pos, uint32_t& out) {
    if (pos + 4 > buf.size()) return false;
    out = uint32_t(buf[pos]) | (uint32_t(buf[pos + 1]) << 8) |
          (uint32_t(buf[pos + 2]) << 16) | (uint32_t(buf[pos + 3]) << 24);
    pos += 4;
    return true;
}
bool read_le_u64(const std::vector<uint8_t>& buf, size_t& pos, uint64_t& out) {
    if (pos + 8 > buf.size()) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= uint64_t(buf[pos + i]) << (i * 8);
    pos += 8;
    return true;
}
bool read_le_i64(const std::vector<uint8_t>& buf, size_t& pos, int64_t& out) {
    uint64_t u = 0;
    if (!read_le_u64(buf, pos, u)) return false;
    out = int64_t(u);
    return true;
}
bool read_le_u8(const std::vector<uint8_t>& buf, size_t& pos, uint8_t& out) {
    if (pos + 1 > buf.size()) return false;
    out = buf[pos++];
    return true;
}

// Build the .idx payload (everything except the trailing checksum) into a byte buffer.
std::vector<uint8_t> build_idx_payload(const DiskEntry& e, const SessionFileIdentity& id,
                                       uint64_t bin_size, int64_t bin_mtime_ns,
                                       uint64_t journal_size, int64_t journal_mtime_ns) {
    std::vector<uint8_t> buf;
    buf.reserve(256 + e.ids.size() * 4 + e.imgs.size() * 16 + e.ckpt_ids.size() * 4 + e.ckpt_imgs.size() * 16);
    for (int i = 0; i < 4; ++i) buf.push_back(uint8_t(kIdxMagic[i]));
    write_le_u32(buf, kIdxVersion);
    write_le_u64(buf, id.model);
    write_le_u64(buf, id.config);
    for (size_t i = 0; i < e.geometry.size(); ++i) write_le_i64(buf, e.geometry[i]);
    write_le_i64(buf, e.layer_lo);
    write_le_i64(buf, e.layer_hi);
    write_le_u8(buf, e.cvec ? 1 : 0);
    write_le_u64(buf, e.file_bytes);
    write_le_u64(buf, bin_size);
    write_le_i64(buf, bin_mtime_ns);
    // the journal's freshness hint: what the .dlt was when this .idx was written. A later
    // append changes the .dlt (and then rewrites this .idx); a mismatch means stale.
    write_le_u64(buf, journal_size);
    write_le_i64(buf, journal_mtime_ns);
    write_le_u32(buf, e.journal_records);
    // the layer split's stage files (v4): their count and carves, so find_delta_base can match
    // a new capture's stage images without parsing the stage .bins
    write_le_u32(buf, e.n_stages);
    for (int64_t v : e.stage_layer_lo) write_le_i64(buf, v);
    for (int64_t v : e.stage_layer_hi) write_le_i64(buf, v);
    // live ids + imgs
    write_le_u32(buf, uint32_t(e.ids.size()));
    for (int32_t v : e.ids) write_le_u32(buf, uint32_t(v));
    write_le_u32(buf, uint32_t(e.imgs.size()));
    for (const auto& im : e.imgs) {
        write_le_i64(buf, im.start);
        write_le_u64(buf, im.hash);
    }
    // checkpoint ids + imgs (the turn boundary — what the next request's prompt actually matches)
    write_le_u32(buf, uint32_t(e.ckpt_ids.size()));
    for (int32_t v : e.ckpt_ids) write_le_u32(buf, uint32_t(v));
    write_le_u32(buf, uint32_t(e.ckpt_imgs.size()));
    for (const auto& im : e.ckpt_imgs) {
        write_le_i64(buf, im.start);
        write_le_u64(buf, im.hash);
    }
    return buf;
}

// Compute the checksum over the payload (everything except the trailing 8-byte checksum field).
uint64_t idx_checksum(const std::vector<uint8_t>& payload) {
    // FNV-1a 64-bit (non-cryptographic, same role as session_hash64 in the .bin format).
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : payload) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

}   // anonymous namespace

std::string ConversationDiskStore::path_for_idx(uint64_t serial, bool prefix) const {
    return dir_ + "/" + kind_name(prefix) + std::to_string(serial) + ".idx";
}

bool ConversationDiskStore::write_idx(const DiskEntry& e, const std::string& bin_path,
                                       const SessionFileIdentity& id) const {
    // Read the .bin's size and mtime for the freshness hint. If either fails, the .idx is
    // written with zeros — reload_from_disk() will fall back to the slow path on the next start.
    uint64_t bin_size = 0;
    int64_t bin_mtime_ns = 0;
    std::error_code ec;
    if (fs::exists(bin_path, ec) && !ec) {
        bin_size = (uint64_t) fs::file_size(bin_path, ec);
        if (ec) bin_size = 0;
        const fs::file_time_type ft = fs::last_write_time(bin_path, ec);
        if (!ec) {
            // file_clock → system_clock → nanoseconds since epoch (C++20 chrono).
            const auto sys = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            bin_mtime_ns = sys.time_since_epoch().count();
        }
    }
    // The journal's freshness hint, the same way: the .dlt must still be exactly what it was
    // when this .idx is written (an append lands BEFORE the .idx refresh, so the pair is
    // always in this order on disk).
    uint64_t journal_size = 0;
    int64_t journal_mtime_ns = 0;
    {
        std::string dlt_path = bin_path;
        if (dlt_path.size() > 4 && dlt_path.compare(dlt_path.size() - 4, 4, ".bin") == 0)
            dlt_path.replace(dlt_path.size() - 4, 4, ".dlt");
        else dlt_path = path_for_dlt(e.serial, e.prefix);
        std::error_code ec2;
        if (fs::exists(dlt_path, ec2) && !ec2) {
            journal_size = (uint64_t) fs::file_size(dlt_path, ec2);
            if (ec2) journal_size = 0;
            const fs::file_time_type ft = fs::last_write_time(dlt_path, ec2);
            if (!ec2) {
                const auto sys = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                    ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                journal_mtime_ns = sys.time_since_epoch().count();
            }
        }
    }
    // Build the payload (everything except the checksum).
    std::vector<uint8_t> payload = build_idx_payload(e, id, bin_size, bin_mtime_ns,
                                                     journal_size, journal_mtime_ns);
    // Append the checksum.
    const uint64_t cs = idx_checksum(payload);
    write_le_u64(payload, cs);
    // Atomic write: temp + rename. The temp file is <idx>.tmp; on any error it is removed.
    // The .idx path is derived from the .bin path (same name, .idx for .bin), so a prefix
    // snapshot's sidecar lands next to its pfx-<serial>.bin as pfx-<serial>.idx.
    std::string idx_path = bin_path;
    if (idx_path.size() > 4 && idx_path.compare(idx_path.size() - 4, 4, ".bin") == 0) {
        idx_path.replace(idx_path.size() - 4, 4, ".idx");
    } else {
        idx_path = path_for_idx(e.serial, e.prefix);   // defensive fallback: derive from the kind
    }
    const std::string tmp_path = idx_path + ".tmp";
    try {
        std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
        if (!f) return false;
        const size_t n = std::fwrite(payload.data(), 1, payload.size(), f);
        std::fclose(f);
        if (n != payload.size()) {
            std::error_code ec2;
            fs::remove(tmp_path, ec2);
            return false;
        }
        std::error_code ren_ec;
        fs::rename(tmp_path, idx_path, ren_ec);
        if (ren_ec) {
            std::error_code ec2;
            fs::remove(tmp_path, ec2);
            return false;
        }
        return true;
    } catch (const std::exception&) {
        std::error_code ec2;
        fs::remove(tmp_path, ec2);
        return false;
    }
}

bool ConversationDiskStore::read_idx(const std::string& idx_path, DiskEntry& out,
                                      const SessionFileIdentity& id) const {
    // Read the whole .idx into a buffer. Never throws — any I/O or parse error returns false.
    std::error_code ec;
    if (!fs::exists(idx_path, ec) || ec) return false;
    const uint64_t fsize = fs::file_size(idx_path, ec);
    if (ec || fsize == 0 || fsize > (1u << 20)) return false;   // sanity bound: 1 MiB max
    std::vector<uint8_t> buf;
    try {
        buf.resize(size_t(fsize));
    } catch (const std::exception&) {
        return false;
    }
    std::FILE* f = std::fopen(idx_path.c_str(), "rb");
    if (!f) return false;
    const size_t n = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    if (n != buf.size()) return false;
    // Parse magic + version.
    size_t pos = 0;
    if (buf.size() < 4 + 4 + 8 + 8) return false;   // header minimum
    for (int i = 0; i < 4; ++i) if (buf[i] != uint8_t(kIdxMagic[i])) return false;
    pos = 4;
    uint32_t version = 0;
    if (!read_le_u32(buf, pos, version) || version != kIdxVersion) return false;
    // Identity check: a .idx from another engine is unusable.
    uint64_t model = 0, config = 0;
    if (!read_le_u64(buf, pos, model) || !read_le_u64(buf, pos, config)) return false;
    if (model != id.model || config != id.config) return false;
    // Parse the rest of the fixed-size header.
    DiskEntry e;
    for (size_t i = 0; i < e.geometry.size(); ++i) {
        if (!read_le_i64(buf, pos, e.geometry[i])) return false;
    }
    if (!read_le_i64(buf, pos, e.layer_lo) || !read_le_i64(buf, pos, e.layer_hi)) return false;
    uint8_t cvec_byte = 0;
    if (!read_le_u8(buf, pos, cvec_byte)) return false;
    e.cvec = (cvec_byte != 0);
    if (!read_le_u64(buf, pos, e.file_bytes)) return false;
    uint64_t bin_size = 0;
    int64_t bin_mtime_ns = 0;
    if (!read_le_u64(buf, pos, bin_size) || !read_le_i64(buf, pos, bin_mtime_ns)) return false;
    // v3: the journal's recorded freshness + record count.
    uint64_t journal_size = 0;
    int64_t journal_mtime_ns = 0;
    if (!read_le_u64(buf, pos, journal_size) || !read_le_i64(buf, pos, journal_mtime_ns)) return false;
    uint32_t journal_records = 0;
    if (!read_le_u32(buf, pos, journal_records)) return false;
    // v4: the stage files' count and carves.
    uint32_t n_stages = 0;
    if (!read_le_u32(buf, pos, n_stages) || n_stages > 64) return false;
    try {
        e.stage_layer_lo.reserve(n_stages);
        e.stage_layer_hi.reserve(n_stages);
    } catch (const std::exception&) {
        return false;
    }
    for (uint32_t i = 0; i < n_stages; ++i) {
        int64_t lo = 0, hi = 0;
        if (!read_le_i64(buf, pos, lo) || !read_le_i64(buf, pos, hi)) return false;
        e.stage_layer_lo.push_back(lo);
        e.stage_layer_hi.push_back(hi);
    }
    e.n_stages = n_stages;
    // Parse ids.
    uint32_t n_ids = 0;
    if (!read_le_u32(buf, pos, n_ids) || n_ids > (1u << 28)) return false;   // sanity bound
    try {
        e.ids.reserve(n_ids);
    } catch (const std::exception&) {
        return false;
    }
    for (uint32_t i = 0; i < n_ids; ++i) {
        uint32_t v = 0;
        if (!read_le_u32(buf, pos, v)) return false;
        e.ids.push_back(int32_t(v));
    }
    // Parse imgs.
    uint32_t n_imgs = 0;
    if (!read_le_u32(buf, pos, n_imgs) || n_imgs > (1u << 20)) return false;
    try {
        e.imgs.reserve(n_imgs);
    } catch (const std::exception&) {
        return false;
    }
    for (uint32_t i = 0; i < n_imgs; ++i) {
        int64_t start = 0;
        uint64_t hash = 0;
        if (!read_le_i64(buf, pos, start) || !read_le_u64(buf, pos, hash)) return false;
        e.imgs.push_back(ConversationImageKey{start, hash});
    }
    // Read checkpoint ids + imgs (the turn boundary).
    {
        uint32_t n_ckpt_ids = 0;
        if (!read_le_u32(buf, pos, n_ckpt_ids) || n_ckpt_ids > (1u << 28)) return false;
        try { e.ckpt_ids.reserve(n_ckpt_ids); } catch (const std::exception&) { return false; }
        for (uint32_t i = 0; i < n_ckpt_ids; ++i) {
            uint32_t v = 0;
            if (!read_le_u32(buf, pos, v)) return false;
            e.ckpt_ids.push_back(int32_t(v));
        }
        uint32_t n_ckpt_imgs = 0;
        if (!read_le_u32(buf, pos, n_ckpt_imgs) || n_ckpt_imgs > (1u << 20)) return false;
        try { e.ckpt_imgs.reserve(n_ckpt_imgs); } catch (const std::exception&) { return false; }
        for (uint32_t i = 0; i < n_ckpt_imgs; ++i) {
            int64_t start = 0;
            uint64_t hash = 0;
            if (!read_le_i64(buf, pos, start) || !read_le_u64(buf, pos, hash)) return false;
            e.ckpt_imgs.push_back(ConversationImageKey{start, hash});
        }
    }
    // Checksum: the last 8 bytes of the buffer must equal idx_checksum(everything before).
    if (pos + 8 > buf.size()) return false;
    const uint64_t stored_cs = [&] {
        uint64_t u = 0;
        for (int i = 0; i < 8; ++i) u |= uint64_t(buf[pos + i]) << (i * 8);
        return u;
    }();
    buf.resize(pos);   // drop the checksum bytes before hashing
    const uint64_t computed_cs = idx_checksum(buf);
    if (stored_cs != computed_cs) return false;
    // Fill in the rest of DiskEntry.
    e.serial = 0;   // caller (reload_from_disk) sets it from the filename
    e.path = std::string();   // caller sets it to the .bin path
    e.used = 0;   // caller assigns next_used_++
    // Freshness check: verify the .bin on disk right now matches what the .idx recorded.
    // If size or mtime differ (the .bin was rewritten or restored), force the slow path.
    const std::string bin_path = idx_path.substr(0, idx_path.size() - 4) + ".bin";   // .idx → .bin
    if (!fs::exists(bin_path, ec) || ec) return false;   // .bin gone → .idx is orphan, ignore
    const uint64_t actual_bin_size = fs::file_size(bin_path, ec);
    if (ec || actual_bin_size != bin_size) return false;   // size mismatch → fallback
    const fs::file_time_type actual_ft = fs::last_write_time(bin_path, ec);
    if (ec) return false;
    const auto actual_sys = std::chrono::time_point_cast<std::chrono::nanoseconds>(
        actual_ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    const int64_t actual_mtime_ns = actual_sys.time_since_epoch().count();
    if (actual_mtime_ns != bin_mtime_ns) return false;   // mtime mismatch → fallback
    // The journal must match too: the manifest data in this .idx describes base+journal, so a
    // .dlt that changed since (an append whose .idx refresh never landed) makes it stale. An
    // absent .dlt reads as (0, 0) — a recorded journal_size > 0 with no file is a mismatch.
    {
        const std::string dlt_path = bin_path.substr(0, bin_path.size() - 4) + ".dlt";   // .bin → .dlt
        uint64_t actual_journal_size = 0;
        int64_t actual_journal_mtime_ns = 0;
        if (fs::exists(dlt_path, ec) && !ec) {
            actual_journal_size = (uint64_t) fs::file_size(dlt_path, ec);
            if (ec) actual_journal_size = 0;
            const fs::file_time_type jft = fs::last_write_time(dlt_path, ec);
            if (!ec) {
                const auto jsys = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                    jft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                actual_journal_mtime_ns = jsys.time_since_epoch().count();
            }
        }
        if (actual_journal_size != journal_size || actual_journal_mtime_ns != journal_mtime_ns)
            return false;   // the .dlt changed (or went missing) since this .idx → fallback
    }
    e.journal_bytes = journal_size;
    e.journal_records = journal_records;
    out = std::move(e);
    out.path = bin_path;
    return true;
}

void ConversationDiskStore::remove_spill_files(const std::string& bin_path) {
    // Best-effort removal of a spill's every file: .bin, .idx, .idx.tmp, and a layer split's
    // conv-<serial>.stage<k>.bin (+ .tmp). The .bin path is the input; the .idx path replaces
    // the .bin suffix with .idx (NOT .bin + ".idx"). Any of them may not exist; all errors are
    // swallowed. Used by every site that today does fs::remove(.bin).
    std::error_code ec;
    fs::remove(bin_path, ec);
    // Derive conv-<serial>.idx from conv-<serial>.bin (replace the last 4 chars ".bin" with ".idx").
    std::string idx_path = bin_path;
    if (idx_path.size() > 4 && idx_path.compare(idx_path.size() - 4, 4, ".bin") == 0) {
        idx_path.replace(idx_path.size() - 4, 4, ".idx");
    }
    fs::remove(idx_path, ec);
    fs::remove(idx_path + ".tmp", ec);
    // Stage files: everything sharing the serial's prefix "conv-<serial>." or "pfx-<serial>." —
    // the '.' right after the serial keeps it from matching a longer serial (conv-808. never
    // matches conv-8081.bin). A directory scan is fine: spill dirs hold a few dozen files and
    // removals are rare. Both kinds can carry stage files: a layer split's conversation AND its
    // shared-prefix snapshot.
    const size_t slash = bin_path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string() : bin_path.substr(0, slash);
    const std::string fname = slash == std::string::npos ? bin_path : bin_path.substr(slash + 1);
    const bool kind_ok = !dir.empty() && fname.size() > 4 &&
        fname.compare(fname.size() - 4, 4, ".bin") == 0 &&
        (fname.rfind("conv-", 0) == 0 || fname.rfind("pfx-", 0) == 0);
    if (kind_ok) {
        const std::string prefix = fname.substr(0, fname.size() - 4) + '.';   // "conv-<serial>."
        std::error_code it_ec;
        for (const auto& entry : fs::directory_iterator(dir, it_ec)) {
            if (it_ec) break;
            const std::string name = entry.path().filename().string();
            if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0)
                fs::remove(entry.path(), ec);
        }
    }
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
                                const SessionFileIdentity& id, const SessionWriteOptions& options,
                                bool prefix) {
    if (!enabled()) return false;
    flush_pending();   // never two writers at once; the sync path writes on the calling thread
    return put_impl(std::move(image), serial, id, options, prefix);
}

bool ConversationDiskStore::put_impl(SavedConversation&& image, uint64_t serial,
                                const SessionFileIdentity& id, const SessionWriteOptions& options,
                                bool prefix) {
    if (!enabled()) return false;
    write_id_ = id;   // the shutdown compaction re-reads what this run wrote
    write_id_set_ = true;
    // A prefix write whose exact prefix is already held is a SUCCESSFUL no-op: the engine spills
    // a pin once per run, but after a restart it does not know a previous run already wrote the
    // same file, and two identical pfx-<serial>.bin files (one per run) would waste the budget
    // forever — the prefix protections never clean a duplicate. (Runs on the writer thread for
    // put_async: entries_ is only mutated between flushes, which is this thread's own window.)
    if (prefix) {
        for (const DiskEntry& held : entries_) {
            if (!is_prefix_entry(held) || held.cvec != image.cvec || held.ids != image.live.ids ||
                held.imgs != image.live.imgs)
                continue;
            std::fprintf(stderr, "strata serve: prefix cache: the disk already holds these %zu "
                                 "shared tokens (%s) — pfx-%llu.bin not written again\n",
                         held.ids.size(), held.path.c_str(), (unsigned long long) serial);
            return true;
        }
    }
    // If this serial is already spilled (re-park of a conversation that came back from disk
    // and got evicted again), replace it: delete the old files (.bin + .idx + .dlt) and re-add
    // the entry — but ONLY when the holder is this same conversation (a branch of the incoming
    // image). A DIFFERENT conversation under a colliding serial (the RAM cache's park counter
    // restarts at 1 every engine start, while reloaded spills keep their old serials) must keep
    // its files: the new image is written under a fresh serial instead, lifted above every
    // serial the manifest has ever seen. Without the guard, the first park after a restart
    // destroyed a reloaded conversation's spill — and with journals, its delta base.
    uint64_t use_serial = serial;
    if (DiskEntry* existing = find_by_serial(serial)) {
        auto ids_prefix = [](const std::vector<int32_t>& cid, const std::vector<int32_t>& full) {
            return !cid.empty() && cid.size() <= full.size() &&
                   std::equal(cid.begin(), cid.end(), full.begin());
        };
        auto imgs_contained = [](const std::vector<ConversationImageKey>& from,
                                 const std::vector<ConversationImageKey>& into) {
            if (from.size() > into.size()) return false;
            size_t k = 0;
            for (size_t a = 0; a < into.size() && k < from.size(); ++a)
                if (into[a] == from[k]) ++k;
            return k == from.size();
        };
        const bool branch =
            (ids_prefix(existing->ids, image.live.ids) && imgs_contained(existing->imgs, image.live.imgs)) ||
            (!existing->ckpt_ids.empty() && ids_prefix(existing->ckpt_ids, image.live.ids) &&
             imgs_contained(existing->ckpt_imgs, image.live.imgs));
        if (!branch) {
            use_serial = next_fresh_serial_++;
            while (find_by_serial(use_serial) != nullptr) use_serial = next_fresh_serial_++;
            std::fprintf(stderr, "strata serve: conversation spill: serial %llu is held by another "
                                 "conversation; writing under the fresh serial %llu instead\n",
                         (unsigned long long) serial, (unsigned long long) use_serial);
            existing = nullptr;
        } else {
            std::error_code ec;
            const std::string old_bin = existing->path;
            const uint64_t old_bytes = existing->file_bytes;
            const size_t idx = static_cast<size_t>(existing - &entries_.front());
            erase(idx);   // invalidates `existing`
            bytes_ -= old_bytes;
            remove_spill_files(old_bin);
        }
    }
    if (use_serial >= next_fresh_serial_) next_fresh_serial_ = use_serial + 1;
    // Capture the cheap manifest data before we move the image out (kv + checkpoints live on).
    // We mirror what ConversationCache's best() matches against: the live state and the deepest
    // checkpoint. session_checkpoints_to_save keeps the deepest one when writing, so the file's
    // image.checkpoints will hold it (and best() below consults both).
    DiskEntry e;
    e.serial = use_serial;
    e.prefix = prefix;
    e.path = path_for(use_serial, prefix);
    e.ids = image.live.ids;                  // copy: the manifest keeps these in RAM
    e.imgs = image.live.imgs;
    // Capture the deepest checkpoint's ids/imgs too — this is the turn boundary that the next
    // request's prompt actually matches (the client re-renders the history without the
    // assistant's thinking tokens, so live.ids diverges after the reasoning; ckpt_ids stops
    // at the <|im_start|> before the new turn, which is what the prompt contains).
    if (!image.checkpoints.empty()) {
        const auto& deepest = image.checkpoints.back();
        e.ckpt_ids = deepest.ids;
        e.ckpt_imgs = deepest.imgs;
    }
    e.geometry = image.geometry;
    e.layer_lo = image.layer_lo;
    e.layer_hi = image.layer_hi;
    e.cvec = image.cvec;
    // The stage files this put is about to write (conv-<serial>.stage<k>.bin) and their layer
    // carves — what find_delta_base matches a later capture's stage images against.
    e.n_stages = uint32_t(image.stage_images.size());
    e.stage_layer_lo.clear();
    e.stage_layer_hi.clear();
    for (const auto& st : image.stage_images) {
        e.stage_layer_lo.push_back(st.layer_lo);
        e.stage_layer_hi.push_back(st.layer_hi);
    }

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
    // ── Stage files (a layer split's later stages), written BEFORE the main .bin. The main is
    // the put's commit point: renamed into place last, so a crash mid-put leaves orphan stage
    // files without a main (cleaned by reload_from_disk), never a published main whose stages
    // are missing — an entry that would restore as 0 stages and be discarded by the engine.
    // On a stage write failure nothing is published: remove this attempt's stage files.
    const size_t n_stages = image.stage_images.size();
    uint64_t stage_bytes = 0;
    for (size_t k = 0; k < n_stages; ++k) {
        size_t s_written = 0;
        std::string s_error;
        SessionStatus s_status;
        const std::string spath = path_for_stage(serial, k, prefix);
        if (!session_file_write(spath, image.stage_images[k], id, s_written, s_error, opts, &s_status)) {
            std::fprintf(stderr, "strata serve: conversation spill: stage %zu/%zu file write failed "
                                 "(%s); spill aborted, nothing published\n",
                         k, n_stages, s_error.c_str());
            std::error_code ec;
            for (size_t j = 0; j <= k; ++j) {
                fs::remove(path_for_stage(serial, j, prefix), ec);          // the ones that published
                fs::remove(path_for_stage(serial, j, prefix) + ".tmp", ec); // the one that failed
            }
            return false;
        }
        stage_bytes += s_written;
    }
    const bool ok = session_file_write(e.path, image, id, written, error, opts, &status);
    // image is consumed conceptually by the write (it reads its K/V); we move it to discard.
    // The disk store does NOT keep a copy in RAM — that's the whole point of the disk tier.
    { SavedConversation discard = std::move(image); (void) discard; }
    if (!ok) {
        // Disk full / I/O error / RAM refusal (shouldn't happen on the write side, but preflight
        // can refuse on min_free_bytes). The file was never published (rename did not happen).
        // Clean up any leftover temp file and the stage files just written (their main never
        // published, so they would be orphans).
        std::error_code ec;
        fs::remove(e.path + ".tmp", ec);   // best-effort
        for (size_t k = 0; k < n_stages; ++k) {
            fs::remove(path_for_stage(serial, k, prefix), ec);
            fs::remove(path_for_stage(serial, k, prefix) + ".tmp", ec);
        }
        return false;
    }
    // The entry's disk footprint is the main .bin plus its stage files — the budget and the LRU
    // must see the true cost of the spill (a layer split doubles it).
    e.file_bytes = written + stage_bytes;
    e.used = next_used_++;
    bytes_ += e.file_bytes;   // main + stage files: erase()/evict_oldest() subtract the same total
    entries_.push_back(std::move(e));
    serials_.push_back(serial);
    ++writes_;
    // Sidecar .idx: write the manifest metadata next to the .bin so the next start can load
    // the entry without parsing the full .bin. Best-effort: on failure, log and move on — the
    // .bin is valid, the slow path (full parse) takes over on the next start and rewrites the
    // .idx then. The .idx is written AFTER the .bin is published (rename done), so a half-written
    // .idx never points to a non-existent .bin (it would point to the previous .bin's path,
    // but write_idx re-stats the .bin and bakes its size/mtime into the .idx, so the freshness
    // check on reload catches any mismatch and falls back to the slow path).
    {
        // entries_.back() is the entry we just pushed (e was moved). We need its data for write_idx.
        const DiskEntry& just_written = entries_.back();
        if (!write_idx(just_written, just_written.path, id)) {
            std::fprintf(stderr, "strata serve: conversation spill: warning: could not write "
                                 "%s.idx (the .bin is fine; the next start will parse it fully)\n",
                         just_written.path.c_str());
        }
    }
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
    flush_pending();   // a queued spill is not readable yet; wait for the writer to land it
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
        // NUNCA borrar el .bin aquí tampoco. Solo se borra en take() exitoso (la conversación
        // volvió a RAM) o por evict_oldest() (budget excedido). Si la lectura falla, loguear
        // y conservar el archivo para diagnóstico.
        std::fprintf(stderr, "[strata diag] spill take: SKIP %s (read failed, KEPT on disk): "
                             "error=%s, status=%d\n",
                     path.c_str(), error.c_str(), (int) status.error);
        // Remove from manifest (the entry is not usable right now), but DON'T delete the file.
        for (size_t i = 0; i < serials_.size(); ++i) {
            if (serials_[i] == m.serial) { bytes_ -= entries_[i].file_bytes; erase(i); break; }
        }
        return std::nullopt;
    }
    // ── Stage files: read back a layer split's later stages (conv-<serial>.stage<k>.bin for a
    // conversation, pfx-<serial>.stage<k>.bin for a shared prefix; k=0..). Stop at the first
    // MISSING k (a single-GPU engine's spill has none). A stage file that exists but fails to
    // read fails the whole take — a partial image would restore as fewer stages than the engine
    // has and be discarded by its stage count check anyway; keep the files (same policy as a
    // failed main read) and let the engine recompute.
    const bool entry_pfx = is_prefix_entry(*e);
    for (size_t k = 0; ; ++k) {
        std::error_code ec;
        const std::string spath = path_for_stage(m.serial, k, entry_pfx);
        if (!fs::exists(spath, ec) || ec) break;
        SavedConversation stage_img;
        size_t s_read = 0;
        std::string s_error;
        SessionStatus s_status;
        if (!session_file_read(spath, id, stage_img, s_read, s_error, limits, &s_status)) {
            std::fprintf(stderr, "[strata diag] spill take: SKIP %s (stage %zu read failed, KEPT on disk): "
                                 "error=%s, status=%d\n",
                         spath.c_str(), k, s_error.c_str(), (int) s_status.error);
            // Same as a failed main read: remove from the manifest (not usable right now),
            // keep every file, return nothing.
            for (size_t i = 0; i < serials_.size(); ++i) {
                if (serials_[i] == m.serial) { bytes_ -= entries_[i].file_bytes; erase(i); break; }
            }
            return std::nullopt;
        }
        back.stage_images.push_back(std::move(stage_img));
    }
    // ── Delta journal: apply the records on top of the base .bin (single pass, hash-checked,
    // chain-checked). A torn or unchainable tail truncates the file at the last good record;
    // a journal whose FIRST record does not chain onto this base is junk for it (a compaction
    // that crashed after publishing the new .bin, or a .bin replaced under a reused serial) —
    // dropped whole.
    DeltaFold fold;   // had_journal=false without a .dlt: the correction below is then a no-op
    if (!entry_pfx) {
        const std::string dlt = path_for_dlt(m.serial, false);
        std::error_code ec;
        if (fs::exists(dlt, ec) && !ec) {
            fold = conversation_delta_apply(dlt, back);
            if (fold.failed) {
                // a mid-apply I/O error: the image is suspect, exactly a corrupt .bin read.
                // The entry leaves the manifest (not usable right now); every file stays.
                std::fprintf(stderr, "[strata diag] spill take: journal conv-%llu.dlt apply failed "
                                     "(I/O); entry dropped from the manifest, files KEPT\n",
                             (unsigned long long) m.serial);
                for (size_t i = 0; i < serials_.size(); ++i) {
                    if (serials_[i] == m.serial) { bytes_ -= entries_[i].file_bytes; erase(i); break; }
                }
                return std::nullopt;
            }
            if (fold.dropped) {
                fs::remove(dlt, ec);
                std::fprintf(stderr, "strata serve: conversation spill: journal conv-%llu.dlt does not "
                                     "chain onto its base — dropped (the base restores alone)\n",
                             (unsigned long long) m.serial);
            } else if (fold.torn) {
                fs::resize_file(dlt, fold.valid_end, ec);
                std::fprintf(stderr, "strata serve: conversation spill: journal conv-%llu.dlt truncated at "
                                     "%llu B (torn or unchainable tail; %zu record%s applied)\n",
                             (unsigned long long) m.serial, (unsigned long long) fold.valid_end,
                             fold.applied, fold.applied == 1 ? "" : "s");
            }
        }
    }
    // Correct the manifest entry to the state the disk actually restores now, and bump its LRU
    // stamp: the entry STAYS in the manifest (the (base + journal) pair is what the next park
    // of this conversation appends its delta to, and what holds() sees to skip re-capturing an
    // unchanged one — the retry case). Without a journal the correction is a no-op except the
    // stamp. The RAM cache's own copy shadows it for matching while it is parked; the disk LRU
    // still governs it under the budget, and drop_superseded still replaces it when a newer
    // spill of the same conversation lands. A shared-prefix entry was never consumed either
    // (many conversations branch from it).
    for (size_t i = 0; i < serials_.size(); ++i) {
        if (serials_[i] != m.serial) continue;
        DiskEntry& en = entries_[i];
        const uint64_t old_bytes = en.file_bytes;
        en.ids = back.live.ids;
        en.imgs = back.live.imgs;
        if (!back.checkpoints.empty()) {
            const auto& deepest = back.checkpoints.back();
            en.ckpt_ids = deepest.ids;
            en.ckpt_imgs = deepest.imgs;
        } else {
            en.ckpt_ids.clear();
            en.ckpt_imgs.clear();
        }
        en.journal_bytes = fold.dropped ? 0 : fold.journal_bytes;
        en.journal_records = fold.dropped ? 0 : fold.records;
        en.file_bytes = (uint64_t) read + stage_files_bytes(m.serial, entry_pfx) + en.journal_bytes;
        bytes_ += en.file_bytes - old_bytes;
        en.used = next_used_++;
        break;
    }
    ++restores_;
    return back;
}

bool ConversationDiskStore::evict_oldest(uint64_t protect_serial) {
    if (entries_.empty()) return false;
    // The least-recently-used entry goes, by the `used` stamp (bumped by put, take and every
    // delta append): with take() now KEEPING the entry in the manifest, the just-restored
    // conversation must not be the victim while older untouched entries exist. Pinned shared
    // prefixes are still spared while an ordinary conversation is available (they are the
    // smallest and most-reused assets — every new conversation's first request restores one).
    // With only prefixes left, the least recently used still goes (a full disk must not wedge
    // parking) — unless the only thing left is a protected append's own base: its record may
    // then run over the budget rather than destroy the base it extends.
    size_t victim = SIZE_MAX;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (is_prefix_entry(entries_[i]) || entries_[i].serial == protect_serial) continue;
        if (victim == SIZE_MAX || entries_[i].used < entries_[victim].used) victim = i;
    }
    if (victim == SIZE_MAX) {
        if (protect_serial != UINT64_MAX) return false;
        victim = 0;
    }
    const DiskEntry e = std::move(entries_[victim]);
    bytes_ -= e.file_bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) victim);
    serials_.erase(serials_.begin() + (std::ptrdiff_t) victim);
    remove_spill_files(e.path);   // .bin + .idx + .idx.tmp + stages (+ .dlt via the serial sweep)
    ++drops_;
    return true;
}

size_t ConversationDiskStore::clear() {
    flush_pending();   // a queued image may be the poisoned one: let it land, then purge it too
    const size_t n = entries_.size();
    for (const auto& e : entries_) remove_spill_files(e.path);   // .bin + .idx + .idx.tmp
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
        // conv-<serial>.bin / pfx-<serial>.bin (a shared-prefix snapshot), their sidecars .idx,
        // and any leftover .tmp from an interrupted write (.bin.tmp or .idx.tmp). All are
        // removed; other files (e.g. keep.txt in the test) are left alone.
        const bool named_conv = name.rfind("conv-", 0) == 0;
        const bool named_pfx = name.rfind("pfx-", 0) == 0;
        const bool is_bin = (named_conv || named_pfx) && name.size() > 4 &&
                            name.compare(name.size() - 4, 4, ".bin") == 0;
        const bool is_idx = (named_conv || named_pfx) && name.size() > 4 &&
                            name.compare(name.size() - 4, 4, ".idx") == 0;
        const bool is_dlt = (named_conv || named_pfx) && name.size() > 4 &&
                            name.compare(name.size() - 4, 4, ".dlt") == 0;
        const bool is_tmp = name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0;
        if (is_bin || is_idx || is_dlt || is_tmp) {
            fs::remove(entry.path(), ec);
        }
    }
}

size_t ConversationDiskStore::reload_from_disk(const SessionFileIdentity& identity) {
    // Reload every valid spill file from the spill dir on engine startup, so a request can
    // restore a conversation the previous engine run spilled. Each conv-<serial>.bin is
    // validated with session_file_read (header + identity + payload hash + full parse), the
    // cheap manifest data (ids, imgs, geometry, layer carve, cvec, file_bytes) is captured in
    // RAM, and the heavy K/V image is dropped right after.
    //
    // The disk budget is enforced by evict_oldest() (the real LRU): files that don't fit are
    // deleted with proper accounting (drops_++). The spill dir can hold conversations from
    // multiple sessions, so we no longer assume the highest serial "includes" the others.
    //
    // Files that fail validation (corruption, identity mismatch, unsupported version) are
    // deleted: they can never match this engine. A SessionError::memory (RAM preflight
    // refused) KEEPS the file — the next engine start with more free RAM may succeed. A
    // std::bad_alloc mid-read stops the loop: the remaining files stay on disk for a later
    // reload (NOT deleted — they were never validated).
    //
    // Iteration order: ASCENDING by serial (oldest first). entries_ is a std::deque
    // oldest-first and evict_oldest() pops the front; pushing back in ascending order keeps
    // the oldest at the front so budget pressure evicts the right entries. Iterating
    // descending would put the newest at the front and evict_oldest() would drop the
    // newest conversation first — wrong.
    //
    // Leftover .tmp files (an interrupted write) are cleaned: they were never renamed
    // to .bin, so they are not valid spills.
    if (!enabled() || dir_.empty()) return 0;
    // Clear any stale manifest first (we are about to rebuild it from disk).
    entries_.clear();
    serials_.clear();
    bytes_ = 0;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return 0;
    // Collect conv-*.bin / pfx-*.bin files sorted by serial (filename sorts lexicographically
    // the same as numeric for the decimal form std::to_string produces, so the LRU order entries_
    // is built oldest-first — mirrors the order put() inserts at).
    struct SpillFile { fs::path path; uint64_t serial; bool prefix; };
    std::vector<SpillFile> files;
    // A layer split's stage files (conv-<serial>.stage<k>.bin for a conversation,
    // pfx-<serial>.stage<k>.bin for a shared-prefix snapshot): collected to clean orphans and
    // to keep them from being mistaken for mains — "conv-808.stage0.bin" ends in .bin and its
    // digits parse to 808, which would load serial 808 twice. Both kinds can carry stages.
    std::vector<SpillFile> stage_files;
    // Delta journals (conv-<serial>.dlt): collected to clean orphans — a journal whose main
    // .bin is gone can never apply to anything. The journals of surviving mains are read
    // through path_for_dlt after the entry loads (the .idx fast path already carries their
    // effect in the manifest data; the slow path folds them into the parsed image).
    std::vector<SpillFile> dlt_files;
    // The highest serial any file names — the fresh-serial floor for the put() collision guard
    // (the park counter restarts at 1 every run; a reloaded serial must never be reused).
    uint64_t max_serial_seen = 0;
    bool any_serial_seen = false;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            // Leftover .tmp from an interrupted write (.bin.tmp or .idx.tmp): clean it up.
            fs::remove(entry.path(), ec);
            continue;
        }
        // <kind>-<digits>.<tail>: kind is "conv" (a whole conversation) or "pfx" (a shared-prefix
        // snapshot); the serial runs to the first '.', the tail names the file's role — "bin"
        // the main, "idx" the sidecar, "stage<k>.bin" a later stage's image.
        const bool is_pfx = name.rfind("pfx-", 0) == 0;
        if (!is_pfx && name.rfind("conv-", 0) != 0) continue;
        const size_t digits0 = is_pfx ? 4 : 5;
        if (name.size() <= digits0 + 1) continue;
        size_t p = digits0;
        while (p < name.size() && name[p] >= '0' && name[p] <= '9') ++p;
        if (p == digits0 || p >= name.size() || name[p] != '.') {
            // Not <kind>-<digits>.… — a junk conv-*.bin / pfx-*.bin name (the old scan removed
            // these): drop it with its sidecar. A conv-.bin (9 chars) is left alone, as before.
            if (name.size() > 9 && name.compare(name.size() - 4, 4, ".bin") == 0)
                remove_spill_files(entry.path().string());
            continue;
        }
        uint64_t serial = 0;
        try {
            serial = std::stoull(name.substr(digits0, p - digits0));
        } catch (...) {
            remove_spill_files(entry.path().string());   // invalid serial: drop .bin + .idx (+ stages)
            continue;
        }
        if (!any_serial_seen || serial > max_serial_seen) { max_serial_seen = serial; any_serial_seen = true; }
        const std::string tail = name.substr(p + 1);
        if (tail == "bin") {
            files.push_back({entry.path(), serial, is_pfx});
        } else if (tail.rfind("stage", 0) == 0) {
            stage_files.push_back({entry.path(), serial, is_pfx});
        } else if (tail == "dlt") {
            dlt_files.push_back({entry.path(), serial, is_pfx});
        }
        // ".idx" (and anything else) is read through path_for_idx, not scanned here
    }
    // Orphan stage files: a later stage's image whose main .bin is gone (an interrupted put
    // whose main never published, or a removal by a build that did not know about stage files).
    // Unreachable — take() only reads stages of a serial whose main read succeeded — remove.
    {
        std::error_code ec2;
        for (const auto& sf : stage_files) {
            const std::string main_path = path_for(sf.serial, sf.prefix);
            if (!fs::exists(main_path, ec2) || ec2) fs::remove(sf.path, ec2);
        }
        // Orphan journals: a .dlt whose main .bin is gone (the main was evicted or replaced and
        // a build that did not know about journals left the .dlt behind — remove_spill_files'
        // serial-prefix sweep takes it too, but a direct removal here also covers a manual
        // delete of the .bin).
        for (const auto& sf : dlt_files) {
            const std::string main_path = path_for(sf.serial, sf.prefix);
            if (!fs::exists(main_path, ec2) || ec2) fs::remove(sf.path, ec2);
        }
    }
    if (any_serial_seen && max_serial_seen + 1 > next_fresh_serial_) next_fresh_serial_ = max_serial_seen + 1;
    write_id_ = identity;   // the shutdown compaction re-reads what this run can see
    write_id_set_ = true;
    if (files.empty()) return 0;
    // Sort ASCENDING by serial so entries_ ends up oldest-first (LRU front = oldest), matching
    // put()'s order. evict_oldest() pops the front, so under budget pressure the OLDEST spill
    // goes (the right one) — never the newest.
    std::sort(files.begin(), files.end(),
              [](const SpillFile& a, const SpillFile& b) { return a.serial < b.serial; });
    size_t loaded = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& path = files[i].path;
        const uint64_t serial = files[i].serial;
        const bool is_pfx = files[i].prefix;
        const std::string bin_path = path.string();
        // ── Fast path: try the sidecar .idx first. It carries exactly the manifest data we
        // need (ids, imgs, geometry, layer carve, cvec, file_bytes) without parsing the .bin's
        // K/V. read_idx validates magic, version, identity, checksum, and freshness (the .bin's
        // size + mtime on disk must match what the .idx recorded). On any mismatch it returns
        // false and we fall through to the slow path (full session_file_read).
        {
            DiskEntry from_idx;
            const std::string idx_path = path_for_idx(serial, is_pfx);
            if (read_idx(idx_path, from_idx, identity)) {
                from_idx.serial = serial;
                from_idx.prefix = is_pfx;   // the kind lives in the file name; the .idx payload does not carry it
                // from_idx.path was set to the .bin path by read_idx
                // Budget check: same as the slow path. evict_oldest() pops the oldest (front).
                while (!entries_.empty() && bytes_ + from_idx.file_bytes > budget_) {
                    if (!evict_oldest()) break;
                }
                if (from_idx.file_bytes > budget_) {
                    // Single file bigger than the whole budget: drop it (same as slow path).
                    remove_spill_files(bin_path);
                    continue;
                }
                from_idx.used = next_used_++;
                entries_.push_back(std::move(from_idx));
                serials_.push_back(serial);
                bytes_ += entries_.back().file_bytes;
                ++loaded;
                continue;   // .bin not read — that is the whole point of the .idx
            }
        }
        // ── Slow path: .idx missing, corrupt, stale, or wrong-identity. Parse the .bin fully.
        try {
            SavedConversation image;
            size_t bytes_read = 0;
            std::string error;
            SessionStatus status;
            if (!session_file_read(bin_path, identity, image, bytes_read, error, {}, &status)) {
                // NUNCA borrar el .bin aquí. Sea identity mismatch, corrupción, RAM rechazada,
                // o cualquier otro error: loguear y conservar. El archivo solo se elimina por:
                //   1. evict_oldest() cuando excede el budget (LRU)
                //   2. take() exitoso (la conversación volvió a RAM)
                // Si borráramos aquí, no podríamos diagnosticar qué pasó, y un cambio de config
                // temporal (como --max-context) destruiría conversaciones guardadas.
                std::fprintf(stderr, "[strata diag] spill reload: SKIP %s (read failed, KEPT on disk): "
                                     "error=%s, status=%d\n",
                             bin_path.c_str(), error.c_str(), (int) status.error);
                continue;
            }
            // ── Stage files: read them BEFORE the journal fold — a v2 record carries one
            // section per image (main + stages), so the fold needs every image the entry
            // owns in memory. Stop at the first MISSING k (take()'s rule); a set shorter
            // than the journal's image count folds as a torn tail and the base restores
            // alone (the slow path rewrites the .idx from whatever it loaded).
            for (size_t k = 0; ; ++k) {
                std::error_code sec;
                const std::string spath = path_for_stage(serial, k, is_pfx);
                if (!fs::exists(spath, sec) || sec) break;
                SavedConversation stage_img;
                size_t s_read = 0;
                std::string s_error;
                SessionStatus s_status;
                if (!session_file_read(spath, identity, stage_img, s_read, s_error, {}, &s_status)) {
                    std::fprintf(stderr, "[strata diag] spill reload: SKIP %s (stage %zu read failed, "
                                         "KEPT on disk): error=%s, status=%d\n",
                                 spath.c_str(), k, s_error.c_str(), (int) s_status.error);
                    break;
                }
                image.stage_images.push_back(std::move(stage_img));
            }
            // ── Delta journal: fold the records onto the parsed images (main + stages, the
            // whole pair) BEFORE the manifest data is captured, so the entry describes the
            // state the disk restores (base + journal). The .idx fast path already carries
            // that (the .idx is rewritten after every append); this is the stale-.idx
            // recovery. A journal that does not chain onto this base is junk for it —
            // dropped whole; a torn tail is truncated.
            uint64_t e_journal_bytes = 0;
            uint32_t e_journal_records = 0;
            if (!is_pfx) {
                const std::string dlt = path_for_dlt(serial, false);
                std::error_code ec2;
                if (fs::exists(dlt, ec2) && !ec2) {
                    const DeltaFold fold = conversation_delta_apply(dlt, image);
                    if (fold.dropped) {
                        fs::remove(dlt, ec2);
                        std::fprintf(stderr, "strata serve: conversation spill: reload: journal "
                                             "conv-%llu.dlt does not chain onto its base — dropped\n",
                                     (unsigned long long) serial);
                    } else if (fold.torn) {
                        fs::resize_file(dlt, fold.valid_end, ec2);
                        std::fprintf(stderr, "strata serve: conversation spill: reload: journal "
                                             "conv-%llu.dlt truncated at %llu B (%zu record%s applied)\n",
                                     (unsigned long long) serial, (unsigned long long) fold.valid_end,
                                     fold.applied, fold.applied == 1 ? "" : "s");
                    }
                    if (!fold.dropped && fold.applied > 0) {
                        // journal bytes join the entry's disk footprint (put's accounting)
                        e_journal_bytes = fold.journal_bytes;
                        e_journal_records = fold.records;
                    }
                }
            }
            // Capture the manifest data; the heavy K/V is dropped right after.
            DiskEntry e;
            e.serial = serial;
            e.prefix = is_pfx;
            e.path = bin_path;
            e.ids = image.live.ids;
            e.imgs = image.live.imgs;
            // Capture the deepest checkpoint too (same as put()).
            if (!image.checkpoints.empty()) {
                const auto& deepest = image.checkpoints.back();
                e.ckpt_ids = deepest.ids;
                e.ckpt_imgs = deepest.imgs;
            }
            e.geometry = image.geometry;
            e.layer_lo = image.layer_lo;
            e.layer_hi = image.layer_hi;
            e.cvec = image.cvec;
            // The stage files this entry owns and their carves (what find_delta_base matches
            // against) — the parsed set, which is exactly the files on disk.
            e.n_stages = uint32_t(image.stage_images.size());
            for (const auto& st : image.stage_images) {
                e.stage_layer_lo.push_back(st.layer_lo);
                e.stage_layer_hi.push_back(st.layer_hi);
            }
            // The .bin parse covers the main image only; a layer split's stage files belong to
            // this entry too (put() counted them into its file_bytes, and the fresh .idx fast
            // path carries that total — the slow path must arrive at the same number). Both
            // kinds can carry stages (a conversation and a shared-prefix snapshot alike).
            e.file_bytes = bytes_read + stage_files_bytes(serial, is_pfx);
            e.journal_bytes = e_journal_bytes;
            e.journal_records = e_journal_records;
            e.file_bytes += e_journal_bytes;
            e.used = next_used_++;
            { SavedConversation discard = std::move(image); (void) discard; }   // free the K/V
            // LRU evict if over budget: drop the oldest entries (the front of entries_, which
            // we just loaded first because of the ascending sort). evict_oldest() also deletes
            // the file (with drops_++ accounting), so the on-disk footprint stays bounded.
            while (!entries_.empty() && bytes_ + e.file_bytes > budget_) {
                if (!evict_oldest()) break;
            }
            if (e.file_bytes > budget_) continue;   // single file bigger than the whole budget
            entries_.push_back(std::move(e));
            serials_.push_back(serial);
            bytes_ += e.file_bytes;   // main + stage files (put()'s accounting, restored here)
            ++loaded;
            // Re-write the .idx that was missing/stale — best-effort, same as put().
            // On failure the slow path runs again next start; no data loss.
            if (!write_idx(entries_.back(), bin_path, identity)) {
                std::fprintf(stderr, "strata serve: conversation spill: warning: could not rewrite "
                                     "%s.idx after slow-path parse (the .bin is fine)\n",
                             bin_path.c_str());
            }
        } catch (const std::bad_alloc&) {
            // RAM exhausted mid-load: stop, keep what we have so far, leave the remaining files
            // on disk for a later reload (when the engine has more free RAM). They are NOT
            // deleted — they were never validated, and deleting unvalidated files was the bug.
            std::fprintf(stderr, "strata serve: conversation spill: reload_from_disk stopped at "
                                 "file %zu/%zu (host RAM exhausted; remaining files kept on disk)\n",
                         i + 1, files.size());
            break;
        }
    }
    // No unconditional deletion of older spills. The LRU (evict_oldest + budget_) already
    // deletes files that don't fit, with proper accounting (drops_). Borrar en el arranque
    // asumiendo que el de mayor serial "incluye" a los anteriores era exactamente el bug:
    // conversaciones distintas conviven en disco y todas menos la última se destruían.
    // ── Pero sí dedup de la MISMA conversación: los spills duplicados que dejaron los runs
    // anteriores (un serial por park; el cleanup superseded no veía a través del re-render del
    // thinking del cliente). El más nuevo de cada conversación gana; los archivos físicos
    // de los dropeados se borran también (esto es lo que colapsa los 18-archivos-de-1-
    // conversación al primer arranque tras el fix).
    const size_t dup = dedupe_loaded();
    if (dup > 0) {
        std::fprintf(stderr, "strata serve: conversation spill: cleaned %zu duplicate spill file%s "
                             "of the same conversation left by earlier runs (manifest=%zu, %.1f MiB)\n",
                     dup, dup == 1 ? "" : "s", entries_.size(), (double) bytes_ / (1024.0 * 1024.0));
    }
    return loaded;
}

// ── Delta journal: the writer-side selection, append and compaction ────────────────────────

std::optional<ConversationDiskStore::DeltaBase> ConversationDiskStore::find_delta_base(
    const SavedConversation& image) const {
    auto ids_prefix = [](const std::vector<int32_t>& cid, const std::vector<int32_t>& full) {
        return !cid.empty() && cid.size() <= full.size() &&
               std::equal(cid.begin(), cid.end(), full.begin());
    };
    auto imgs_contained = [](const std::vector<ConversationImageKey>& from,
                             const std::vector<ConversationImageKey>& into) {
        if (from.size() > into.size()) return false;
        size_t k = 0;
        for (size_t a = 0; a < into.size() && k < from.size(); ++a)
            if (into[a] == from[k]) ++k;
        return k == from.size();
    };
    std::optional<DeltaBase> best;
    for (size_t i = entries_.size(); i-- > 0;) {   // newest first: the tie prefers the most recent
        const DiskEntry& e = entries_[i];
        if (is_prefix_entry(e) || e.cvec != image.cvec || e.geometry != image.geometry ||
            e.layer_lo != image.layer_lo || e.layer_hi != image.layer_hi)
            continue;
        // A layer split's spill is a delta base only for the SAME split: the record carries one
        // section per image, so the stage count and every stage file's carve must line up with
        // the new capture's stage images (the codec's per-section geometry echo re-checks it at
        // apply time). An entry without carve lists (a pre-v4 .idx) is skipped until a full
        // write replaces it with one that has them.
        if (e.n_stages != image.stage_images.size()) continue;
        if (e.stage_layer_lo.size() != e.n_stages || e.stage_layer_hi.size() != e.n_stages) continue;
        bool carves_ok = true;
        for (size_t k = 0; k < e.n_stages; ++k)
            if (e.stage_layer_lo[k] != image.stage_images[k].layer_lo ||
                e.stage_layer_hi[k] != image.stage_images[k].layer_hi) { carves_ok = false; break; }
        if (!carves_ok) continue;
        uint64_t keep = 0;
        if (ids_prefix(e.ids, image.live.ids) && imgs_contained(e.imgs, image.live.imgs)) {
            keep = (uint64_t) e.ids.size();   // the conversation grew on top of this spill
        } else if (ids_prefix(e.ckpt_ids, image.live.ids) &&
                   imgs_contained(e.ckpt_imgs, image.live.imgs)) {
            keep = (uint64_t) e.ckpt_ids.size();   // the client re-rendered the history: the turn
                                                   // boundary this entry's deepest checkpoint
                                                   // holds is what the new render still contains
        }
        if (keep == 0) continue;
        // The base must carry at least half the new image (the codec's own rule): otherwise the
        // record would not save a rewrite and a full write is strictly better.
        if (keep * 2 < (uint64_t) image.live.ids.size()) continue;
        if (!best || keep > best->keep_tokens) best = DeltaBase{i, keep};
    }
    return best;
}

bool ConversationDiskStore::append_delta(size_t base_index, uint64_t keep_tokens,
                                         const SavedConversation& image, const SessionFileIdentity& id) {
    if (base_index >= entries_.size()) return false;
    // Copies up front: the make-room loop below may evict OTHER entries, and erasing from the
    // middle of a deque invalidates every reference into it.
    const uint64_t base_serial = entries_[base_index].serial;
    const std::string bin_path = entries_[base_index].path;
    const uint64_t base_len = (uint64_t) entries_[base_index].ids.size();
    const uint64_t base_journal = entries_[base_index].journal_bytes;
    // The journal file must be exactly what the entry's accounting says: a size mismatch means
    // a previous append landed but its manifest update did not (a crash window), or the file
    // was touched from outside — either way the safe move is the full write (whose superseded
    // cleanup then replaces this entry, .bin and .dlt included).
    const std::string dlt = path_for_dlt(base_serial, false);
    std::error_code ec;
    const bool existed = fs::exists(dlt, ec) && !ec;
    if (existed) {
        const uint64_t size = fs::file_size(dlt, ec);
        if (ec || size != base_journal) return false;
    } else if (base_journal != 0) {
        return false;   // the entry remembers a journal that is not there: full write
    }
    // Every stage file the record's sections extend must be there: a missing conv-<serial>.
    // stage<k>.bin (an external deletion) would leave the record's section applying onto
    // nothing — the full write's superseded cleanup replaces the whole entry instead.
    for (size_t k = 0; k < entries_[base_index].n_stages; ++k) {
        std::error_code ec2;
        const std::string spath = path_for_stage(base_serial, k, false);
        if (!fs::exists(spath, ec2) || ec2) return false;
    }
    DeltaPlan plan;
    if (!conversation_delta_plan(plan, image, base_len, keep_tokens)) return false;
    // Make room for the record in the disk budget (put_impl's pattern): the base is marked
    // in use first, so the LRU victim is never it while anything else can go.
    entries_[base_index].used = next_used_++;
    while (bytes_ + plan.record_bytes > budget_ && !entries_.empty()) {
        if (!evict_oldest(base_serial)) break;   // nothing else to give: over budget, best effort
    }
    DiskEntry* base = find_by_serial(base_serial);
    if (base == nullptr) return false;   // impossible (it was protected), belt and braces
    uint64_t rec = 0;
    if (!conversation_delta_append(dlt, image, plan, rec)) return false;
    if (!existed) {
        // The journal's NAME must survive a crash right after its creation: flush the folder,
        // the same discipline the .bin's publish applies to its rename (Windows has no folder
        // flush — the same exception session_file_write documents).
#if !defined(_WIN32)
        const int dfd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
#endif
    }
    // The entry becomes the post-apply state; the .idx is refreshed so the next start's fast
    // path carries it (the .dlt's freshness pair rides along).
    base->ids = image.live.ids;
    base->imgs = image.live.imgs;
    if (!image.checkpoints.empty()) {
        const auto& deepest = image.checkpoints.back();
        base->ckpt_ids = deepest.ids;
        base->ckpt_imgs = deepest.imgs;
    } else {
        base->ckpt_ids.clear();
        base->ckpt_imgs.clear();
    }
    base->journal_bytes += rec;
    ++base->journal_records;
    base->file_bytes += rec;
    bytes_ += rec;
    if (!write_idx(*base, bin_path, id))
        std::fprintf(stderr, "strata serve: conversation spill: warning: could not refresh "
                             "%s.idx after a delta append (the .dlt is fine; the next start "
                             "parses the .bin and folds the journal)\n", bin_path.c_str());
    std::fprintf(stderr, "strata serve: conversation spill [delta]: +%zu tokens (%.1f MiB) "
                         "appended to conv-%llu.dlt (base %zu tokens; journal %.1f MiB, %u record%s)\n",
                 image.live.ids.size() - size_t(plan.keep_tokens), rec / (1024.0 * 1024.0),
                 (unsigned long long) base_serial, size_t(plan.keep_tokens),
                 base->journal_bytes / (1024.0 * 1024.0), base->journal_records,
                 base->journal_records == 1 ? "" : "s");
    return true;
}

void ConversationDiskStore::compact_journal(DiskEntry& e, const SessionFileIdentity& id) {
    // Read the base .bin AND every stage file, apply the journal to all of them, rewrite each
    // file with the folded state, drop the .dlt. Every step is fail-soft: an error leaves the
    // entry exactly as it was (the journal stays valid — the rewrites are atomic temp+rename
    // over the SAME names, and a crash between a stage's rename and the main's leaves the
    // journal applying cleanly onto whatever mix is on disk: the records' spans [at, at+n) are
    // identical below the boundary either way, so the fold converges; a crash AFTER the main's
    // rename is the idempotency case the reader's chain rule already handles — the stale
    // journal does not chain onto the new base and is dropped).
    try {
        SavedConversation image;
        size_t read = 0;
        std::string error;
        SessionStatus status;
        if (!session_file_read(e.path, id, image, read, error, {}, &status)) {
            std::fprintf(stderr, "strata serve: conversation spill: compaction could not re-read "
                                 "%s (%s); the journal stays as it is\n", e.path.c_str(), error.c_str());
            return;
        }
        // The stage files, exactly like take() reads them: stop at the first missing k. A
        // partial set cannot fold (the record's image count would not match) and the fold
        // below reports it as a torn tail — the entry keeps its files and the next idle pass
        // retries once the .bin is whole again.
        for (size_t k = 0; ; ++k) {
            std::error_code sec;
            const std::string spath = path_for_stage(e.serial, k, false);
            if (!fs::exists(spath, sec) || sec) break;
            SavedConversation stage_img;
            size_t s_read = 0;
            std::string s_error;
            SessionStatus s_status;
            if (!session_file_read(spath, id, stage_img, s_read, s_error, {}, &s_status)) {
                std::fprintf(stderr, "strata serve: conversation spill: compaction could not re-read "
                                     "%s (%s); the journal stays as it is\n", spath.c_str(), s_error.c_str());
                return;
            }
            image.stage_images.push_back(std::move(stage_img));
        }
        const std::string dlt = path_for_dlt(e.serial, false);
        std::error_code ec;
        if (!fs::exists(dlt, ec) || ec) return;   // nothing to fold
        const DeltaFold fold = conversation_delta_apply(dlt, image);
        if (fold.applied == 0) {
            // The journal does not chain onto this base (already folded, or foreign): drop it.
            fs::remove(dlt, ec);
            return;
        }
        if (fold.torn) fs::resize_file(dlt, fold.valid_end, ec);   // heal, then fold what healed
        SessionWriteOptions opts;   // durable, default free-space preflight
        opts.min_free_bytes = 256ull * 1024 * 1024;
        // Rewrite the STAGES first and the main LAST — put_impl's commit order: a crash
        // mid-rewrite leaves orphan/duplicate state the journal's chain rule sorts out, never
        // a published main whose stages are missing.
        uint64_t stages_written = 0;
        for (size_t k = 0; k < image.stage_images.size(); ++k) {
            size_t s_written = 0;
            std::string s_error;
            SessionStatus s_status;
            const std::string spath = path_for_stage(e.serial, k, false);
            if (!session_file_write(spath, image.stage_images[k], id, s_written, s_error, opts, &s_status)) {
                std::fprintf(stderr, "strata serve: conversation spill: compaction could not rewrite "
                                     "%s (%s); the journal stays as it is\n", spath.c_str(), s_error.c_str());
                return;
            }
            stages_written += s_written;
        }
        size_t written = 0;
        SessionStatus wstatus;
        if (!session_file_write(e.path, image, id, written, error, opts, &wstatus)) {
            std::fprintf(stderr, "strata serve: conversation spill: compaction could not rewrite "
                                 "%s (%s); the journal stays as it is\n", e.path.c_str(), error.c_str());
            return;
        }
        fs::remove(dlt, ec);
        // Re-account: the .bin (and its stages) now carry the folded state, the journal is gone.
        bytes_ -= e.file_bytes;
        e.ids = image.live.ids;
        e.imgs = image.live.imgs;
        if (!image.checkpoints.empty()) {
            const auto& deepest = image.checkpoints.back();
            e.ckpt_ids = deepest.ids;
            e.ckpt_imgs = deepest.imgs;
        } else {
            e.ckpt_ids.clear();
            e.ckpt_imgs.clear();
        }
        e.n_stages = uint32_t(image.stage_images.size());
        e.stage_layer_lo.clear();
        e.stage_layer_hi.clear();
        for (const auto& st : image.stage_images) {
            e.stage_layer_lo.push_back(st.layer_lo);
            e.stage_layer_hi.push_back(st.layer_hi);
        }
        e.journal_bytes = 0;
        e.journal_records = 0;
        e.file_bytes = (uint64_t) written + stages_written;
        bytes_ += e.file_bytes;
        if (!write_idx(e, e.path, id))
            std::fprintf(stderr, "strata serve: conversation spill: warning: could not refresh "
                                 "%s.idx after compaction (the .bin is fine)\n", e.path.c_str());
        std::fprintf(stderr, "strata serve: conversation spill [delta]: compacted conv-%llu.dlt "
                             "(%zu record%s, %.1f MiB) into %s (%zu tokens, %.1f MiB%s)\n",
                     (unsigned long long) e.serial, fold.applied, fold.applied == 1 ? "" : "s",
                     fold.journal_bytes / (1024.0 * 1024.0), e.path.c_str(), image.live.ids.size(),
                     (double) (written + stages_written) / (1024.0 * 1024.0),
                     image.stage_images.empty() ? "" : " + stages");
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "strata serve: conversation spill: compaction ran out of RAM; the "
                             "journal stays as it is (the next idle pass retries)\n");
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "strata serve: conversation spill: compaction failed (%s); the "
                             "journal stays as it is\n", ex.what());
    }
}

bool ConversationDiskStore::compact_idle(const SessionFileIdentity& id) {
    for (size_t i = entries_.size(); i-- > 0;) {
        DiskEntry& e = entries_[i];
        if (is_prefix_entry(e) || e.journal_records == 0) continue;
        const uint64_t bin_bytes = e.file_bytes > e.journal_bytes ? e.file_bytes - e.journal_bytes : 0;
        if (e.journal_bytes > kJournalCompactBytes || e.journal_records > kJournalCompactRecords ||
            e.journal_bytes * 5 > bin_bytes) {
            compact_journal(e, id);
            return true;   // at most one per idle pass
        }
    }
    return false;
}

void ConversationDiskStore::compact_all() {
    if (!write_id_set_) return;
    for (size_t i = entries_.size(); i-- > 0;) {
        DiskEntry& e = entries_[i];
        if (is_prefix_entry(e) || e.journal_records == 0) continue;
        const uint64_t bin_bytes = e.file_bytes > e.journal_bytes ? e.file_bytes - e.journal_bytes : 0;
        if (e.journal_bytes > kJournalCompactBytes || e.journal_records > kJournalCompactRecords ||
            e.journal_bytes * 5 > bin_bytes)
            compact_journal(e, write_id_);
    }
}

// Older reload duplicates: entry i is superseded when a NEWER entry (higher index = higher
// serial, reload pushed ascending) is the same conversation — the strict-prefix rule, or the
// re-render rule (the old entry's deepest checkpoint is contained in the newer entry's live
// ids). A different conversation sharing only the system root is kept, and so is an older
// LONGER spill the newer one does not contain (a rollback branch: the client may go back).
// Files go too: an entry not in the manifest is unreachable forever (take() reads by serial
// through best(), and best() only sees entries_).
size_t ConversationDiskStore::dedupe_loaded() {
    auto contained_in = [](const std::vector<int32_t>& cid, const std::vector<int32_t>& full) {
        if (cid.empty() || cid.size() > full.size()) return false;
        return std::equal(cid.begin(), cid.end(), full.begin());
    };
    auto imgs_contained = [](const std::vector<ConversationImageKey>& from,
                             const std::vector<ConversationImageKey>& into) {
        if (from.size() > into.size()) return false;
        size_t k = 0;
        for (size_t a = 0; a < into.size() && k < from.size(); ++a)
            if (into[a] == from[k]) ++k;
        return k == from.size();
    };
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        // A pinned shared prefix is never "an older duplicate of this conversation": every
        // conversation of the same client contains it, which is exactly why it stays.
        if (is_prefix_entry(entries_[i])) { ++i; continue; }
        bool superseded = false;
        for (size_t j = i + 1; j < entries_.size() && !superseded; ++j) {
            if (entries_[j].cvec != entries_[i].cvec || entries_[j].ids.empty()) continue;
            if (!entries_[i].ids.empty() && contained_in(entries_[i].ids, entries_[j].ids) &&
                imgs_contained(entries_[i].imgs, entries_[j].imgs)) {
                superseded = true;   // the plain "grew" case (or an exact retry)
                continue;
            }
            if (!entries_[i].ckpt_ids.empty() && contained_in(entries_[i].ckpt_ids, entries_[j].ids) &&
                imgs_contained(entries_[i].ckpt_imgs, entries_[j].imgs)) {
                superseded = true;   // the re-rendered same-conversation case
            }
        }
        if (!superseded) { ++i; continue; }
        const std::string drop_path = entries_[i].path;
        const uint64_t drop_bytes = entries_[i].file_bytes;
        erase(i);
        bytes_ -= drop_bytes;
        remove_spill_files(drop_path);
        ++dropped;
    }
    return dropped;
}

size_t ConversationDiskStore::drop_superseded(const std::vector<int32_t>& ids,
                                              const std::vector<ConversationImageKey>& images,
                                              const std::vector<ConversationCheckpoint>& checkpoints,
                                              bool cvec) {
    flush_pending();
    return drop_superseded_impl(ids, images, checkpoints, cvec);
}

size_t ConversationDiskStore::drop_superseded_impl(const std::vector<int32_t>& ids,
                                              const std::vector<ConversationImageKey>& images,
                                              const std::vector<ConversationCheckpoint>& checkpoints,
                                              bool cvec, uint64_t protect_serial) {
    auto held = [&](const std::vector<int32_t>& cid, const std::vector<ConversationImageKey>& cimgs) {
        if (cid == ids && cimgs == images) return true;
        for (const auto& k : checkpoints)
            if (k.ids == cid && k.imgs == cimgs) return true;
        return false;
    };
    // A spill whose ids are a strict prefix of the new conversation's ids is OBSOLETE: the new
    // spill (about to be written, or just written by the regular park path) carries that older
    // conversation as a prefix, plus the new turn on top. Keeping the old spill on disk wastes
    // the disk budget (and best() would never pick it over the longer new one). So drop it too.
    // This is what fixes the "11 files for 1 conversation" bug: each request grew the live state
    // by ~80 tokens, the spill stayed at the old length, and exact-equality drop never matched.
    auto is_prefix_of = [](const std::vector<int32_t>& cid, const std::vector<int32_t>& longer) {
        if (cid.empty() || cid.size() > longer.size()) return false;
        return std::equal(cid.begin(), cid.end(), longer.begin());
    };
    // cid is contained in full at its start (prefix, strict OR equal): the "grew" and the
    // "identical retry" cases in one predicate.
    auto contained_in = [](const std::vector<int32_t>& cid, const std::vector<int32_t>& full) {
        if (cid.empty() || cid.size() > full.size()) return false;
        return std::equal(cid.begin(), cid.end(), full.begin());
    };
    // every image of `from` appears in `into`, in order (a prefix of the image list)
    auto imgs_contained = [](const std::vector<ConversationImageKey>& from,
                             const std::vector<ConversationImageKey>& into) {
        if (from.size() > into.size()) return false;
        size_t k = 0;
        for (size_t a = 0; a < into.size() && k < from.size(); ++a)
            if (into[a] == from[k]) ++k;
        return k == from.size();
    };
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        const auto& e = entries_[i];
        if (e.cvec != cvec) { ++i; continue; }
        if (is_prefix_entry(e)) { ++i; continue; }   // a pinned shared prefix: not this conversation's branch
        // The delta base the writer just appended to: after the append this entry IS the newest
        // state of this conversation, so every branch below (exact, prefix, re-render) would
        // describe it as "superseded by the park it just absorbed" and delete it — .bin, .dlt
        // and .idx — one step after the append. Protected here; every OTHER serial of the same
        // branch (an older base, a pre-journal spill) still goes.
        if (e.serial == protect_serial) { ++i; continue; }
        // The deepest checkpoint the spill file carries is the one ConversationCache handed to
        // session_checkpoints_to_save. The manifest only stores the live prefix; if the live
        // prefix itself is what the new conversation supersedes, drop. (Phase 2: store the
        // deepest checkpoint's ids/imgs in the manifest too for a more accurate match.)
        if (!e.ids.empty() && held(e.ids, e.imgs)) {
            // Exact match: the new spill is the same conversation. Replace the old file.
            const std::string drop_path = e.path;
            const uint64_t drop_bytes = e.file_bytes;
            erase(i);
            bytes_ -= drop_bytes;
            remove_spill_files(drop_path);   // BORRAR: reemplazo por versión más nueva
            ++dropped;
            continue;
        }
        // Also drop spills whose ids are a strict prefix of the new live state: they are
        // snapshots of an earlier turn of the SAME conversation, now included in the longer one.
        // This is the "20 messages replaces 10 messages" case.
        if (!e.ids.empty() && is_prefix_of(e.ids, ids)) {
            if (imgs_contained(e.imgs, images)) {
                const std::string drop_path = e.path;
                const uint64_t drop_bytes = e.file_bytes;
                erase(i);
                bytes_ -= drop_bytes;
                remove_spill_files(drop_path);   // BORRAR: la conversación mayor reemplaza a la menor
                ++dropped;
                continue;
            }
        }
        // ── Same conversation, re-rendered. The client rewrites every assistant turn without
        // its thinking tokens, so two successive spills of the SAME conversation match NEITHER
        // rule above: the old live ids end inside the reasoning of a turn the new spill
        // re-rendered empty, so old ⊄ new, and one spill per park piles up (measured: 18 files
        // for 1 conversation, ~1 GiB each). The identity that DOES survive the re-render is
        // the old entry's DEEPEST CHECKPOINT — the last turn boundary of the client's own
        // render, a region every later prompt of this conversation still contains verbatim.
        // When the new conversation contains that boundary, the old spill holds nothing the
        // new one does not: drop it. A different conversation of the same client does NOT
        // contain this conversation's boundary (it shares only the system root), so it stays.
        // The "larger size replaces the existing one" rule the user asked for.
        if (!e.ckpt_ids.empty() && contained_in(e.ckpt_ids, ids) && imgs_contained(e.ckpt_imgs, images)) {
            const std::string drop_path = e.path;
            const uint64_t drop_bytes = e.file_bytes;
            erase(i);
            bytes_ -= drop_bytes;
            remove_spill_files(drop_path);   // BORRAR: misma conversación, re-render más nuevo
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
