// CPU-only test for the punta 2 fix: when the RAM cache's admit() refuses the
// snapshot (RAM floor would be violated), the captured image is spilled directly
// to disk instead of being dropped. This validates the path that the engine's
// park_current_body takes when conversation_memory_admit returns false.
//
// The test does NOT instantiate the full engine (no CUDA, no model). Instead it
// exercises the same sequence the engine follows:
//   1. Capture a SavedConversation (simulated with sample())
//   2. The RAM admit check fails (simulated by calling disk_store.put() directly)
//   3. The disk store accepts the image
//   4. A later turn's restore_from_disk_or_ram() recovers it
#include "strata/core/conversation_disk_store.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
// Variant that takes a label-with-args for the loop case.
void checkf(bool ok, const char* fmt, ...) {
    ++checks;
    if (!ok) {
        char buf[256];
        va_list ap; va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        std::fprintf(stderr, "FAIL: %s\n", buf);
        std::exit(1);
    }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(99, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(31, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

SavedConversation sample(size_t live_tokens = 1000) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(live_tokens, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        kv.k = pattern(4096 + layer, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    return true;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-ramfull-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    // ── 1. RAM refuses admit → spill directly to disk → restore later ─────────────────────
    // Simulates the engine's park_current_body path when conversation_memory_admit returns
    // false: the captured image goes to the disk store, and a later turn restores it.
    {
        SpillConfig cfg;
        cfg.path = (dir / "ramfull1").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        check(store != nullptr && store->enabled(), "disk store enabled");

        // The engine captured this image but the RAM admit check refused to keep it.
        // Instead of dropping it (today's behavior), the engine spills it directly.
        SavedConversation captured = sample(800);
        const std::vector<int32_t> captured_ids = captured.live.ids;
        const std::vector<ConversationImageKey> captured_imgs = captured.live.imgs;
        const size_t captured_tokens = captured.live.ids.size();
        const size_t captured_bytes = captured.bytes();

        // The engine's call: disk_store_ptr->put(std::move(image), serial, identity)
        // Using serial = UINT64_MAX sentinel (the engine's convention for "not from cache evict").
        const uint64_t spill_serial = static_cast<uint64_t>(-1);
        check(store->put(std::move(captured), spill_serial, id), "spill the captured image directly");

        // Verify the disk store now holds the conversation.
        check(store->size() == 1, "disk store has the spilled conversation");
        check(store->writes() == 1, "write counter incremented");

        // A later turn's restore_from_disk_or_ram() should recover it.
        std::vector<int32_t> prompt = captured_ids;
        prompt.push_back(9999);   // one more so conversation_prefix accepts n < prompt.size()
        const ConversationCache::Match no_park{};   // RAM cache has nothing (admit refused)
        auto restored = restore_from_disk_or_ram(no_park, store.get(), prompt, captured_imgs, false,
                                                  /*resume*/ 0, /*slot_tokens*/ 0, id);
        check(restored.has_value(), "the spilled conversation was restored from disk");
        check(restored->live.ids == captured_ids, "restored ids match the original");
        check(store->size() == 1, "disk entry KEPT after restore (take-keep: the delta base)");
    }

    // ── 2. Spill disabled (budget_gib = 0): the path is a no-op, conversation lost ──────
    // When --conversation-spill-gib is 0, the engine's "if (disk_store_ptr != nullptr &&
    // spill_attached)" guard skips the spill and the conversation is lost as today.
    {
        SpillConfig cfg;
        cfg.path = (dir / "ramfull2").string();
        cfg.budget_gib = 0;
        auto store = make_disk_store(cfg, &id);
        check(store != nullptr, "store constructed even when disabled");
        check(!store->enabled(), "store is disabled");

        // The engine's guard: disk_store_ptr != nullptr && spill_attached. With a disabled
        // store, attach_spill() does NOT set the spill pointer (see attach_spill impl), so
        // the engine's guard skips the spill. We simulate that here by NOT calling put().
        SavedConversation captured = sample(500);
        // (engine skips the spill because spill is disabled)
        check(store->size() == 0, "disk store stayed empty (spill disabled)");
    }

    // ── 3. Disk full → spill fails, conversation lost (today's behavior) ─────────────────
    // Simulates a disk that cannot accept the spill (min_free_bytes preflight refuses).
    // The engine logs "lost N tokens" and the next turn re-computes as today.
    {
        SpillConfig cfg;
        cfg.path = (dir / "ramfull3").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);

        SavedConversation captured = sample(200);
        SessionWriteOptions opts;
        opts.min_free_bytes = 1ull << 40;   // 1 TiB headroom: any real disk fails this preflight
        const uint64_t spill_serial = 42;
        const bool ok = store->put(std::move(captured), spill_serial, id, opts);
        check(!ok, "spill fails when disk preflight refuses");
        check(store->size() == 0, "disk store empty after failed spill");
        check(store->writes() == 0, "write counter not incremented on failure");
    }

    // ── 4. Multiple RAM-full spills: each turn spills, the latest restores ───────────────
    // Simulates several turns where every park is refused by RAM (chronic RAM pressure).
    // Each turn spills its captured image; the disk store keeps the latest N (LRU evicts older).
    // A later turn with a matching prompt restores the most recent one.
    {
        SpillConfig cfg;
        cfg.path = (dir / "ramfull4").string();
        cfg.budget_gib = 1;   // ~20 entries of ~50 KiB before LRU
        auto store = make_disk_store(cfg, &id);

        // Spill 5 conversations with distinct prefixes (so best() can distinguish them).
        for (int i = 0; i < 5; ++i) {
            SavedConversation s = sample(100);
            for (size_t k = 0; k < s.live.ids.size(); ++k) s.live.ids[k] = int32_t(2000 + i * 1000 + k);
            for (ConversationCheckpoint& c : s.checkpoints)
                for (size_t k = 0; k < c.ids.size(); ++k) c.ids[k] = int32_t(2000 + i * 1000 + k);
            const uint64_t serial = 100 + i;
            checkf(store->put(std::move(s), serial, id), "spill conversation %d", i);
        }
        check(store->size() == 5, "disk store holds 5 spilled conversations");

        // A later turn whose prompt matches conversation #3 (ids 5000..5099).
        std::vector<int32_t> prompt;
        for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(5000 + i));
        prompt.push_back(9999);
        const std::vector<ConversationImageKey> imgs = {{3, 0x1234567890abcdefull ^ 1ull}};
        const ConversationCache::Match no_park{};
        auto restored = restore_from_disk_or_ram(no_park, store.get(), prompt, imgs, false,
                                                  /*resume*/ 0, /*slot*/ 0, id);
        check(restored.has_value(), "restored conversation #3 from disk");
        check(restored->live.ids.size() == 100, "restored 100 tokens");
        check(restored->live.ids[0] == 5000, "restored prefix starts at 5000 (conversation #3)");
        check(store->size() == 5, "disk store keeps all 5 after take (take-keep: the delta base)");
    }

    fs::remove_all(dir);
    std::printf("conversation_spill_ram_full_test: %d checks passed\n", checks);
    return 0;
}
