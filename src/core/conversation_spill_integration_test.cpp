// CPU-only integration tests for the ConversationCache + ConversationDiskStore cooperation
// (the 7 integration points the engine calls). Mirrors conversation_disk_store_test.cpp's style
// but exercises the FREE FUNCTIONS (attach_spill, restore_from_disk_or_ram, spill_drop_superseded,
// spill_clear, spill_purge_on_startup, make_disk_store) that generate.cpp invokes — so a test run
// validates the engine's wiring without needing a model or GPU.
#include "strata/core/conversation_disk_store.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// Same builders as conversation_disk_store_test.cpp (small payload, fast tests).
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
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-integration-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    // ── 1. make_disk_store + attach_spill: built and bound, evict_oldest spills ────────────
    {
        SpillConfig cfg;
        cfg.path = (dir / "build1").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        check(store != nullptr, "make_disk_store returns a store");
        check(store->enabled(), "store is enabled with budget_gib > 0");
        check(fs::is_directory(dir / "build1"), "spill dir was created");

        // Build a small RAM cache with 1 slot, so the second put evicts the first.
        ConversationCache cache(/*budget*/ 1ull << 20, /*slots*/ 1, /*min_tokens*/ 0);
        attach_spill(cache, store.get(), &id);
        // Park a conversation; with the spill attached, evict_oldest (called by make_room when
        // the next put needs room) writes it to disk instead of dropping it.
        SavedConversation first = sample(200);
        const std::vector<ConversationImageKey> imgs1 = first.live.imgs;
        check(cache.put(std::move(first)), "first put succeeds");
        check(cache.size() == 1, "RAM cache holds one entry");
        // A second put with a different prefix triggers make_room → evict_oldest → spill.
        // Mutate BOTH live.ids AND every checkpoint's ids so drop_superseded does not mistake
        // the second for a newer copy of the first (their checkpoints would otherwise share ids).
        SavedConversation second = sample(300);
        for (size_t i = 0; i < second.live.ids.size(); ++i)
            second.live.ids[i] = int32_t(5000 + i);   // distinct prefix
        for (ConversationCheckpoint& c : second.checkpoints)
            for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = int32_t(5000 + i);
        const std::vector<ConversationImageKey> imgs2 = second.live.imgs;
        check(cache.put(std::move(second)), "second put succeeds (make_room evicted the first)");
        check(store->size() >= 1, "the evicted entry was spilled to disk (not dropped)");

        // The spilled entry is gone from RAM (it was evicted) but findable on disk via best().
        std::vector<int32_t> prompt1;
        for (int i = 0; i < 200; ++i) prompt1.push_back(int32_t(1000 + i));
        prompt1.push_back(9999);
        const auto disk_match = store->best(prompt1, imgs1, false);
        check(disk_match.tokens == 200, "disk has the spilled conversation's prefix");
    }

    // ── 2. restore_from_disk_or_ram: RAM wins when it has a match, disk as fallback ──────
    {
        SpillConfig cfg;
        cfg.path = (dir / "build2").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        ConversationCache cache(1ull << 20, 2, 0);
        attach_spill(cache, store.get(), &id);

        // Park a conversation in RAM.
        SavedConversation ram_entry = sample(500);
        const std::vector<ConversationImageKey> imgs = ram_entry.live.imgs;
        check(cache.put(std::move(ram_entry)), "park in RAM");
        // Spill a DIFFERENT conversation to disk via evict (force pressure with a third one).
        SavedConversation disk_entry = sample(800);
        for (size_t i = 0; i < disk_entry.live.ids.size(); ++i)
            disk_entry.live.ids[i] = int32_t(8000 + i);
        const std::vector<ConversationImageKey> disk_imgs = disk_entry.live.imgs;
        // Force it to disk: put a much larger entry that triggers eviction of disk_entry first.
        // Easier path: spill it directly with put().
        check(store->put(std::move(disk_entry), 999, id), "spill the 800-token conversation directly");

        // Case A: RAM has a match → restore_from_disk_or_ram returns nullopt (engine takes from RAM).
        std::vector<int32_t> prompt_ram;
        for (int i = 0; i < 500; ++i) prompt_ram.push_back(int32_t(1000 + i));
        prompt_ram.push_back(9999);
        const auto parked_ram = cache.best(prompt_ram, imgs, false);
        check(parked_ram.tokens == 500, "RAM has the 500-token prefix");
        auto from_disk_a = restore_from_disk_or_ram(parked_ram, store.get(), prompt_ram, imgs, false,
                                                     /*resume*/ 0, /*slot_tokens*/ 0, id);
        check(!from_disk_a.has_value(), "RAM wins; restore_from_disk_or_ram defers to the caller's RAM take");

        // Case B: RAM has no match for the disk-only conversation → disk fallback restores it.
        std::vector<int32_t> prompt_disk;
        for (int i = 0; i < 800; ++i) prompt_disk.push_back(int32_t(8000 + i));
        prompt_disk.push_back(9999);
        const auto parked_disk = cache.best(prompt_disk, disk_imgs, false);
        check(parked_disk.tokens == 0, "RAM has no prefix for the disk-only conversation");
        auto from_disk_b = restore_from_disk_or_ram(parked_disk, store.get(), prompt_disk, disk_imgs, false,
                                                     /*resume*/ 0, /*slot_tokens*/ 0, id);
        check(from_disk_b.has_value(), "disk fallback restored the conversation");
        check(from_disk_b->live.ids.size() == 800, "restored image has 800 tokens");
        check(store->size() == 0, "disk entry removed after restore (take = remove)");
    }

    // ── 3. restore_from_disk_or_ram: resume/slot_tokens floor honors the live session ────
    // When the engine already holds `resume` tokens (the live session is at that point), the
    // disk fallback must NOT restore a shorter prefix — re-computing from `resume` is cheaper.
    {
        SpillConfig cfg;
        cfg.path = (dir / "build3").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        // Park a 100-token conversation to disk.
        SavedConversation small = sample(100);
        const std::vector<ConversationImageKey> imgs = small.live.imgs;
        check(store->put(std::move(small), 7, id), "spill a 100-token conversation");

        std::vector<int32_t> prompt;
        for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);
        const ConversationCache::Match no_park{};   // RAM has no entry
        // Case A: resume = 0 → disk's 100 tokens beat 0, restore.
        auto a = restore_from_disk_or_ram(no_park, store.get(), prompt, imgs, false,
                                          /*resume*/ 0, /*slot*/ 0, id);
        check(a.has_value(), "disk restores when resume is 0");
        // Re-spill for the next case (take removed it).
        SavedConversation small2 = sample(100);
        check(store->put(std::move(small2), 8, id), "re-spill for case B");
        // Case B: resume = 200 → disk's 100 tokens DO NOT beat the live session's 200.
        auto b = restore_from_disk_or_ram(no_park, store.get(), prompt, imgs, false,
                                          /*resume*/ 200, /*slot*/ 0, id);
        check(!b.has_value(), "disk does NOT restore when resume already exceeds its prefix");
    }

    // ── 4. spill_drop_superseded: fresh park with same prefix drops the old spill ───────
    {
        SpillConfig cfg;
        cfg.path = (dir / "build4").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        SavedConversation original = sample(400);
        check(store->put(std::move(original), 50, id), "spill for supersede test");
        check(store->size() == 1, "one spill file");
        // The next turn parks a fresh conversation whose live ids+imgs match the spilled one's.
        SavedConversation next = sample(400);
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(next.live);
        const size_t dropped = spill_drop_superseded(store.get(), next.live.ids, next.live.imgs,
                                                      chain, next.cvec);
        check(dropped == 1, "spill_drop_superseded removed the matching spill");
        check(store->size() == 0, "spill store empty after drop");
    }

    // ── 5. spill_clear: poison (#879) clears the disk store ───────────────────────────────
    {
        SpillConfig cfg;
        cfg.path = (dir / "build5").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        for (int i = 0; i < 3; ++i) {
            SavedConversation s = sample(50);
            check(store->put(std::move(s), 200 + i, id), "spill for clear test");
        }
        check(store->size() == 3, "three spills before clear");
        const size_t cleared = spill_clear(store.get());
        check(cleared == 3, "spill_clear returned the count");
        check(store->size() == 0, "store empty after spill_clear");
        // The files are gone too.
        int leftover = 0;
        for (const auto& e : fs::directory_iterator(dir / "build5"))
            if (e.path().filename().string().rfind("conv-", 0) == 0) ++leftover;
        check(leftover == 0, "no spill files left after spill_clear");
    }

    // ── 6. spill_purge_on_startup: cleans conv-*.bin and leftover .tmp at engine boot ────
    {
        SpillConfig cfg;
        cfg.path = (dir / "build6").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        // Drop fake files as if a previous engine run left them.
        std::ofstream(dir / "build6" / "conv-1.bin") << "stale";
        std::ofstream(dir / "build6" / "conv-2.bin") << "stale";
        std::ofstream(dir / "build6" / "abc.tmp") << "interrupted write";
        std::ofstream(dir / "build6" / "keep.txt") << "preserved";
        spill_purge_on_startup(store.get());
        check(!fs::exists(dir / "build6" / "conv-1.bin"), "purge removed conv-1.bin");
        check(!fs::exists(dir / "build6" / "conv-2.bin"), "purge removed conv-2.bin");
        check(!fs::exists(dir / "build6" / "abc.tmp"), "purge removed leftover .tmp");
        check(fs::exists(dir / "build6" / "keep.txt"), "purge kept unrelated files");
    }

    // ── 7. End-to-end: a full park → evict → restore cycle through the wrappers ──────────
    // This is the integration test the engine's wiring claims to support: park A, park B (which
    // evicts A to disk), then a request whose prompt matches A finds nothing in RAM and restores
    // A from disk via restore_from_disk_or_ram.
    {
        SpillConfig cfg;
        cfg.path = (dir / "build7").string();
        cfg.budget_gib = 1;
        auto store = make_disk_store(cfg, &id);
        ConversationCache cache(/*budget*/ 1ull << 20, /*slots*/ 1, 0);   // 1 slot → second put evicts first
        attach_spill(cache, store.get(), &id);

        // Conversation A: 300 tokens starting at id 1000.
        SavedConversation a = sample(300);
        const std::vector<int32_t> a_ids = a.live.ids;
        const std::vector<ConversationImageKey> a_imgs = a.live.imgs;
        check(cache.put(std::move(a)), "park A in RAM");
        check(cache.size() == 1, "RAM holds A");

        // Conversation B: 300 tokens starting at id 5000 (distinct prefix → forces eviction of A).
        // Mutate B's checkpoint ids too, so drop_superseded does not mistake B for a newer copy of A.
        SavedConversation b = sample(300);
        for (size_t i = 0; i < b.live.ids.size(); ++i) b.live.ids[i] = int32_t(5000 + i);
        for (ConversationCheckpoint& c : b.checkpoints)
            for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = int32_t(5000 + i);
        check(cache.put(std::move(b)), "park B evicts A (which spills to disk)");
        check(cache.size() == 1, "RAM now holds B");
        check(store->size() >= 1, "A is on disk");

        // A request whose prompt matches A: RAM's best() will return B (or nothing matching A),
        // so parked.tokens <= max(resume, slot_tokens) → restore_from_disk_or_ram kicks in.
        std::vector<int32_t> prompt_a = a_ids;
        prompt_a.push_back(9999);   // one more so conversation_prefix accepts n < prompt.size()
        const auto parked = cache.best(prompt_a, a_imgs, false);
        // The RAM cache holds B (5000..), not A (1000..); parked.tokens for prompt_a should be 0.
        check(parked.tokens == 0, "RAM has no prefix matching A's prompt");
        auto restored = restore_from_disk_or_ram(parked, store.get(), prompt_a, a_imgs, false,
                                                  /*resume*/ 0, /*slot*/ 0, id);
        check(restored.has_value(), "A was restored from disk");
        check(restored->live.ids == a_ids, "restored A's ids match the original");
    }

    // ── 8. Spill disabled (budget_gib = 0): all wrappers are no-ops, behavior is today's ──
    {
        SpillConfig cfg;
        cfg.path = (dir / "build8").string();
        cfg.budget_gib = 0;   // disabled
        auto store = make_disk_store(cfg, &id);
        check(store != nullptr, "store is constructed even when disabled");
        check(!store->enabled(), "store is disabled");
        ConversationCache cache(1ull << 20, 1, 0);   // 1 slot: the second put evicts the first
        attach_spill(cache, store.get(), &id);
        // attach_spill with a disabled store should NOT set the spill pointer (no-op).
        // Verify by parking an entry and forcing eviction: nothing should land on disk.
        SavedConversation a = sample(100);
        check(cache.put(std::move(a)), "park A");
        SavedConversation b = sample(100);
        for (size_t i = 0; i < b.live.ids.size(); ++i) b.live.ids[i] = int32_t(9000 + i);
        for (ConversationCheckpoint& c : b.checkpoints)
            for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = int32_t(9000 + i);
        check(cache.put(std::move(b)), "park B evicts A (no spill — disabled)");
        check(store->size() == 0, "disk store stayed empty (spill disabled)");
        // Wrappers are no-ops:
        check(spill_clear(store.get()) == 0, "spill_clear is a no-op on disabled store");
        check(spill_drop_superseded(store.get(), {}, {}, {}, false) == 0,
              "spill_drop_superseded is a no-op on disabled store");
        spill_purge_on_startup(store.get());   // must not crash
        // restore_from_disk_or_ram returns nullopt:
        const ConversationCache::Match no_match{};
        auto r = restore_from_disk_or_ram(no_match, store.get(), std::vector<int32_t>{1, 2, 3}, {}, false, 0, 0, id);
        check(!r.has_value(), "restore_from_disk_or_ram returns nullopt on disabled store");
    }

    // ── 9. Null store: all wrappers tolerate nullptr (engine calls them unconditionally) ──
    // generate.cpp calls spill_drop_superseded(disk_store_ptr, ...) and disk_store_ptr can be
    // nullptr when --conversation-spill-gib was not given. The wrappers must not crash.
    {
        ConversationDiskStore* null_store = nullptr;
        check(spill_clear(null_store) == 0, "spill_clear(nullptr) is a no-op");
        check(spill_drop_superseded(null_store, {1, 2}, {}, {}, false) == 0,
              "spill_drop_superseded(nullptr) is a no-op");
        spill_purge_on_startup(null_store);   // must not crash
        const ConversationCache::Match no_match{};
        auto r = restore_from_disk_or_ram(no_match, null_store, std::vector<int32_t>{1, 2, 3}, {}, false, 0, 0, id);
        check(!r.has_value(), "restore_from_disk_or_ram(nullptr) returns nullopt");
        ConversationCache cache(1ull << 20, 1, 0);   // 1 slot: the second put evicts the first
        attach_spill(cache, nullptr, &id);   // must not crash, must not attach
        // Verify the cache behaves as today (evict drops, does not spill):
        SavedConversation a = sample(100);
        check(cache.put(std::move(a)), "park A");
        SavedConversation b = sample(100);
        for (size_t i = 0; i < b.live.ids.size(); ++i) b.live.ids[i] = int32_t(9000 + i);
        for (ConversationCheckpoint& c : b.checkpoints)
            for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = int32_t(9000 + i);
        check(cache.put(std::move(b)), "park B evicts A (drops, no spill)");
    }

    fs::remove_all(dir);
    std::printf("conversation_spill_integration_test: %d checks passed\n", checks);
    return 0;
}
