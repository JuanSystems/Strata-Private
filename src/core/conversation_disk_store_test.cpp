// CPU-only tests for the disk-tier spill store (include/strata/core/conversation_disk_store.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model. Mirrors the style of
// src/core/conversation_file_test.cpp: a single main(), anonymous-namespace sample builders,
// a check() helper that exits on failure, and a scratch dir under fs::temp_directory_path().
#include "strata/core/conversation_disk_store.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

// Build a small SavedConversation with the same shape as conversation_file_test.cpp's sample(),
// but smaller (we want fast tests, not 17 MiB files). The disk store's logic is independent of
// the payload size; we only need round-trip identity and prefix-match determinism.
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
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};

    // ── 1. Round trip: put → best → take returns the same image, file is gone after take ──────
    {
        ConversationDiskStore store(dir.string() + "/rt", 1ull << 30, &id);
        check(store.enabled(), "store enabled with non-zero budget");
        SavedConversation original = sample(1000);
        // best() uses conversation_prefix() which requires the prompt's image list to match the
        // entry's live.imgs (one image at start=3 here). Pass the same image list so the prefix matches.
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        const uint64_t serial = 42;
        check(store.put(std::move(original), serial, id), "put succeeds");
        check(store.size() == 1, "manifest holds one entry");
        // The manifest's prefix matches the original's live ids (1000 tokens).
        std::vector<int32_t> prompt;
        for (int i = 0; i < 1000; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: the next turn's token
        const auto m = store.best(prompt, imgs, /*cvec=*/false);
        check(m.tokens == 1000, "best() matches the live prefix (1000 tokens)");
        check(m.serial == serial, "best() returns the correct serial");
        auto back = store.take(m, id);
        check(back.has_value(), "take returns the image");
        check(same(sample(1000), *back), "round trip is identical");
        check(store.size() == 0, "take removes the entry");
        check(store.bytes() == 0, "take accounts bytes");
        check(!fs::exists(dir / "rt" / ("conv-" + std::to_string(serial) + ".bin")), "file deleted after take");
        check(store.restores() == 1, "restore counter incremented");
    }

    // ── 2. best() matches prefix without reading the payload (file truncated mid-payload) ────
    // The manifest keeps ids in RAM; a corrupt file must not break best(). take() on a corrupt
    // file returns nullopt (the hash check fails inside session_file_read) and the entry is
    // dropped from the manifest.
    {
        ConversationDiskStore store(dir.string() + "/corrupt", 1ull << 30, &id);
        SavedConversation original = sample(500);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 7, id), "put for corrupt test");
        // Truncate the file to half its size.
        const std::string path = dir.string() + "/corrupt/conv-7.bin";
        std::error_code ec;
        const uintmax_t sz = fs::file_size(path, ec);
        check(!ec && sz > 100, "spill file exists with content");
        fs::resize_file(path, sz / 2, ec);
        check(!ec, "truncate succeeded");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 500; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: the next turn's token (conversation_prefix needs n < prompt.size())
        const auto m = store.best(prompt, imgs, false);
        check(m.tokens == 500, "best() still matches the prefix (manifest in RAM, payload unread)");
        auto back = store.take(m, id);
        check(!back.has_value(), "take rejects corrupt file");
        check(store.size() == 0, "take removed the corrupt entry");
    }

    // ── 3. take() with a wrong SessionFileIdentity returns nullopt and drops the entry ──────
    {
        ConversationDiskStore store(dir.string() + "/id", 1ull << 30, &id);
        SavedConversation original = sample(200);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 11, id), "put with correct id");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 200; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: conversation_prefix needs n < prompt.size()
        const auto m = store.best(prompt, imgs, false);
        check(m.tokens == 200, "best matches");
        const SessionFileIdentity wrong{id.model ^ 1, id.config};
        auto back = store.take(m, wrong);
        check(!back.has_value(), "take with wrong id rejected");
        check(store.size() == 0, "wrong-id entry removed from manifest");
    }

    // ── 4. take() with RAM admit refusal returns nullopt but keeps the file ──────────────────
    // The caller's admit callback says "no RAM now". The file is kept so a later turn can retry.
    {
        ConversationDiskStore store(dir.string() + "/admit", 1ull << 30, &id);
        SavedConversation original = sample(300);
        const std::vector<ConversationImageKey> imgs = original.live.imgs;
        check(store.put(std::move(original), 13, id), "put for admit test");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 300; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more: conversation_prefix needs n < prompt.size()
        const auto m = store.best(prompt, imgs, false);
        SessionReadLimits limits;
        limits.admit = [](uint64_t, std::string&) { return false; };   // refuse any allocation
        auto back = store.take(m, id, limits);
        check(!back.has_value(), "take refused by RAM admit");
        check(store.size() == 0, "entry removed from manifest (no retry tracking in Phase 1)");
        // The file may or may not be present: Phase 1's take() removes the file on every
        // failure path. Phase 2 will keep it on memory refusal for retry. We test Phase 1
        // behavior here: file is gone (the manifest no longer references it).
    }

    // ── 5. LRU eviction: fill the budget, put one more, the oldest is dropped ──────────────
    {
        // Budget = 1 MiB; each sample() is ~50 KiB (small K/V). ~20 fit before LRU kicks in.
        ConversationDiskStore store(dir.string() + "/lru", 1ull << 20, &id);
        std::vector<uint64_t> serials;
        for (int i = 0; i < 30; ++i) {
            SavedConversation s = sample(50);
            const uint64_t ser = 100 + i;
            // Force a unique prefix per entry so best() can distinguish them.
            for (size_t k = 0; k < s.live.ids.size(); ++k) s.live.ids[k] = int32_t(2000 + i * 1000 + k);
            if (!store.put(std::move(s), ser, id)) break;
            serials.push_back(ser);
        }
        check(store.size() < 30, "LRU evicted some entries to fit the budget");
        check(store.drops() > 0, "drop counter incremented");
        // The oldest serial (100) should have been evicted; the newest should still be there.
        // sample()'s checkpoint uses imgs = {{3, 0x1234567890abcdef ^ 1}} (the seed is always 1).
        const std::vector<ConversationImageKey> lru_imgs = {{3, 0x1234567890abcdefull ^ 1ull}};
        std::vector<int32_t> prompt_old;
        for (int i = 0; i < 50; ++i) prompt_old.push_back(int32_t(2000 + i));   // matches serial 100
        prompt_old.push_back(9999);   // one more
        const auto m_old = store.best(prompt_old, lru_imgs, false);
        check(m_old.tokens == 0, "oldest entry evicted (no prefix match)");
        std::vector<int32_t> prompt_new;
        for (int i = 0; i < 50; ++i) prompt_new.push_back(int32_t(2000 + 29 * 1000 + i));   // matches serial 129
        prompt_new.push_back(9999);   // one more
        const auto m_new = store.best(prompt_new, lru_imgs, false);
        check(m_new.tokens == 50, "newest entry still present");
    }

    // ── 6. clear() deletes every spill file and empties the manifest (#879 poison case) ─────
    {
        ConversationDiskStore store(dir.string() + "/clear", 1ull << 30, &id);
        for (int i = 0; i < 3; ++i) {
            SavedConversation s = sample(50);
            check(store.put(std::move(s), 200 + i, id), "put for clear test");
        }
        check(store.size() == 3, "3 entries before clear");
        const size_t dropped = store.clear();
        check(dropped == 3, "clear returned the count");
        check(store.size() == 0, "manifest empty after clear");
        check(store.bytes() == 0, "bytes zero after clear");
        // All conv-*.bin files gone.
        int leftover = 0;
        for (const auto& e : fs::directory_iterator(dir / "clear")) {
            if (e.path().filename().string().rfind("conv-", 0) == 0) ++leftover;
        }
        check(leftover == 0, "no spill files left in dir");
    }

    // ── 7. Disk full: put() returns false, store keeps its existing state, no crash ─────────
    // Simulated via SessionWriteOptions.min_free_bytes set absurdly high (more than the test
    // machine's free space).
    {
        ConversationDiskStore store(dir.string() + "/full", 1ull << 30, &id);
        SavedConversation original = sample(100);
        SessionWriteOptions opts;
        opts.min_free_bytes = 1ull << 40;   // 1 TiB headroom: any real disk fails this
        const bool ok = store.put(std::move(original), 99, id, opts);
        check(!ok, "put refuses on simulated full disk");
        check(store.size() == 0, "no entry left from the failed put");
        check(store.writes() == 0, "write counter not incremented on failure");
    }

    // ── 8. purge_on_startup() removes conv-*.bin and leftover .tmp files ────────────────────
    {
        const fs::path pdir = dir / "purge";
        fs::create_directories(pdir);
        // Drop a fake spill file and a fake .tmp leftover from an interrupted write.
        std::ofstream(pdir / "conv-1.bin") << "garbage";
        std::ofstream(pdir / "conv-2.bin") << "garbage";
        std::ofstream(pdir / "abc.tmp") << "interrupted write";
        std::ofstream(pdir / "unrelated.txt") << "kept";
        ConversationDiskStore store(pdir.string(), 1ull << 30, &id);
        store.purge_on_startup();
        check(!fs::exists(pdir / "conv-1.bin"), "purge removed conv-1.bin");
        check(!fs::exists(pdir / "conv-2.bin"), "purge removed conv-2.bin");
        check(!fs::exists(pdir / "abc.tmp"), "purge removed leftover .tmp");
        check(fs::exists(pdir / "unrelated.txt"), "purge kept unrelated files");
    }

    // ── 9. Disabled store (budget = 0): put returns false, best returns no match ───────────
    {
        ConversationDiskStore store(dir.string() + "/disabled", 0, &id);
        check(!store.enabled(), "budget 0 disables store");
        SavedConversation original = sample(100);
        check(!store.put(std::move(original), 1, id), "put on disabled store fails");
        std::vector<int32_t> prompt;
        for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
        prompt.push_back(9999);   // one more
        // best() on a disabled store: the prompt's image list is irrelevant (entries_ is empty).
        const auto m = store.best(prompt, {}, false);
        check(m.tokens == 0, "best on disabled store returns no match");
    }

    // ── 10. drop_superseded: a fresh park with the same live prefix drops the old spill ─────
    {
        ConversationDiskStore store(dir.string() + "/super", 1ull << 30, &id);
        SavedConversation original = sample(400);
        check(store.put(std::move(original), 50, id), "put for supersede test");
        check(store.size() == 1, "one entry");
        // A new conversation whose live ids + imgs match the spilled one supersedes it.
        SavedConversation next = sample(400);   // same live prefix (1000..1399)
        std::vector<ConversationCheckpoint> chain;
        chain.push_back(next.live);
        const size_t dropped = store.drop_superseded(next.live.ids, next.live.imgs, chain, next.cvec);
        check(dropped == 1, "drop_superseded removed the matching spill");
        check(store.size() == 0, "manifest empty after drop_superseded");
    }

    // ── 11. reload_from_disk: spills written by a previous engine run are picked up ──────
    // Simulates the restart scenario: engine A writes a spill, exits; engine B creates a new
    // ConversationDiskStore on the same dir, calls reload_from_disk(), and the spilled
    // conversation is in the manifest (best() finds it, take() restores it).
    {
        const auto rdir = dir / "reload";
        // Engine A writes one spill.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.enabled(), "store A enabled");
            SavedConversation original = sample(100);
            check(store_a.put(std::move(original), 100, id), "store A put");
            check(store_a.size() == 1, "store A has one entry");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
        }
        // Engine B starts (a new ConversationDiskStore on the same dir). Its manifest starts empty;
        // reload_from_disk rebuilds it from the .bin files written by A. The identity must match
        // (engine B is the same model + config as A — that's what makes the spills readable).
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            check(store_b.enabled(), "store B enabled");
            check(store_b.size() == 0, "store B manifest empty before reload");
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 1, "reloaded 1 entry from disk");
            check(store_b.size() == 1, "store B has 1 entry after reload");
            check(store_b.bytes() > 0, "store B has bytes after reload");
            // The manifest's prefix data is what best() matches against — the K/V is gone from RAM.
            // sample() builds the live checkpoint with imgs = {{3, 0x1234567890abcdefull ^ 1}}.
            // The prompt's image list must match for best() to accept the prefix.
            std::vector<int32_t> prompt;
            for (int i = 0; i < 100; ++i) prompt.push_back(int32_t(1000 + i));
            prompt.push_back(9999);   // one more token (the "new message")
            const std::vector<ConversationImageKey> prompt_imgs = {{3, 0x1234567890abcdefull ^ 1}};
            const auto m = store_b.best(prompt, prompt_imgs, false);
            check(m.tokens == 100, "best() found the reloaded entry's 100-token prefix");
            check(m.serial == 100, "best() returned the right serial");
            // take() reads the file and restores the full image (and deletes the file).
            SavedConversation back;
            size_t bytes = 0;
            std::string error;
            SessionStatus status;
            bool ok = session_file_read(rdir.string() + "/conv-100.bin", id, back, bytes, error, {}, &status);
            check(ok, "reloaded file is readable");
            check(back.live.ids.size() == 100, "reloaded image has 100 live ids");
        }
        check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin kept on disk after reload (only take() removes it)");
    }

    // ── 11b. reload_from_disk with multiple spills: all are loaded (no unconditional deletion) ─
    // The fix: when the dir has multiple spills (conversations from multiple sessions, or
    // growing snapshots of the same one), reload_from_disk() validates and loads ALL of them.
    // The disk budget is enforced by evict_oldest() (the real LRU); older spills are NOT
    // unconditionally deleted on startup. With an ample budget, all 3 should survive.
    {
        const auto rdir = dir / "reload_multi";
        // Engine A writes 3 spills with growing serials.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(100)), 100, id), "put serial 100");
            check(store_a.put(std::move(sample(200)), 101, id), "put serial 101");
            check(store_a.put(std::move(sample(300)), 102, id), "put serial 102");
            check(store_a.size() == 3, "store A has 3 entries");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin on disk");
        }
        // Engine B reloads: with an ample budget, ALL 3 are kept and loaded.
        {
            ConversationDiskStore store_b(rdir.string(), 1ull << 30, &id);
            const size_t loaded = store_b.reload_from_disk(id);
            check(loaded == 3, "reloaded 3 entries (all valid spills loaded)");
            check(store_b.size() == 3, "store B has 3 entries after reload");
            check(store_b.bytes() > 0, "store B has bytes after reload");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin kept on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin kept on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin kept on disk");
            // Iteration order is ASCENDING (oldest first), so entry_serial(0) is the OLDEST
            // serial (100), entry_serial(2) is the NEWEST (102). This is what makes evict_oldest()
            // pop the right one under budget pressure (see 11c).
            check(store_b.entry_serial(0) == 100, "manifest[0] is the oldest serial (100)");
            check(store_b.entry_serial(1) == 101, "manifest[1] is serial 101");
            check(store_b.entry_serial(2) == 102, "manifest[2] is the newest serial (102)");
        }
    }

    // ── 11c. reload_from_disk under budget pressure: oldest evicted, newest kept ──────────
    // The disk budget is enforced by evict_oldest() (the real LRU). With 3 spills and a budget
    // that fits only 2, the OLDEST (lowest serial) is evicted — both from the manifest and from
    // disk. This is the regression test for the ascending-iteration order: if anyone inverts
    // it (newest first), the NEWEST conversation would be at the front of entries_ and
    // evict_oldest() would drop the wrong one.
    {
        const auto rdir = dir / "reload_budget";
        // Engine A writes 3 spills with a huge budget, just to populate the dir.
        {
            ConversationDiskStore store_a(rdir.string(), 1ull << 30, &id);
            check(store_a.put(std::move(sample(100)), 100, id), "put serial 100");
            check(store_a.put(std::move(sample(200)), 101, id), "put serial 101");
            check(store_a.put(std::move(sample(300)), 102, id), "put serial 102");
            check(store_a.size() == 3, "store A has 3 entries");
            check(fs::exists(rdir / "conv-100.bin"), "conv-100.bin on disk");
            check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin on disk");
            check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin on disk");
            // Each spill file is ~40427 bytes (sample() with 3 KV layers of 4096+ bytes each).
            // 2 spills = ~80854, 3 spills = ~121281. A budget of 100000 fits 2 but not 3: when
            // the 3rd is loaded, evict_oldest() pops the 1st (serial 100). `loaded` counts the
            // files that were read and pushed into the manifest (3); `size()` counts the
            // survivors (2).
            {
                const uint64_t budget_2 = 100000;   // between 80854 (2 spills) and 121281 (3 spills)
                ConversationDiskStore store_b(rdir.string(), budget_2, &id);
                check(store_b.budget() == budget_2, "store B budget is the small one");
                const size_t loaded = store_b.reload_from_disk(id);
                check(loaded == 3, "reloaded 3 entries (all 3 files were read and pushed)");
                check(store_b.size() == 2, "store B has 2 entries after LRU eviction (oldest evicted)");
                check(store_b.entry_serial(0) == 101, "manifest[0] is serial 101 (oldest kept)");
                check(store_b.entry_serial(1) == 102, "manifest[1] is serial 102 (newest kept)");
                check(!fs::exists(rdir / "conv-100.bin"), "oldest conv-100.bin evicted from disk");
                check(fs::exists(rdir / "conv-101.bin"), "conv-101.bin kept on disk");
                check(fs::exists(rdir / "conv-102.bin"), "conv-102.bin kept on disk");
                check(store_b.drops() >= 1, "drops() counted the LRU eviction");
            }
        }
    }

    // ── 12. reload_from_disk deletes corrupt files, keeps valid ones ─────────────────────
    // A corrupt conv-*.bin is deleted (it can never match this engine); a valid one with a
    // lower serial is loaded into the manifest and KEPT on disk. Under the new contract the
    // valid file is not "superseded" by the corrupt one — both are independent entries.
    {
        const auto rdir = dir / "reload_corrupt";
        ConversationDiskStore store(rdir.string(), 1ull << 30, &id);
        // Write a valid spill with a low serial.
        SavedConversation original = sample(50);
        check(store.put(std::move(original), 200, id), "put valid spill");
        // Drop a corrupt conv-*.bin file with a HIGHER serial.
        std::ofstream(rdir / "conv-300.bin") << "garbage not a real session file";
        check(store.size() == 1, "store has 1 valid entry");
        const size_t loaded = store.reload_from_disk(id);
        check(loaded == 1, "reloaded 1 entry (the valid conv-200.bin)");
        check(store.size() == 1, "manifest has 1 entry after reload");
        check(!fs::exists(rdir / "conv-300.bin"), "corrupt conv-300.bin was deleted");
        check(fs::exists(rdir / "conv-200.bin"), "valid conv-200.bin kept on disk");
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_store_test: %d checks passed\n", checks);
    return 0;
}
